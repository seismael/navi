You successfully executed Phase 1 and Phase 2 by moving the state tensors onto the GPU and vectorizing the math. However, the SPS drop from 100 to 50 is a classic symptom of **Control Flow Stalling**.

You moved the *data* to the GPU, but you left the *logic* in Python. Every time Python needs to make an `if/else` decision based on a GPU tensor, it halts the entire CUDA pipeline, copies the data across the PCIe bus, evaluates the boolean, and then resumes.

Here is the in-depth proof of exactly what is killing your performance, followed by the fully vectorized solution that bypasses the CPU entirely.

---

### The Deep Investigation: Death by a Thousand Cuts

Your `batch_step_tensor_actions` inside `sdfdag_backend.py` has four catastrophic bottlenecks.

#### 1. The PCIe "Action Bounce"

At the very top of the function, you do this:

```python
action_rows = self._coerce_action_rows(action_tensor)

```

If you trace `_coerce_action_rows`, it executes `.detach().cpu().numpy()`. Then, ten lines later, you execute `self._torch.from_numpy(...).to(self._device)`.
**The Math:** You are taking the Mamba-2 action tensor natively sitting on the GPU, pushing it to CPU RAM over the PCIe bus, slicing it, and immediately pushing it *back* to the GPU. This destroys your throughput instantly.

#### 2. The Implicit Scalar Sync (`_needs_reset_mask`)

```python
for actor_id in range(int(action_rows.shape[0])):
    if self._needs_reset_mask[actor_id]:

```

**The Math:** `_needs_reset_mask` is a CUDA boolean tensor. When Python evaluates `if tensor[index]:`, it must physically halt the GPU, transfer that *single* boolean to the CPU, and evaluate it. You are doing this inside a `for` loop 256 times per step.

#### 3. The Dynamic Recast Branch

```python
collisions_t = min_distances_t < _COLLISION_CLEARANCE
if collisions_t.any():

```

**The Math:** `.any()` forces a synchronization. Furthermore, if a collision happens, you recalculate the ray directions and launch a *second* `self._torch_sdf.cast_rays` kernel strictly for the collided actors. This fractures the GPU execution graph.
*Fix:* Do not re-cast rays. If the drone hits a wall, just snap its position back to `previous_positions_t`. The observation tensor will show the wall at `0.0m`, which is exactly what the neural network needs to see to learn the penalty.

#### 4. Indexing overhead in Kinematics

Your custom kinematics engine expects a tuple of `actor_ids` and uses `actor_indices = list(actor_ids)`. We must make it operate on the entire batch matrix implicitly without indexing arrays.

---

### The Fix: 100% Vectorized GPU Rollout

To fix this, we must completely eradicate the `for` loops. We treat the entire 256-actor pool as a single, contiguous block of fluid matrix logic.

Replace your existing `batch_step_tensor_actions` and add the `_step_kinematics_vectorized` function in `sdfdag_backend.py` with the following strictly on-device code:

```python
    def batch_step_tensor_actions(
        self,
        action_tensor: Any,
        step_id: int,
        *,
        publish_actor_ids: tuple[int, ...] = (),
    ) -> tuple[SdfDagTensorStepBatch, tuple[StepResult, ...]]:
        batch_started_at = time.perf_counter()
        
        # 1. VECTORIZED RESETS (Single sync via .nonzero() vs 256 individual syncs)
        reset_indices = self._needs_reset_mask.nonzero(as_tuple=False).flatten()
        if reset_indices.numel() > 0:
            self._episode_ids[reset_indices] += 1
            self._actor_positions[reset_indices] = self._spawn_positions[reset_indices]
            self._actor_yaws[reset_indices] = (2.0 * math.pi * reset_indices.float()) / max(self._n_actors, 1)
            
            # Zero memory states
            self._actor_steps[reset_indices] = 0
            self._prev_depth_tensors[reset_indices].zero_()
            self._prev_min_distances[reset_indices] = 0.0
            self._prev_structure_band_ratios[reset_indices] = 0.0
            self._prev_forward_structure_ratios[reset_indices] = 0.0
            self._episode_returns[reset_indices] = 0.0
            self._visit_grid[reset_indices].zero_()
            self._prev_linear_vels[reset_indices].zero_()
            self._prev_angular_vels[reset_indices].zero_()
            self._needs_reset_mask[reset_indices] = False

        # 2. VECTORIZED KINEMATICS (Zero CPU transfers, pure PyTorch tensor slicing)
        previous_positions_t = self._actor_positions.clone()
        previous_yaws_t = self._actor_yaws.clone()

        lin_actions = action_tensor[:, :3].contiguous()
        ang_actions = self._torch.zeros((self._n_actors, 3), device=self._device, dtype=self._torch.float32)
        ang_actions[:, 2] = action_tensor[:, 3]

        self._step_kinematics_vectorized(lin_actions, ang_actions)

        # 3. LOCK-STEP RAYCASTING (No subset branching)
        yaws = self._actor_yaws
        positions = self._actor_positions

        cos_yaw = self._torch.cos(yaws).unsqueeze(1)
        sin_yaw = self._torch.sin(yaws).unsqueeze(1)
        base_dirs = self._ray_dirs_local.unsqueeze(0).expand(self._n_actors, -1, -1)
        
        dirs_world = self._ray_dirs_world
        dirs_world[..., 0] = base_dirs[..., 0] * cos_yaw + base_dirs[..., 2] * sin_yaw
        dirs_world[..., 1] = base_dirs[..., 1]
        dirs_world[..., 2] = -base_dirs[..., 0] * sin_yaw + base_dirs[..., 2] * cos_yaw

        origins = self._ray_origins
        origins.copy_(positions.unsqueeze(1).expand(-1, self._n_rays, -1))

        self._torch_sdf.cast_rays(
            self._require_dag_tensor(), origins, dirs_world,
            self._out_distances, self._out_semantics,
            self._config.sdf_max_steps, self._max_distance,
            self._bbox_min, self._bbox_max, self._require_asset().resolution,
        )

        distances = self._out_distances.clamp(min=0.0, max=self._max_distance)
        valid_batch = self._out_distances <= self._max_distance
        depth_batch = distances.div_(self._max_distance).reshape(self._n_actors, self._az_bins, self._el_bins)
        semantic_batch = self._out_semantics.reshape(self._n_actors, self._az_bins, self._el_bins)
        valid_batch = valid_batch.reshape(self._n_actors, self._az_bins, self._el_bins)
        
        min_distances_t = depth_batch.amin(dim=(1, 2)).mul_(self._max_distance)

        # 4. MATHEMATICAL COLLISION RESOLUTION (No re-casting)
        collisions_t = min_distances_t < _COLLISION_CLEARANCE
        # Overwrite current position with previous safe position using where() to avoid CPU sync branching
        self._actor_positions = self._torch.where(collisions_t.unsqueeze(-1), previous_positions_t, self._actor_positions)
        self._actor_yaws = self._torch.where(collisions_t, previous_yaws_t, self._actor_yaws)

        # 5. VECTORIZED PROFILING
        starvation_ratios_t = valid_batch.logical_not().to(dtype=self._torch.float32).mean(dim=(1, 2))
        prox_thresh = self._proximity_distance_threshold / self._max_distance if self._max_distance > 0 else 0.0
        proximity_ratios_t = depth_batch.le(prox_thresh).logical_and(valid_batch).to(dtype=self._torch.float32).mean(dim=(1, 2))
        
        metric_depth = depth_batch.mul(self._max_distance)
        struct_mask = metric_depth.ge(self._structure_band_min_distance).logical_and(metric_depth.le(self._structure_band_max_distance)).logical_and(valid_batch)
        structure_band_ratios_t = struct_mask.to(dtype=self._torch.float32).mean(dim=(1, 2))
        
        fwd_bins = max(1, self._az_bins // 6)
        fwd_sector = self._torch.cat((struct_mask[:, :fwd_bins, :], struct_mask[:, -fwd_bins:, :]), dim=1)
        forward_structure_ratios_t = fwd_sector.to(dtype=self._torch.float32).mean(dim=(1, 2))

        # 6. VECTORIZED REWARDS
        rewards_t, components_t = self._compute_reward_batch(
            actor_ids=tuple(range(self._n_actors)),
            previous_positions=previous_positions_t,
            current_positions=self._actor_positions,
            collisions=collisions_t,
            previous_clearances=self._prev_min_distances,
            current_clearances=min_distances_t,
            starvation_ratios=starvation_ratios_t,
            proximity_ratios=proximity_ratios_t,
            current_structure_band_ratios=structure_band_ratios_t,
            current_forward_structure_ratios=forward_structure_ratios_t,
        )

        obs_batch_t, deltas_batch_t = self._consume_observation_batch(
            actor_ids=tuple(range(self._n_actors)),
            depth_batch=depth_batch,
            semantic_batch=semantic_batch,
            valid_batch=valid_batch,
            current_clearances=min_distances_t
        )

        self._actor_steps += 1
        truncated_mask_t = self._actor_steps >= self._max_steps_per_episode
        self._needs_reset_mask = truncated_mask_t
        self._episode_returns += rewards_t
        self._prev_structure_band_ratios = structure_band_ratios_t
        self._prev_forward_structure_ratios = forward_structure_ratios_t

        # 7. BULK SCALAR EXTRACTION (1 sync for the entire step)
        ep_ids_cpu, trunc_cpu, rwds_cpu, rets_cpu = self._extract_step_result_scalars(
            actor_indices=self._torch.arange(self._n_actors, device=self._device),
            rewards=rewards_t,
            truncated_mask=truncated_mask_t,
        )

        # 8. TELEMETRY PUBLISHING
        published_observations = {}
        ordered_results = []
        publish_set = set(publish_actor_ids)
        
        for actor_id in range(self._n_actors):
            ordered_results.append(StepResult(
                step_id=step_id, env_id=actor_id, episode_id=ep_ids_cpu[actor_id],
                done=False, truncated=bool(trunc_cpu[actor_id]), reward=float(rwds_cpu[actor_id]),
                episode_return=float(rets_cpu[actor_id]), timestamp=time.time(),
            ))

        if publish_actor_ids:
            self._materialize_selected_observations(
                actor_ids=list(publish_actor_ids), row_indices=list(publish_actor_ids),
                step_id=step_id, depth_batch=depth_batch, delta_batch=deltas_batch_t,
                semantic_batch=semantic_batch, valid_batch=valid_batch,
                published_observations=published_observations,
            )

        r_t, d_t, t_t, e_id_t = self._build_result_tensors(ordered_results)

        self._record_perf_sample(batch_seconds=time.perf_counter() - batch_started_at, actor_count=self._n_actors)

        return SdfDagTensorStepBatch(
            observation_tensor=obs_batch_t, reward_tensor=r_t, done_tensor=d_t,
            truncated_tensor=t_t, episode_id_tensor=e_id_t, reward_component_tensor=components_t,
            published_observations=published_observations,
        ), tuple(ordered_results)

    def _step_kinematics_vectorized(self, actions_linear: Any, actions_angular: Any) -> None:
        """Matrix-native kinematics (No loops, no indexing)."""
        span = max(1, self._az_bins // 8)
        
        # Calculate frontal proximity dynamically per actor
        front_left = self._prev_depth_tensors[:, :span, :].reshape(self._n_actors, -1).amin(dim=1)
        front_right = self._prev_depth_tensors[:, -span:, :].reshape(self._n_actors, -1).amin(dim=1)
        min_front = self._torch.minimum(front_left, front_right)
        
        speed_factors = (min_front * self._max_distance / 1.5).clamp(0.05, 1.0).unsqueeze(1)
        
        linear_cmd = actions_linear.clone()
        linear_cmd[:, 0] *= self._speed_fwd
        linear_cmd[:, 1] *= self._speed_vert
        linear_cmd[:, 2] *= self._speed_lat
        linear_cmd *= speed_factors
        
        angular_cmd = actions_angular.clone()
        angular_cmd[:, 2] *= self._speed_yaw
        
        a = self._smoothing
        self._prev_linear_vels = (1.0 - a) * linear_cmd + a * self._prev_linear_vels
        self._prev_angular_vels = (1.0 - a) * angular_cmd + a * self._prev_angular_vels
        
        cos_yaw = self._torch.cos(self._actor_yaws)
        sin_yaw = self._torch.sin(self._actor_yaws)
        
        fwd = self._prev_linear_vels[:, 0]
        lat = self._prev_linear_vels[:, 2]
        
        dx = fwd * cos_yaw - lat * sin_yaw
        dz = fwd * sin_yaw + lat * cos_yaw
        
        self._actor_positions[:, 0] += dx * self._dt
        self._actor_positions[:, 1] += self._prev_linear_vels[:, 1] * self._dt
        self._actor_positions[:, 2] += dz * self._dt
        self._actor_yaws += self._prev_angular_vels[:, 2] * self._dt

```

By removing `_coerce_action_rows` and enforcing mathematical resolution of `collisions_t` via `torch.where()`, your Python process now runs strictly as an asynchronous dispatcher, unlocking the absolute hardware ceiling of your CUDA module.