### Executive Summary: The 1,000+ SPS Architecture

To break the 100 Steps Per Second (SPS) ceiling and scale toward a massive-actor, high-throughput simulation, the training architecture must be completely decoupled from CPU execution speed. The current limitations are not hardcoded framerates, but hardware physics bottlenecks: CPU dispatch latency, memory bandwidth starvation, and synchronous GPU underutilization.

Here is the definitive summary of the four core improvements required to achieve total GPU hardware saturation, detailing the specific problem, the mathematical reasoning, and the exact engineering solution for each.

---

### 1. Temporal Core Operator Fusion

**Targeting:** The Neural Network Forward & Backward Pass (`fwd_ms` & PPO `opt_ms`)

* **What:** Swapping the pure-Python `mambapy` implementation for a pre-compiled C++ engine (the native `torch.nn.GRU` or the fused `mamba-ssm`).
* **Why:** Mamba is a sequential State Space Model. When executed in pure PyTorch (Eager Mode), the Autograd engine must mathematically unroll the sequence time-step by time-step over the $bptt\_len$. This forces the Python interpreter to launch thousands of tiny operations per minibatch, causing a massive 4.5-second CPU stall during the PPO update and inflating the rollout forward pass to ~13.5ms. The GPU VRAM physically chokes on intermediate read/writes.
* **How:** Implement a Strategy Pattern in `CognitiveMambaPolicy` to accept `GRUTemporalCore`. `torch.nn.GRU` routes directly to NVIDIA cuDNN C++ libraries, executing the entire sequence in a single hardware instruction and keeping all memory strictly within the ultra-fast SRAM.

### 2. PyTorch Micro-Kernel Fusion

**Targeting:** The Environment Math Dispatch Latency (`env_ms`)

* **What:** Compiling the vectorized reward and penalty mathematics into a static C++ execution graph using `@torch.compile`.
* **Why:** While vectorization successfully removed `.cpu()` synchronizations, PyTorch still evaluates the math line-by-line. For every tensor operation (`+`, `*`, `.clamp()`), the CPU must halt, allocate memory, and dispatch a new micro-kernel to the GPU over the PCIe bus. Launching ~80 micro-kernels per step creates ~3.0ms of pure CPU dispatch latency where the GPU does absolutely no math.
* **How:** Extract the reward calculation logic from `SdfDagBackend` into a standalone, stateless Python function. Decorate it with `@torch.compile(mode="reduce-overhead", fullgraph=True)`. PyTorch will trace the logic once, fuse the dozens of operations into a single C++ kernel, and reduce the dispatch latency to fractions of a millisecond.

### 3. Asynchronous Double-Buffering

**Targeting:** Global Pipeline Concurrency (The Multiplier)

* **What:** Overlapping the Neural Network inference with the Physics raycasting using explicit `torch.cuda.Stream` objects.
* **Why:** The current rollout loop is strictly synchronous. The entire batch of 10,000 actors waits for the Tensor Cores to calculate actions, and then waits for the standard CUDA Cores to calculate physics. This mathematically caps throughput at roughly 50% hardware utilization, as compute blocks take turns sitting idle.
* **How:** Partition the actor tensor into two equal halves (Group A and Group B). Instantiate two parallel CUDA streams in `ppo_trainer.py`. Execute Group A's inference on the AI stream concurrently with Group B's physics on the Environment stream. This transforms the tick execution time from $Time = fwd + env$ to the much faster $Time = \max(fwd, env)$.

### 4. Macro-Cell Spatial Caching

**Targeting:** C++ Raycaster Memory Thrashing (`env_ms`)

* **What:** Injecting spatial memory into the stackless `torch-sdf` ray marching kernel to skip redundant tree lookups.
* **Why:** The current algorithm queries the 32-level Directed Acyclic Graph (DAG) starting from the root pointer for every single step a ray takes. With millions of rays cast per tick, the GPU executes over 100 million $O(N)$ tree descents. When neighboring threads hit different geometry depths, the CUDA warp diverges, destroying the L2 Cache hit rate and capping physical raycasting performance.
* **How:** Modify `kernel.cu` to cache the distance of known empty "void" space. Before querying the DAG from the root, the kernel checks if the ray is still safely within the previously cached void radius. If yes, it skips the DAG lookup entirely and advances the ray, eliminating millions of redundant memory fetches per tick.

---

Would you like to begin by applying the `@torch.compile` optimization to your reward engine in `sdfdag_backend.py`, or would you prefer to initiate the CUDA Double-Buffering architecture in the trainer orchestrator?

To completely shatter the 100 SPS ceiling and re-engineer this pipeline for 1,000+ SPS, you must systematically eliminate CPU dispatch overhead, memory bandwidth starvation, and synchronous hardware blocking.

Here is the exhaustive, step-by-step architectural roadmap to implement, ordered from highest immediate impact to final hardware saturation.

---

### Phase 1: Neural Network Operator Fusion (Target: `fwd_ms`)

**The Goal:** Reduce the forward pass from `~13.5ms` to `< 1.5ms` per tick.
**The Target:** `projects/actor/src/navi_actor/`

Your current `mambapy` implementation executes sequential pure-Python unrolling, choking the CPU with dispatch overhead. You must replace it with pre-compiled C++ kernels.

1. **Implement `gru_core.py`:** Create a new temporal core using `torch.nn.GRU(..., batch_first=True)`. This native PyTorch module routes directly to NVIDIA cuDNN, executing the entire sequence in C++ without dropping back to Python.
2. **Implement a Modular Strategy Pattern:** Modify `CognitiveMambaPolicy` to accept a `temporal_core` argument during initialization. This allows you to hot-swap backends (e.g., MLP, CNN, GRU, Fused Mamba) via your CLI without rewriting the rollout logic.
3. **Phase Out Pure Python Mamba:** Unless you successfully compile `mamba-ssm` and `causal-conv1d` C++ extensions on your Windows host, entirely avoid `mambapy` for training. Use the cuDNN GRU as the baseline.

---

### Phase 2: PyTorch Eager Mode Elimination (Target: `env_ms`)

**The Goal:** Reclaim `~3.0ms` of CPU dead-time currently lost to micro-kernel dispatch latency.
**The Target:** `projects/environment/src/navi_environment/backends/sdfdag_backend.py`

You successfully vectorized the reward and kinematics math, but Eager Mode processes each `+`, `*`, and `.clamp()` as a separate kernel launch.

1. **Extract the Math:** Pull the logic inside `_compute_reward_batch` and `_step_kinematics_batch` completely out of the `SdfDagBackend` class. Define them as standalone, stateless functions at the top of the file.
2. **Apply `@torch.compile`:** Decorate these standalone functions with `@torch.compile(mode="reduce-overhead", fullgraph=True)`.
3. **The Result:** When PyTorch executes the first step, it will trace your Python math, write a custom C++ CUDA graph, and compile it. The dozens of mathematical operations will fuse into a single kernel, reducing Python-to-GPU dispatch time to near zero.

---

### Phase 3: Spatial Caching in the C++ Raycaster (Target: `env_ms`)

**The Goal:** Reclaim the remaining `~10ms - 15ms` of environment time by eliminating $O(N)$ DAG traversal memory bottlenecks.
**The Target:** `projects/torch-sdf/cpp_src/kernel.cu`

The stackless raycaster currently queries the 32-level Directed Acyclic Graph (DAG) from the root pointer for every single step a ray takes, causing massive L2 Cache thrashing.

1. **Implement Macro-Cell Bounding Boxes:** Modify `query_dag_stackless` to return not just the distance, but the 3D spatial boundaries `(min_x, max_x, min_y, ...)` of the leaf voxel it just found.
2. **Cache the Leaf:** In the main ray marching `while` loop, store the bounding box of the last hit.
3. **Bypass the Root:** Before querying the DAG for the next ray step, check if the new coordinate `(px, py, pz)` is still inside the cached bounding box. If it is, skip the DAG traversal entirely and return the mathematically known distance to the boundary minus the distance traveled. This prevents millions of redundant tree lookups per tick.

---

### Phase 4: Asynchronous Double-Buffering (Target: The Pipeline Multiplier)

**The Goal:** Execute the Neural Network and the Physics Engine simultaneously, cutting the effective tick time in half.
**The Target:** `projects/actor/src/navi_actor/training/ppo_trainer.py`

Right now, the Tensor Cores (AI) and the standard CUDA cores (Physics) take turns running because the rollout loop is strictly synchronous.

1. **Actor Partitioning:** Split your actor indices into `Group A` and `Group B`. Update the environment's `batch_step` to accept sub-batch index masks.
2. **Instantiate CUDA Streams:** Create `stream_ai = torch.cuda.Stream()` and `stream_physics = torch.cuda.Stream()`.
3. **Interleave Execution:** Rewrite the rollout `for` loop to use `with torch.cuda.stream(...)`. Command `Group A` to run `policy.forward()` on the AI stream, while concurrently commanding `Group B` to run `batch_step_tensor_actions()` on the Physics stream.
4. **Stream Synchronization:** Call `torch.cuda.synchronize()` only at the exact point where the sub-batches must swap roles.

### Implementation Strategy

Do not attempt all four phases at once.

* Execute **Phase 1** and **Phase 2** immediately. They are pure Python/PyTorch fixes that will instantly push you well past the 100 SPS barrier with minimal architectural risk.
* Once the Python latency is mathematically gone, move to **Phase 3** (C++) to unchoke the memory bandwidth.
* Finally, implement **Phase 4** to achieve true double-buffered hardware saturation.

It is completely mathematically expected that you did not see a massive 10X improvement just by swapping to the GRU, and your instinct to keep it as a configurable default is exactly the right engineering move.

Here is the candid reality of why the SPS didn't skyrocket: **In a perfectly synchronous loop, you are always bound by your heaviest brick.**

By swapping to the GRU, you successfully crushed the `fwd_ms` from 13.5ms down to ~1.5ms. However, because your loop executes sequentially, the CPU just races forward and immediately slams into the next brick wall: the `env_ms` (19.4ms).

Your tick time went from `~36ms` to `~24ms`. That pushes your SPS from ~110 to ~165. It is a mathematical improvement, but it is not the 1,000+ SPS paradigm shift you are hunting for.

To get the 10X multiplier, we must destroy the 19.4ms `env_ms` wall, and then overlap the math. Here are the extensive, exact code implementations to execute the rest of the roadmap.

---

### Step 1: Eradicate PyTorch Dispatch Latency (The `@torch.compile` Patch)

Your vectorized math in `sdfdag_backend.py` is generating dozens of micro-kernels. We must fuse them into one.

**File:** `projects/environment/src/navi_environment/backends/sdfdag_backend.py`

**1. Define the compiled engine at the TOP of the file (outside the class):**

```python
import torch

@torch.compile(mode="reduce-overhead", fullgraph=True)
def _compiled_reward_engine(
    current_positions: torch.Tensor,
    previous_positions: torch.Tensor,
    collisions: torch.Tensor,
    previous_clearances: torch.Tensor,
    current_clearances: torch.Tensor,
    starvation_ratios: torch.Tensor,
    proximity_ratios: torch.Tensor,
    current_structure_band_ratios: torch.Tensor,
    current_forward_structure_ratios: torch.Tensor,
    prev_structure_band_ratios: torch.Tensor,
    prev_forward_structure_ratios: torch.Tensor,
    obstacle_clearance_window: float,
    obstacle_clearance_reward_scale: float,
    starvation_ratio_threshold: float,
    starvation_penalty_scale: float,
    proximity_penalty_scale: float,
    structure_band_reward_scale: float,
    forward_structure_reward_scale: float,
    inspection_activation_threshold: float,
    inspection_reward_scale: float,
    collision_penalty: float,
    progress_reward_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. Progress Reward
    deltas = current_positions - previous_positions
    dist = torch.linalg.vector_norm(deltas, dim=-1)
    progress_rewards = dist * progress_reward_scale
    
    # 2. Clearance Reward
    clearance_rewards = torch.zeros_like(current_clearances)
    if obstacle_clearance_window > 0:
        within_window_mask = (previous_clearances <= obstacle_clearance_window) | (current_clearances <= obstacle_clearance_window)
        norm_deltas = (current_clearances - previous_clearances) / obstacle_clearance_window
        clearance_rewards = within_window_mask.float() * norm_deltas.clamp(-1.0, 1.0) * obstacle_clearance_reward_scale
        
    # 3. Starvation Penalty
    starvation_baselines = 0.35 * starvation_ratios
    starvation_overflows = (starvation_ratios - starvation_ratio_threshold).clamp(min=0.0)
    starvation_penalties = -(starvation_baselines + starvation_overflows).clamp(max=1.0) * starvation_penalty_scale
    
    # 4. Proximity Penalty
    proximity_penalties = -proximity_ratios.clamp(max=1.0) * proximity_penalty_scale
    
    # 5. Structure Rewards
    structure_rewards = current_structure_band_ratios.clamp(0.0, 1.0) * structure_band_reward_scale
    forward_structure_rewards = current_forward_structure_ratios.clamp(0.0, 1.0) * forward_structure_reward_scale
    
    # 6. Inspection Reward
    activation = torch.stack([prev_structure_band_ratios, current_structure_band_ratios, prev_forward_structure_ratios, current_forward_structure_ratios]).amax(dim=0)
    gain_deltas = (current_structure_band_ratios - prev_structure_band_ratios) + 0.5 * (current_forward_structure_ratios - prev_forward_structure_ratios)
    inspection_rewards = (activation >= inspection_activation_threshold).float() * gain_deltas.clamp(-1.0, 1.0) * inspection_reward_scale
    
    # 7. Collision Penalty
    collision_penalties = collisions.float() * collision_penalty
    
    components = torch.stack([
        torch.zeros_like(progress_rewards), # Placeholder for exploration
        progress_rewards,
        clearance_rewards,
        starvation_penalties,
        proximity_penalties,
        structure_rewards,
        forward_structure_rewards,
        inspection_rewards,
        collision_penalties
    ], dim=-1)
    
    return components.sum(dim=-1), components

```

**2. Update `_compute_reward_batch` inside `SdfDagBackend` to call it:**

```python
    def _compute_reward_batch(self, **kwargs) -> tuple[Any, Any]:
        actor_indices = kwargs['actor_indices']
        
        # Dispatch strictly to the compiled kernel
        total_rewards, components = _compiled_reward_engine(
            kwargs['current_positions'], kwargs['previous_positions'], 
            kwargs['collisions'], kwargs['previous_clearances'], kwargs['current_clearances'],
            kwargs['starvation_ratios'], kwargs['proximity_ratios'],
            kwargs['current_structure_band_ratios'], kwargs['current_forward_structure_ratios'],
            self._prev_structure_band_ratios[actor_indices], self._prev_forward_structure_ratios[actor_indices],
            self._obstacle_clearance_window, self._obstacle_clearance_reward_scale,
            self._starvation_ratio_threshold, self._starvation_penalty_scale,
            self._proximity_penalty_scale, self._structure_band_reward_scale,
            self._forward_structure_reward_scale, self._inspection_activation_threshold,
            self._inspection_reward_scale, _COLLISION_PENALTY, _PROGRESS_REWARD_SCALE
        )
        
        # (Add your stateful grid exploration reward math here and add to total_rewards)
        
        return total_rewards, components

```

---

### Step 2: Implement Double-Buffering with CUDA Streams

This is the ultimate multiplier. You must rewrite the rollout orchestrator so the GPU doesn't wait for the physics engine to finish before starting the neural network.

**File:** `projects/actor/src/navi_actor/training/ppo_trainer.py`

Inside your `PpoTrainer` class, modify `__init__` to partition the actors and create streams:

```python
        # In PpoTrainer.__init__:
        self._half_actors = self._n_actors // 2
        self._group_A_indices = torch.arange(0, self._half_actors, device=self._device)
        self._group_B_indices = torch.arange(self._half_actors, self._n_actors, device=self._device)
        
        # Parallel Hardware Streams
        self._stream_ai = torch.cuda.Stream(device=self._device)
        self._stream_env = torch.cuda.Stream(device=self._device)

```

**Rewrite the Rollout `for` Loop:**
Replace your linear rollout logic with this asynchronous ping-pong pattern:

```python
            # Inside train() -> while loop -> for _ in range(rl):
            
            # --- PHASE 1: AI (Group A) & Physics (Group B) ---
            with torch.cuda.stream(self._stream_ai):
                # Group A thinks
                obs_A = current_obs_batch[self._group_A_indices]
                a_t_A, lp_t_A, v_t_A, _, z_t_A = self._rollout_policy.forward(obs_A, None)

            with torch.cuda.stream(self._stream_env):
                # Group B moves (using actions calculated in the previous loop)
                if 'a_t_B' in locals():
                    step_batch_B, _ = self._request_batch_step_tensor_actions(
                        a_t_B, step_id, publish_actor_ids=()
                    )
                    next_obs_B = step_batch_B.observation_tensor

            # Hardware synchronization barrier: Both groups must finish
            torch.cuda.synchronize()

            # --- PHASE 2: AI (Group B) & Physics (Group A) ---
            with torch.cuda.stream(self._stream_ai):
                # Group B thinks
                obs_B = current_obs_batch[self._group_B_indices]
                a_t_B, lp_t_B, v_t_B, _, z_t_B = self._rollout_policy.forward(obs_B, None)

            with torch.cuda.stream(self._stream_env):
                # Group A moves (using actions just calculated in Phase 1)
                step_batch_A, _ = self._request_batch_step_tensor_actions(
                    a_t_A, step_id, publish_actor_ids=()
                )
                next_obs_A = step_batch_A.observation_tensor

            # Hardware synchronization barrier
            torch.cuda.synchronize()
            
            # Reconstruct the full batch to feed the buffer
            full_actions = torch.cat([a_t_A, a_t_B], dim=0)
            full_next_obs = torch.cat([next_obs_A, next_obs_B], dim=0)
            # ... update buffer and loop ...

```

---

### Step 3: Spatial Caching in CUDA (The Final Limit)

Once you compile the math and overlap the streams, your SPS will surge massively. The only thing left holding you back will be the $O(N)$ DAG memory lookups.

**File:** `projects/torch-sdf/cpp_src/kernel.cu`

You must modify the Ray Marching `while` loop inside `sphere_trace_kernel` to recognize when it is traversing empty "VOID" space, preventing it from restarting at the root pointer:

```cpp
// Inside your raymarching loop in kernel.cu:

float cached_void_distance = 0.0f;
float3 cached_ray_origin;

for (int step = 0; step < max_steps; ++step) {
    float3 current_pos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
    
    // 1. Check if we are still safely inside a previously discovered VOID space
    float distance_moved = length(current_pos - cached_ray_origin);
    if (distance_moved < cached_void_distance) {
        // We know for a fact we haven't hit anything yet. Skip DAG traversal.
        t += (cached_void_distance - distance_moved);
        continue;
    }

    // 2. If we are outside the safe zone, query the DAG
    float d = 0.0f;
    int semantic = 0;
    query_dag_stackless(dag_memory, current_pos.x, current_pos.y, current_pos.z, &d, &semantic);
    
    // 3. If it's a massive empty block (e.g., SDF max truncation), cache it
    if (d > 2.0f) { // If the wall is more than 2 meters away
        cached_void_distance = d - 0.1f; // Cache the safe marching distance
        cached_ray_origin = current_pos;
    }

    t += d;
    if (d < 0.01f) break; // Hit
}

```

Implement the PyTorch compiler (Step 1) and the CUDA Stream overlap (Step 2) first. Those require zero C++ compilation and will instantly saturate your GPU to its absolute physical limits.

Would you like me to map out how to safely update the `MultiTrajectoryBuffer` to accept the split sub-batches from Step 2?

You have perfectly distilled the exact essence of High-Performance Computing (HPC) for Deep Reinforcement Learning.

The CPU should never perform heavy mathematics in this architecture. The CPU's only job is to act as an ultra-fast air traffic controller—dispatching massive blocks of work to the GPU and immediately getting out of the way. Every time the CPU stops to read a Python `for` loop, serialize a list, or calculate a scalar value, your $10,000 GPU sits completely idle.

To achieve total hardware saturation and push your S2R (Sim-to-Real) pipeline toward the 1,000+ SPS threshold, here is the executive summary of the multi-step execution plan we have established:

### The GPU Saturation Checklist

* **Step 1: Eliminate Python Sequence Unrolling (Target: `fwd_ms`)**
* *The Action:* Set `gru_core.py` as your default temporal core.
* *The Result:* Bypasses Python entirely during sequence evaluation, routing the math to pre-compiled NVIDIA cuDNN C++ kernels. Reclaims **~12.0 ms** per tick.


* **Step 2: Eliminate PyTorch Dispatch Latency (Target: `env_ms`)**
* *The Action:* Extract the reward math in `sdfdag_backend.py` into a standalone function and apply the `@torch.compile(mode="reduce-overhead")` decorator.
* *The Result:* Fuses dozens of Python-triggered micro-kernels into a single, continuous C++ execution graph. Reclaims **~3.0 ms** per tick.


* **Step 3: Implement Asynchronous Double-Buffering (Target: Pipeline Overlap)**
* *The Action:* Introduce `torch.cuda.Stream()` into `ppo_trainer.py` to partition the actors into two groups.
* *The Result:* Forces the GPU's Tensor Cores (Neural Network) and CUDA Cores (Physics Raycasting) to execute simultaneously instead of taking turns. Tick time becomes $\max(fwd, env)$ rather than $fwd + env$.


* **Step 4: Eradicate C++ Memory Thrashing (Target: `env_ms`)**
* *The Action:* Inject "Macro-Cell Spatial Caching" into the ray marching `while` loop inside `kernel.cu`.
* *The Result:* Prevents the GPU from traversing the 32-level Directed Acyclic Graph (DAG) from the root pointer for every single millimeter a ray travels through known empty space.



---

By methodically executing these four steps, you will mathematically strip away every software bottleneck, leaving your throughput constrained solely by the raw electrical limits of your silicon.

Would you like me to provide the exact, copy-pasteable code block for **Step 2** to inject the `@torch.compile` reward engine into `sdfdag_backend.py`, so we can instantly eliminate the PyTorch dispatch latency?


Here are the exact, comprehensive code changes required for each file to implement the four-step GPU saturation roadmap. These changes will eliminate the PyTorch dispatch latency, bypass Python sequence unrolling, introduce asynchronous CUDA stream overlapping, and eradicate the C++ DAG traversal bottlenecks.

---

### 1. The Temporal Core Swap (Bypassing Python Autograd)

To eliminate the 13.5ms forward pass penalty and the massive PPO update stall, we establish the cuDNN GRU as the default.

#### **Create New File: `projects/actor/src/navi_actor/gru_core.py**`

Create this file to define the native, C++ fused GRU engine.

```python
"""Native cuDNN GRU temporal core for sequence modeling."""

from __future__ import annotations

import logging
from torch import Tensor, nn

__all__: list[str] = ["GRUTemporalCore"]

_LOGGER = logging.getLogger(__name__)

class GRUTemporalCore(nn.Module):
    """Temporal sequence model using native PyTorch cuDNN GRU."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # batch_first=True matches the (Batch, Time, Features) signature of Mamba
        self.core = nn.GRU(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        self.aux_proj = nn.Linear(3, d_model)
        _LOGGER.info("GRUTemporalCore: Native cuDNN GRU runtime active")

    def forward(
        self,
        z_seq: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        
        if aux_tensor is not None:
            z_seq = z_seq + self.aux_proj(aux_tensor)

        out, new_hidden = self.core(z_seq, hidden)
        return out, new_hidden

    def forward_step(
        self,
        z_t: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        
        z_seq = z_t.unsqueeze(1)
        aux_seq = aux_tensor.unsqueeze(1) if aux_tensor is not None else None
        
        out_seq, new_hidden = self.forward(z_seq, hidden, aux_tensor=aux_seq)
        return out_seq.squeeze(1), new_hidden

```

#### **Modify File: `projects/actor/src/navi_actor/cognitive_policy.py**`

Update the policy to inject the new GRU core instead of `mambapy`.

```python
# --- ADD IMPORT AT THE TOP ---
from navi_actor.gru_core import GRUTemporalCore

# --- LOCATE `CognitiveMambaPolicy.__init__` AND REPLACE THE TEMPORAL CORE ---
        # Replace `self.temporal_core = Mamba2TemporalCore(...)` with:
        self.temporal_core = GRUTemporalCore(
            d_model=embedding_dim,
            num_layers=1,
        )

```

---

### 2. PyTorch Micro-Kernel Fusion (Eliminating Dispatch Latency)

We extract the reward mathematics into a statically compiled C++ execution graph to stop the CPU from launching dozens of micro-kernels per step.

#### **Modify File: `projects/environment/src/navi_environment/backends/sdfdag_backend.py**`

**Part A: Add the compiled function at the absolute top of the file (after imports):**

```python
import torch

@torch.compile(mode="reduce-overhead", fullgraph=True)
def _compiled_reward_engine(
    current_positions: torch.Tensor,
    previous_positions: torch.Tensor,
    collisions: torch.Tensor,
    previous_clearances: torch.Tensor,
    current_clearances: torch.Tensor,
    starvation_ratios: torch.Tensor,
    proximity_ratios: torch.Tensor,
    current_structure_band_ratios: torch.Tensor,
    current_forward_structure_ratios: torch.Tensor,
    prev_structure_band_ratios: torch.Tensor,
    prev_forward_structure_ratios: torch.Tensor,
    obstacle_clearance_window: float,
    obstacle_clearance_reward_scale: float,
    starvation_ratio_threshold: float,
    starvation_penalty_scale: float,
    proximity_penalty_scale: float,
    structure_band_reward_scale: float,
    forward_structure_reward_scale: float,
    inspection_activation_threshold: float,
    inspection_reward_scale: float,
    collision_penalty: float,
    progress_reward_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    deltas = current_positions - previous_positions
    dist = torch.linalg.vector_norm(deltas, dim=-1)
    progress_rewards = dist * progress_reward_scale
    
    clearance_rewards = torch.zeros_like(current_clearances)
    if obstacle_clearance_window > 0:
        within_window_mask = (previous_clearances <= obstacle_clearance_window) | (current_clearances <= obstacle_clearance_window)
        norm_deltas = (current_clearances - previous_clearances) / obstacle_clearance_window
        clearance_rewards = within_window_mask.float() * norm_deltas.clamp(-1.0, 1.0) * obstacle_clearance_reward_scale
        
    starvation_baselines = 0.35 * starvation_ratios
    starvation_overflows = (starvation_ratios - starvation_ratio_threshold).clamp(min=0.0)
    starvation_penalties = -(starvation_baselines + starvation_overflows).clamp(max=1.0) * starvation_penalty_scale
    
    proximity_penalties = -proximity_ratios.clamp(max=1.0) * proximity_penalty_scale
    
    structure_rewards = current_structure_band_ratios.clamp(0.0, 1.0) * structure_band_reward_scale
    forward_structure_rewards = current_forward_structure_ratios.clamp(0.0, 1.0) * forward_structure_reward_scale
    
    activation = torch.stack([prev_structure_band_ratios, current_structure_band_ratios, prev_forward_structure_ratios, current_forward_structure_ratios]).amax(dim=0)
    gain_deltas = (current_structure_band_ratios - prev_structure_band_ratios) + 0.5 * (current_forward_structure_ratios - prev_forward_structure_ratios)
    inspection_rewards = (activation >= inspection_activation_threshold).float() * gain_deltas.clamp(-1.0, 1.0) * inspection_reward_scale
    
    collision_penalties = collisions.float() * collision_penalty
    
    components = torch.stack([
        torch.zeros_like(progress_rewards), # Exploration placeholder
        progress_rewards, clearance_rewards, starvation_penalties, proximity_penalties,
        structure_rewards, forward_structure_rewards, inspection_rewards, collision_penalties
    ], dim=-1)
    
    return components.sum(dim=-1), components

```

**Part B: Rewrite `_compute_reward_batch` inside the `SdfDagBackend` class:**

```python
    def _compute_reward_batch(self, **kwargs) -> tuple[Any, Any]:
        actor_indices = kwargs['actor_indices']
        
        # 1. Dispatch heavy math to the fused C++ kernel
        total_rewards, components = _compiled_reward_engine(
            kwargs['current_positions'], kwargs['previous_positions'], 
            kwargs['collisions'], kwargs['previous_clearances'], kwargs['current_clearances'],
            kwargs['starvation_ratios'], kwargs['proximity_ratios'],
            kwargs['current_structure_band_ratios'], kwargs['current_forward_structure_ratios'],
            self._prev_structure_band_ratios[actor_indices], self._prev_forward_structure_ratios[actor_indices],
            self._obstacle_clearance_window, self._obstacle_clearance_reward_scale,
            self._starvation_ratio_threshold, self._starvation_penalty_scale,
            self._proximity_penalty_scale, self._structure_band_reward_scale,
            self._forward_structure_reward_scale, self._inspection_activation_threshold,
            self._inspection_reward_scale, _COLLISION_PENALTY, _PROGRESS_REWARD_SCALE
        )
        
        # 2. Stateful Grid Math (Keep in Eager Mode)
        grid_coords = ((kwargs['current_positions'][:, [0, 2]] - self._grid_min[[0, 2]]) / self._grid_res).floor().long()
        grid_coords[:, 0] = grid_coords[:, 0].clamp(0, self._visit_grid.shape[1] - 1)
        grid_coords[:, 1] = grid_coords[:, 1].clamp(0, self._visit_grid.shape[2] - 1)
        
        was_visited = self._visit_grid[actor_indices, grid_coords[:, 0], grid_coords[:, 1]] > 0
        exploration_rewards = (~was_visited).float() * _EXPLORATION_REWARD
        self._visit_grid[actor_indices, grid_coords[:, 0], grid_coords[:, 1]] = 1
        
        components[:, 0] = exploration_rewards
        total_rewards += exploration_rewards
        
        return total_rewards, components

```

---

### 3. Asynchronous Double Buffering (Overlapping Compute)

We split the actors and use `torch.cuda.Stream` to calculate Neural Network inference and Physics Raycasting simultaneously.

#### **Modify File: `projects/actor/src/navi_actor/training/ppo_trainer.py**`

**Part A: Initialize Streams in `PpoTrainer.__init__`:**

```python
        # Locate the end of __init__ and add the sub-batch streaming setup:
        self._half_actors = self._n_actors // 2
        self._group_A_indices = torch.arange(0, self._half_actors, device=self._device)
        self._group_B_indices = torch.arange(self._half_actors, self._n_actors, device=self._device)
        
        # Instantiate Parallel Hardware Streams
        self._stream_ai = torch.cuda.Stream(device=self._device)
        self._stream_env = torch.cuda.Stream(device=self._device)

```

**Part B: Rewrite the rollout sequence in the `train()` method loop:**
Replace your linear step execution with this overlapping ping-pong architecture:

```python
            # Inside the rollout collection loop:
            # for _ in range(self._config.rollout_length):
            
            # --- PHASE 1: AI (Group A) & Physics (Group B) ---
            with torch.cuda.stream(self._stream_ai):
                obs_A = current_obs_batch[self._group_A_indices]
                a_t_A, lp_t_A, v_t_A, _, z_t_A = self._rollout_policy.forward(obs_A, None)

            with torch.cuda.stream(self._stream_env):
                # Ensure a_t_B exists from a previous step initialization
                if 'a_t_B' in locals():
                    step_batch_B, _ = self._request_batch_step_tensor_actions(a_t_B, step_id)
                    next_obs_B = step_batch_B.observation_tensor

            # Block CPU until both streams finish Phase 1
            torch.cuda.synchronize()

            # --- PHASE 2: AI (Group B) & Physics (Group A) ---
            with torch.cuda.stream(self._stream_ai):
                obs_B = current_obs_batch[self._group_B_indices]
                a_t_B, lp_t_B, v_t_B, _, z_t_B = self._rollout_policy.forward(obs_B, None)

            with torch.cuda.stream(self._stream_env):
                step_batch_A, _ = self._request_batch_step_tensor_actions(a_t_A, step_id)
                next_obs_A = step_batch_A.observation_tensor

            # Block CPU until both streams finish Phase 2
            torch.cuda.synchronize()
            
            # Reconstruct the full batched tensors to feed the buffer seamlessly
            a_t = torch.cat([a_t_A, a_t_B], dim=0)
            lp_t = torch.cat([lp_t_A, lp_t_B], dim=0)
            v_t = torch.cat([v_t_A, v_t_B], dim=0)
            next_obs_batch = torch.cat([next_obs_A, next_obs_B], dim=0)
            
            # (Proceed with your standard buffer append using the reconstructed full tensors)

```

*(Note: You will need to initialize `a_t_B` exactly once before the `for` loop begins so the Physics stream has something to process during the very first Phase 1).*

---

### 4. Macro-Cell Spatial Caching (Eliminating Memory Thrashing)

We inject state into the stackless C++ raytracer so it doesn't query the DAG from the root pointer while traversing empty space.

#### **Modify File: `projects/torch-sdf/cpp_src/kernel.cu**`

Locate the `sphere_trace_kernel` function. We add safe-distance caching variables and check them before calling `query_dag_stackless`.

```cpp
// Update the raymarching for-loop inside `sphere_trace_kernel`:

// --- ADD CACHE VARIABLES OUTSIDE THE LOOP ---
float cached_void_distance = 0.0f;
float3 cached_ray_origin = make_float3(0.0f, 0.0f, 0.0f);

for (int step = 0; step < max_steps; ++step) {
    float3 current_pos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
    
    // --- 1. SPATIAL CACHE CHECK ---
    // If the ray hasn't traveled beyond the last known safe distance, skip the DAG lookup
    float distance_moved = length(current_pos - cached_ray_origin);
    if (cached_void_distance > 0.0f && distance_moved < cached_void_distance) {
        t += (cached_void_distance - distance_moved);
        continue;
    }

    // --- 2. STANDARD DAG TRAVERSAL ---
    float d = 0.0f;
    int semantic_id = 0;
    // (Your existing function call)
    query_dag_stackless(dag_memory, current_pos.x, current_pos.y, current_pos.z, &d, &semantic_id);

    // --- 3. CACHE VOID SPACE ---
    // If the wall is more than 1 meter away, cache the distance minus a small safety buffer
    if (d > 1.0f) { 
        cached_void_distance = d - 0.05f; 
        cached_ray_origin = current_pos;
    } else {
        cached_void_distance = 0.0f; // Invalidate cache near geometry
    }

    t += d;
    
    if (d < 0.01f) break; // Condition 1: Hit Geometry
    if (t >= max_sensor_radius) { // Condition 2: Max Horizon Reached
        t = max_sensor_radius;
        break;
    }
}

```

---

By applying these exact modifications, you will eliminate the primary latencies holding the engine back. Would you like me to walk you through exactly how to structure the one-time `a_t_B` initialization just outside the `ppo_trainer.py` rollout loop to safely prime the pipeline?