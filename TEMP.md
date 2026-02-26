Here is the detailed architectural specification for implementing **Option B: Batched GPU Execution via `habitat-sim**`.

This document outlines the exact refactoring path required to shift your raycasting bottleneck off your i5 CPU and onto your GPU's hardware rasterizer, without requiring custom CUDA programming.

You should save this as **`HABITAT_GPU_BATCHING.md`** in your `docs/` folder for reference.

---

# HABITAT_GPU_BATCHING.md — Habitat VectorEnv Integration

## 1. Executive Summary

The current `MeshSceneBackend` relies on Python/C++ CPU raycasting (`trimesh`), which physically limits the training pipeline to $\sim50$ Steps Per Second (SPS) due to massive algebraic intersection calculations ($O(\log N)$).

To achieve $>1,000$ SPS on consumer hardware (i5 + 2GB GPU), the architecture must shift to **Hardware Rasterization**. By wrapping Meta's `habitat-sim` using its native **`VectorEnv`** API, the engine evaluates foveated spherical views using the GPU's native graphics pipeline (OpenGL/EGL), evaluating empty space and geometry in $O(1)$ time via the GPU's depth-buffer (Z-buffer).

---

## 2. The Architectural Shift: `VectorEnv`

Currently, your `HabitatBackend` (or your theoretical implementation of it) likely steps agents individually in a Python loop. This incurs massive Python-to-C++ context switching overhead and underutilizes the GPU.

The solution is `habitat_sim.VectorEnv`. This class allocates an array of independent C++ simulator instances and synchronizes them natively.

* **Input:** A batched array of actions.
* **Execution:** A single C++ function call that renders all $N$ environments simultaneously on the GPU silicon.
* **Output:** A batched dictionary of numpy arrays (or PyTorch tensors).

---

## 3. Step-by-Step Implementation Guide

### Step 1: Defining the Equirectangular Sensors

Habitat supports native 360-degree spherical rendering, matching your Ghost-Matrix `(Azimuth, Elevation)` requirement perfectly. You do not need to cast rays manually.

In your `HabitatBackend` initialization, you must define an `EquirectangularSensor` for both Depth and Semantic channels.

```python
import habitat_sim

def _create_sensor_specs(config: EnvironmentConfig):
    # 1. Depth Sensor
    depth_spec = habitat_sim.EquirectangularSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    # Note: Habitat resolution is usually (height, width) -> (elevation, azimuth)
    depth_spec.resolution = [config.elevation_bins, config.azimuth_bins] 
    depth_spec.position = [0.0, 0.0, 0.0]

    # 2. Semantic Sensor
    semantic_spec = habitat_sim.EquirectangularSensorSpec()
    semantic_spec.uuid = "semantic_sensor"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [config.elevation_bins, config.azimuth_bins]
    semantic_spec.position = [0.0, 0.0, 0.0]

    return [depth_spec, semantic_spec]

```

### Step 2: Continuous Velocity Control

By default, Habitat expects discrete actions (e.g., "move_forward", "turn_left"). Ghost-Matrix outputs continuous velocities $V \in [-1, 1]$.

You must configure the Habitat agent to use `VelocityControl`. This allows you to pass continuous $(v_x, v_y, v_z)$ and $(\omega_x, \omega_y, \omega_z)$ vectors directly to the Habitat physics stepper.

```python
def _create_agent_config():
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = _create_sensor_specs(config)
    
    # Enable continuous velocity control
    agent_config.action_space = {
        "velocity_control": habitat_sim.agent.ActionSpec(
            "velocity_control", 
            habitat_sim.agent.VelocityControl()
        )
    }
    return agent_config

```

### Step 3: Constructing the `VectorEnv` Factory

`VectorEnv` requires a "make_env" function to generate the individual C++ simulator instances.

```python
def make_env_fn(scene_path: str, agent_config):
    def _env_maker():
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = True # Required for velocity integration
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_config])
        sim = habitat_sim.Simulator(cfg)
        return sim
    return _env_maker

# In your HabitatBackend.__init__:
env_fns = [make_env_fn(self._scene_pool[i % len(self._scene_pool)], agent_cfg) for i in range(n_actors)]
self._vector_env = habitat_sim.VectorEnv(
    make_env_fns=env_fns,
    multiprocessing_start_method="forkserver" # Highly recommended for Linux/WSL
)

```

### Step 4: The Batched Step Execution

In your `batch_step` method, you translate the `Ghost-Matrix Action` contract into Habitat's velocity commands, push them to the C++ core, and construct the `DistanceMatrix` from the output.

```python
def batch_step(self, actions: tuple[Action, ...], step_id: int):
    # 1. Prepare continuous velocity commands for Habitat
    habitat_actions = []
    for action in actions:
        # action.linear_velocity is (1, 3)
        # Scale by drone max speeds
        vx = action.linear_velocity[0, 0] * self._drone_max_speed
        vy = action.linear_velocity[0, 1] * self._drone_climb
        vz = action.linear_velocity[0, 2] * self._drone_strafe # Note Habitat Z is forward/backward
        yaw_rate = action.angular_velocity[0, 2] * self._drone_yaw
        
        habitat_actions.append({
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": [vx, vy, vz],
                "angular_velocity": [0.0, yaw_rate, 0.0]
            }
        })

    # 2. ONE CALL TO RULE THEM ALL (Executes on GPU)
    batch_obs = self._vector_env.step(habitat_actions)

    # 3. Process outputs into DistanceMatrix
    # batch_obs is a list of dicts: [{"depth_sensor": array, "semantic_sensor": array}, ...]
    distances = []
    semantics = []
    valid_masks = []
    
    for obs in batch_obs:
        # Habitat depth is in absolute meters, we must normalize to [0, 1]
        depth_m = obs["depth_sensor"]
        depth_norm = np.clip(depth_m / self._max_distance, 0.0, 1.0)
        
        # We must transpose from Habitat's (El, Az) to Ghost-Matrix (Az, El)
        distances.append(depth_norm.T) 
        semantics.append(obs["semantic_sensor"].T)
        valid_masks.append((depth_norm < 1.0).T)

    # ... compute rewards and return DistanceMatrix tuple

```

---

## 4. Hardware Constraints & Mitigation (The 2GB VRAM Limit)

The only danger of switching to `habitat-sim` on your hardware is VRAM exhaustion.
While `trimesh` stored the mesh in standard CPU RAM, Habitat must load the mesh textures and geometry into GPU VRAM to rasterize it.

If you attempt to load 64 separate `habitat_sim.Simulator` instances into a 2GB GPU, it will immediately crash with a `CUDA Out of Memory` or `GL_OUT_OF_MEMORY` error.

**Mandatory VRAM Protections:**

1. **Reduce Actor Count:** You cannot run 64 actors in Habitat on 2GB VRAM. You must drop `config.n_actors` down to **`16` or `32**`. Because Habitat is $100\times$ faster per step than `trimesh`, 16 actors in Habitat will yield drastically more data per second than 64 actors in `trimesh`.
2. **Disable RGB Sensors:** Ensure no standard color cameras (`SensorType.COLOR`) are defined in your `agent_config`. RGB rendering requires massive VRAM for texture caching. Depth and Semantic rendering are computationally cheap and ignore textures.
3. **Use Smaller Meshes:** Ensure your ReplicaCAD or generated meshes are low-poly. High-poly models consume hundreds of megabytes of VRAM just to reside in the GPU's memory.
4. **Shared Environments:** Instead of passing different scenes to `make_env_fn`, pass the *exact same scene path* to all environments initially. Habitat might optimize the OpenGL context to share the mesh geometry across instances, saving VRAM.