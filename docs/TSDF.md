## Fixed Horizon Policy

This file records the implementation verdict for sensor horizon handling in the
canonical `sdfdag` runtime.

### Verdict

Use a **fixed configured horizon** for the full duration of a run.

- Do not use a dynamic per-step or per-actor runtime radius.
- The canonical horizon is `EnvironmentConfig.max_distance`.
- `projects/environment` and `projects/torch-sdf` must share this single value
    for tracing, clamping, validity, and observation normalization.

### Why This Is Canonical

1. Performance:
     - a fixed horizon keeps CUDA ray batches on one contract and avoids dynamic
         runtime branching in the hot path
     - the `torch_sdf.cast_rays()` call receives one explicit tracing horizon for
         the whole batch
2. Correctness:
     - normalized `DistanceMatrix.depth` remains tied to one divisor
     - validity semantics stay stable: values beyond the configured horizon are
         treated as saturated / invalid consistently
3. Engineering quality:
     - one horizon contract is easier to benchmark, test, and reason about than
         dynamic radius behavior

### Current Repository Status

The canonical repository already implements the accepted part of this design:

- `projects/environment/src/navi_environment/backends/sdfdag_backend.py`
    passes `EnvironmentConfig.max_distance` into `torch_sdf.cast_rays()`
- returned distances are clamped to that horizon before normalization
- validity is derived from the same horizon
- starvation / proximity shaping is computed from the resulting batched depth
    and validity tensors

### Non-Canonical Experiments

The following ideas remain benchmark-gated experiments only and are not part of
the production implementation unless they beat the current path on throughput,
load cost, and file size:

- compiler-side TSDF truncation metadata
- truncated-distance storage in `.gmdag`
- Morton / Z-order DAG layout changes

### Implementation Rule

If the horizon needs to change, change it through configuration before the run.
Do not adapt it dynamically inside the runtime sphere-tracing loop.

## Canonical Status Notes

Parts of this file preserve imported design rationale. The current canonical
repository status is:

- production observation default is `256x48`, not `128x24`
- the active production temporal-core default is `mamba2` (pure-PyTorch Mamba-2 SSD), not fused Mamba-2
- high-resolution trainer scaling is currently limited by actor-side RayViT
    attention before the environment runtime itself hits the same wall on the
    active MX150 machine
- information-foraging and fixed-horizon ideas below remain valid, but current
    performance conclusions must be cross-checked against `docs/PERFORMANCE.md`
    and `docs/ACTOR.md`


## Information Foraging

You have correctly identified a critical vulnerability in Partially Observable Markov Decision Processes (POMDPs). In autonomous robotics, this exact phenomenon is formally known as **Sensor Starvation** or **Perceptual Aliasing**.

Your intuition about the blind person is precisely how real-world Simultaneous Localization and Mapping (SLAM) and Visual Inertial Odometry (VIO) systems operate. If a drone flies into a massive empty hall where all walls exceed its 15-meter sensor radius, its depth matrix becomes a flat uniform array of `[15.0, 15.0, ..., 15.0]`.

Mathematically, the optical flow drops to zero. The selected temporal core loses the ability to update its hidden state because it has no reference points to calculate its own velocity. In the real world, a drone in this state will inevitably succumb to IMU drift and crash.

To engineer the agent to actively avoid the "abyss" and forage for structural geometry, you must implement **Information Foraging** directly into your reward function.

Here is the architectural breakdown of how to structure this reward tensor in PyTorch without degrading your $>100$ SPS throughput.

---

### The Objective: The "Sweet Spot" Navigation Profile

The reward function must balance three conflicting physical constraints:

1. **Too Close ($< 1.0$m):** Imminent collision risk (Hard Penalty).
2. **Too Far ($= 15.0$m):** Sensor starvation and IMU drift risk (Soft Penalty).
3. **The Sweet Spot ($2.0$m - $10.0$m):** High information density, safe reaction time (Positive Reward).

### Implementing the Information Foraging Reward

Instead of a simple sparse reward (e.g., $+1$ for reaching a goal, $-1$ for crashing), you must apply a dense, continuous shaping reward computed directly from the TSDF `DistanceMatrix` at every step.

Here is the vectorized mathematical formulation for your PyTorch `step()` function:

#### 1. The Starvation Penalty (The Abyss)

You must penalize the agent if the percentage of "maxed out" rays exceeds a critical threshold.

Let $D$ be your flattened distance tensor of size $N$ (where $N = Az \times El$;
for the canonical production contract, $N = 256 \times 48 = 12288$).
Let $\tau$ be your truncation radius ($15.0$).

$$\text{Starvation Ratio } (\rho) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(D_i \ge \tau)$$

If $\rho \to 1.0$, the agent is blind.

#### 2. The Vectorized PyTorch Implementation

To keep this fully vectorized and bounded-cost on the GPU, avoid Python
`if/else` loops. Use native PyTorch tensor operations.

```python
import torch

def compute_foraging_reward(distance_matrix: torch.Tensor, max_radius: float = 15.0) -> torch.Tensor:
    """
    Computes the dense reward for a batch of agents based on sensor starvation.
    Args:
        distance_matrix: Tensor of shape (batch_size, 128, 24)
    """
    batch_size = distance_matrix.shape[0]
    
    # 1. Calculate the starvation ratio per agent
    # How many rays are hitting the absolute void?
    starved_rays = (distance_matrix >= (max_radius - 0.1)).float()
    starvation_ratio = starved_rays.mean(dim=(1, 2)) # Shape: (batch_size,)
    
    # 2. Define the safe threshold (e.g., at least 20% of rays must see geometry)
    max_allowed_starvation = 0.80 
    
    # 3. Calculate the penalty (Continuous and differentiable)
    # If ratio > 0.80, penalty grows linearly. If < 0.80, penalty is 0.
    starvation_penalty = torch.relu(starvation_ratio - max_allowed_starvation) * -2.0
    
    # 4. The Wall-Hugging Penalty (Optional but recommended)
    # Penalize if too many rays are critically close (< 1.0m) to prevent 
    # the agent from just scraping along walls to maximize "information".
    critical_rays = (distance_matrix < 1.0).float()
    proximity_ratio = critical_rays.mean(dim=(1, 2))
    proximity_penalty = proximity_ratio * -5.0
    
    # Return the shaped reward to be added to the global step reward
    return starvation_penalty + proximity_penalty

```

### The Behavioral Outcome

By injecting this tensor math into your PPO environment:

1. **Emergent Wall-Following:** If the drone is spawned in a massive open hangar, it will learn to immediately fly toward the nearest wall or pillar and navigate alongside it, keeping it within the $10$-meter peripheral vision.
2. **Cornering:** When approaching a blind corner, the agent will learn to swing wide. If it hugs the inside corner, its `proximity_penalty` spikes. If it flies entirely into the open room, its `starvation_penalty` spikes. It will naturally find the geometric center-line.

For the canonical Ghost-Matrix trainer, this base foraging logic is extended with two low-cost observation-derived terms:

1. **Structure-Band Reward:** positive weight on valid mid-range geometry so the agent prefers navigable interiors and corridors over staring into empty halls.
2. **Inspection Reward:** positive reward only when a turn or slight repositioning increases visible structure density, plus negative reward when the agent looks away from known geometry into emptier views.

---

Would you like me to map out the exact integration point for this reward tensor within your existing `navi` rollout collection loop, ensuring it scales correctly with your Mamba-2 Critic network's Advantage estimation?