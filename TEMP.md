# COGNITION_AND_TRAINING.md — Ghost-Matrix

## 1. Executive Summary

This document defines the advanced training dynamics, cognitive constraints, and reward psychology of the Ghost-Matrix architecture. Transitioning an autonomous agent from a simulated environment to physical hardware (Inference Parity) requires absolute mathematical discipline to prevent the agent from exploiting the simulator.

This specification codifies three major architectural pillars:

1. **Strict POMDP Compliance (Zero Data Leakage):** Guaranteeing the agent is mathematically blind to privileged simulation data.
2. **Information Foraging Reward Structure:** Sealing the "Void Exploit" by mathematically forcing the agent to seek and scan structural geometry.
3. **Asymmetric Sensor Allocation:** Maximizing functional visual resolution for doorway detection without violating the strict hardware VRAM constraints.

---

## 2. The POMDP Guarantee & Inference Parity

The most common failure mode in Reinforcement Learning is the **Partially Observable Markov Decision Process (POMDP) Violation**, where the agent accidentally learns to navigate using privileged simulator data (like GPS coordinates or a global map) that will not exist on a real drone.

Ghost-Matrix enforces strict data gatekeeping at the network boundary to guarantee 100% parity between training and real-world deployment.

### 2.1. The Observation Gatekeeper (`_obs_to_tensor`)

The simulation layer transmits the `DistanceMatrix` contract containing privileged data (`robot_pose`, `overhead` BGR minimap). **These fields are strictly quarantined.**

* Upon entering the `CognitiveMambaPolicy`, the `_obs_to_tensor` function extracts exactly three continuous arrays: `depth`, `semantic`, and `valid_mask`.
* The absolute $X, Y, Z$ coordinates and top-down maps are mathematically dropped. They never enter the PyTorch computational graph, ensuring the policy cannot overfit to the simulator's global coordinate system.

### 2.2. Internally Generated Auxiliary States

The Mamba-2 Temporal Core utilizes an `aux_tensor` to maintain temporal awareness. This tensor contains `[prev_reward, prev_loop_similarity, prev_intrinsic_reward]`.

* None of these values require external telemetry.
* **`loop_sim`** is generated entirely inside the brain via the `EpisodicMemory` comparing the current visual latent vector $z_t$ against historical visual latents.
* **`intrinsic_r`** is generated internally by the `RNDModule` measuring its own prediction error.
* **Deployment Parity:** When deployed on physical hardware, the neural network relies entirely on the onboard depth camera and its internal memory tensors, operating identically to the simulated training loop.

---

## 3. The "Information Foraging" Reward Architecture

Standard RL reward functions (e.g., rewarding forward displacement) fail in unbounded continuous space, creating the **Void Exploit**. If an agent receives a reward simply for moving, it will mathematically deduce that buildings are "high-risk penalty zones" and the open void is "infinite risk-free reward," causing it to fly away from the map.

The Ghost-Matrix reward pipeline transforms the agent into an **Information Forager**. It must "paint" physical structures with its sensors to survive.

### 3.1. The Void Existential Tax

An agent staring into empty space receives zero useful navigational data.

* **Mechanism:** If the sensor array returns absolute emptiness (`min_depth == 1.0`), a continuous negative existential tax (e.g., $-0.02$) is applied.
* **Behavioral Result:** The agent feels a mathematical "hunger" when lost in the void, forcing it to rotate and search for structures.

### 3.2. Structure Attraction (Depth Variance)

To guide the agent toward buildings, it is rewarded for the informational complexity of its view.

* **Mechanism:** The environment calculates the statistical variance of the foveated rays (`depth_variance`). An empty void has a variance of $0.0$. A complex building facade (walls, windows, corridors) yields high variance.
* **Behavioral Result:** The reward `+ (depth_variance * 0.1)` acts as a magnetic pull. The moment the agent's periphery clips a structure, the variance spikes, and the agent turns toward the building.

### 3.3. Proximity-Gated Exploration (The Paintbrush Rule)

Grid-based exploration rewards are easily hacked if the agent simply flies far above or far away from the map.

* **Mechanism:** The agent is only granted the `_EXPLORATION_REWARD` (+0.3) for entering a novel coordinate cell **IF** it is actively scanning a structure (`min_depth < 0.95`).
* **Behavioral Result:** The agent cannot farm rewards by flying in the void. It must closely trace the contours, walls, and interiors of the geometric mesh to earn exploration bonuses.

### 3.4. Wiggle-Proof Anti-Circling

Continuous action spaces naturally produce micro-oscillations ("wiggles"). Summing absolute frame-to-frame displacement artificially inflates the agent's traveled distance, allowing it to bypass hovering penalties.

* **Mechanism:** The penalty window strictly evaluates the **Net Displacement** between $T_0$ and $T_{current}$.
* **Behavioral Result:** If the agent rotates 360 degrees but its net spatial translation is $< 1.0$ meter, it receives a harsh `_CIRCLING_PENALTY`, forcing forward momentum.

---

## 4. Asymmetric Sensor Allocation & Resolution Optimization

Providing the agent with symmetric high-resolution vision (e.g., $128 \times 128$) triggers the **Curse of Dimensionality** and causes immediate VRAM exhaustion (CUDA Out Of Memory).

Because 2.5D flight environments consist of massive, flat structural boundaries at the top and bottom (ceilings and floors), symmetrical vertical rays are mathematically wasted compute. Ghost-Matrix utilizes **Asymmetric Sensor Allocation** to achieve extreme visual clarity for doorway navigation at zero extra performance cost.

### 4.1. The "Doorway Math" (Spatial Aliasing)

To successfully route through an indoor topology, the agent must be able to detect a standard $0.9m$ doorway from at least $5.0m$ away.

* At this distance, the doorway occupies a $\sim10^\circ$ field of view.
* If the horizontal ray spacing exceeds this angle, the rays straddle the gap, rendering the door mathematically invisible to the neural network until the agent crashes into the wall.

### 4.2. The Optimal Ghost-Matrix Configuration ($128 \times 24$)

To provide definitive horizontal clarity while staying strictly within the 2GB VRAM hardware ceiling, the canonical `matrix_shape` is set to `(128, 24)`.

* **Azimuth (Horizontal = 128):** Yields a ray spacing of **$2.8^\circ$**. This guarantees that a doorway 5 meters away will be struck by 3 to 4 independent rays, allowing the `RayViTEncoder` to distinctly identify the topological opening.
* **Elevation (Vertical = 24):** Reduces wasted compute on the sky/floor while maintaining enough vertical definition to detect overhangs or staircases.
* **ViT Patch Compatibility:** Both dimensions are perfectly divisible by the Vision Transformer's `patch_size=4`, eliminating the compute overhead of tensor zero-padding.

### 4.3. Low-VRAM Mitigation Strategy

Increasing the ray count to $3,072$ rays ($128 \times 24$) increases the memory footprint of the `TrajectoryBuffer` during the Backpropagation Through Time (BPTT) phase. If the Mamba-2 temporal core hits a VRAM ceiling during the `train_ppo_epoch` step, the resolution must not be lowered. Instead, the BPTT constraints are tightened:

1. **Reduce Minibatch:** Drop `minibatch_size` from $64$ to $32$.
2. **Truncate BPTT Window:** Drop `seq_len` from $32$ to $16$. The SSM will retain episode memory, but the gradient chain is safely truncated to prevent memory explosion.