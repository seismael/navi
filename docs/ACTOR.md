# ACTOR.md — Cognitive Actor Architecture

**Subsystem:** Brain Layer — Sacred Cognitive Engine  
**Package:** `navi-actor`  
**Status:** Active canonical specification  
**Policy:** See [AGENTS.md](../AGENTS.md) for implementation rules and non-negotiables

---

## 1. Cognitive Pipeline Overview

The Actor implements a 5-stage neural pipeline that transforms raw spherical
depth observations into continuous 4-DOF velocity commands. This pipeline is
**sacred and immutable** — it is never modified to accommodate new data sources
or sensor types. External data connects only through `DatasetAdapter` instances
in `section-manager/backends/` that transform raw observations *to* the
engine's canonical `(B, 2, Az, El)` input format.

```text
DistanceMatrix v2
  │
  ├── depth (B, 1, Az, El) ──┐
  │                           ├── stack → (B, 2, Az, El)
  └── semantic (B, 1, Az, El)─┘
          │
          ▼
  ┌──────────────────────────┐
  │   1. FoveatedEncoder     │  (B, 2, Az, El) → (B, 128) z_t
  │      4-layer Conv2d CNN  │
  └──────────┬───────────────┘
             │ z_t
     ┌───────┼────────┐
     ▼       ▼        ▼
  ┌──────┐ ┌──────┐ ┌───────────────────┐
  │2. RND│ │3. Ep.│ │                   │
  │Module│ │Memory│ │4. Mamba2 Temporal  │
  └──┬───┘ └──┬───┘ │   Core            │
     │        │     │   z_t → f_t       │
     │        │     └────────┬──────────┘
     │        │              │ f_t
     │        │              ▼
     │        │     ┌──────────────────┐
     │        │     │5. ActorCritic    │
     │        │     │   Heads          │
     │        │     │   f_t → (μ, V)   │
     │        │     └────────┬─────────┘
     │        │              │
     ▼        ▼              ▼
  intrinsic  loop       Action v2
  reward     penalty    (4-DOF velocity)
```

**Module:** `cognitive_policy.py` — `CognitiveMambaPolicy` composes the
encoder, temporal core, and actor-critic heads as a single `nn.Module`. RND and
episodic memory operate externally on $z_t$ during training.

---

## 2. FoveatedEncoder — Spatial Perception

**Module:** `perception.py`

A 4-layer convolutional neural network that compresses the full spherical
observation into a dense spatial embedding.

**Architecture:**

| Layer | Type | Channels | Kernel | Stride | Activation |
|-------|------|----------|--------|--------|------------|
| 1 | `Conv2d` | 2 → 32 | 3×3 | 2 | ReLU |
| 2 | `Conv2d` | 32 → 64 | 3×3 | 2 | ReLU |
| 3 | `Conv2d` | 64 → 128 | 3×3 | 2 | ReLU |
| 4 | `Conv2d` | 128 → 128 | 3×3 | 2 | ReLU |
| Pool | `AdaptiveAvgPool2d(1)` | 128 → 128 | — | — | — |
| FC | `Linear` | 128 → D | — | — | — |

- **Input:** `(B, 2, Az, El)` float tensor. Channel 0 = normalized depth
  $\in [0, 1]$, channel 1 = semantic class ID (raw float cast).
- **Output:** $z_t \in \mathbb{R}^{B \times D}$, where $D = 128$ (configurable
  `embedding_dim`).
- **Padding:** All convolutions use `padding=1` to preserve spatial information
  through stride-2 downsampling.

The encoder is deliberately resolution-agnostic — it works with any
`(Az, El)` grid size thanks to the adaptive average pooling layer.

---

## 3. RND Curiosity Module — Intrinsic Exploration

**Module:** `rnd.py`

Random Network Distillation (RND) forces the agent into mathematically unmapped
regions by rewarding high prediction errors in novel spatial embeddings.

**Architecture:**

- **Target network** (frozen, randomly initialized):
  `Linear(D, 128) → ReLU → Linear(128, 64)`
- **Predictor network** (trainable):
  `Linear(D, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 64)`

The predictor has an extra hidden layer to give it enough capacity to learn
the target's mapping without trivially memorizing it.

**Intrinsic reward computation:**

$$r_{intrinsic} = \text{clamp}\left(\frac{\|f_{target}(z_t) - f_{predictor}(z_t)\|^2 - \mu}{\sigma}, -5, 5\right)$$

Where $\mu$ and $\sigma$ are running mean/variance statistics updated via
Welford's online algorithm. The clamping prevents extreme outliers from
destabilizing training.

**Distillation loss:** MSE between target and predictor outputs, optimized
alongside PPO gradients:

$$\mathcal{L}_{RND} = \mathbb{E}\left[\|f_{target}(z_t) - f_{predictor}(z_t)\|^2\right]$$

---

## 4. Episodic Memory — Loop Detection

**Module:** `memory/episodic.py`

A non-parametric KNN memory buffer that detects revisited spatial states,
solving the infinite-looping failure mode inherent to standard Markovian RL.

**Data structure:** When FAISS is available, uses `faiss.IndexFlatIP` (CPU
inner-product index) over L2-normalized embeddings — equivalent to cosine
similarity. Falls back to brute-force numpy dot products when FAISS is not
installed.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 128 | Dimensionality of stored embeddings |
| `capacity` | 10,000 | Maximum embeddings retained per episode |
| `exclusion_window` | 50 | Most recent entries skipped when querying (avoids trivial self-match) |
| `similarity_threshold` | 0.85 | Cosine similarity above which a loop is declared |

**API:**

- `reset()` — Clear all stored embeddings (called at episode start).
- `add(embedding)` — Store an L2-normalized embedding. Evicts oldest entry
  when capacity is reached (FIFO). FAISS index is rebuilt on eviction.
- `query(embedding) → (similarity, matched_embedding, is_loop)` — Search
  the first `N - exclusion_window` entries. Returns max cosine similarity,
  the closest historical embedding (or `None`), and a boolean loop flag.

**Loop penalty integration:** When `is_loop` is `True`, the `RewardShaper`
applies a penalty proportional to `max(0, similarity - threshold)`. This
instantly repels the agent from cyclical paths.

---

## 5. Mamba2 Temporal Core — Sequence Engine

**Module:** `mamba_core.py`

Replaces standard RNNs (GRUs) and Transformer attention with a Selective State
Space Model (SSM), providing infinite-horizon temporal memory with $O(n)$
linear-time complexity.

**Dual-mode implementation:**

- **Primary:** `mamba_ssm.Mamba2` when the `mamba-ssm` package is installed.
  Parameters: `d_model=128`, `d_state=64`, `d_conv=4`, `expand=2`.
- **Fallback:** Single-layer `nn.GRU` with `batch_first=True` when `mamba-ssm`
  is unavailable.

**Selective filtering mechanism:** The SSM's learned matrices $A$ and $B$
mathematically learn to filter redundant spatial frames (e.g., staring at a
uniform wall while moving forward) and selectively update the hidden state $h_t$
only at critical geometric junctions — doorways, corners, open-space transitions.
This is achieved through the input-dependent selection mechanism that gates how
much of each input token influences the state transition.

**Architecture details:**

- **Input:** `(B, T, D)` sequence of encoder outputs.
- **Residual connection:** Output = `LayerNorm(core(z_seq) + z_seq)`.
- **Output:** $f_t \in \mathbb{R}^{B \times D}$ temporal features.

**Online inference:** `forward_step(z_t, hidden)` wraps a single embedding as
a length-1 sequence for step-by-step inference in the server loop. Hidden state
management:

- Mamba2: stateless in training (full sequence at once); uses internal cache for
  single-step inference.
- GRU fallback: explicit `(1, B, D)` hidden state tensor, passed between steps.

---

## 6. Actor-Critic Heads — Decision Layer

**Module:** `actor_critic.py`

Parallel MLPs project the temporal features $f_t$ into continuous velocity
distributions and a state-value estimate.

### Actor Head (Gaussian Policy)

```text
Linear(D, 64) → ReLU → Linear(64, 4) → Tanh → scale by action_scales
```

- **Output:** $\mu \in \mathbb{R}^{B \times 4}$ — mean of a diagonal Gaussian.
- **Action dimensions:** `[forward, vertical, lateral, yaw]`.
- **Action scaling:** Each dimension is multiplied by a configurable maximum:

  | Dimension | Default Max |
  |-----------|-------------|
  | Forward | 1.2 |
  | Vertical | 0.8 |
  | Lateral | 0.8 |
  | Yaw | 1.2 |

- **Learnable `log_std`:** Per-dimension `nn.Parameter` initialized to zero
  ($\sigma_0 = 1.0$). The policy learns to reduce variance as training
  progresses.

### Critic Head (Value Function)

```text
Linear(D, 64) → ReLU → Linear(64, 1) → squeeze
```

- **Output:** $V(s) \in \mathbb{R}^B$ — scalar state-value estimate.

### Sampling and Evaluation

- `sample(features)` → `(actions, log_probs, values)`: draws from
  $\mathcal{N}(\mu, \sigma^2)$, returns sampled actions + log probabilities.
- `log_prob(features, actions)` → `(B,)`: evaluates given actions under the
  current policy (used for PPO importance ratio).
- `entropy()` → scalar: differential entropy of the diagonal Gaussian
  ($\sum_i 0.5 + 0.5\ln(2\pi) + \ln\sigma_i$).

---

## 7. Reward Shaping — Tripartite Reward Formulation

**Module:** `reward_shaping.py`

The total reward combines three distinct signal sources to drive efficient,
exploration-oriented navigation:

$$r_{total} = r_{extrinsic} + r_{collision} + r_{tax} + r_{velocity} + \beta(t) \cdot r_{intrinsic} - \lambda_{loop} \cdot \max(0, s_{loop} - \tau)$$

### 7.1. Extrinsic Physics

| Component | Formula | Default | Purpose |
|-----------|---------|---------|---------|
| Collision penalty | $r_{col}$ (applied on `done`) | -10.0 | Instant penalty for wall contact |
| Existential tax | $r_{tax}$ (per step) | -0.01 | Forces temporal efficiency |
| Velocity bonus | $w_v \cdot (\max(0, v_{fwd}) - 0.5 \cdot |v_{ang}|)$ | $w_v = 0.1$ | Encourages forward momentum, penalizes spinning |

### 7.2. Intrinsic Curiosity (RND)

$$r_{intrinsic} = \beta(t) \cdot \hat{r}_{RND}$$

Where $\beta(t)$ anneals linearly from `intrinsic_coeff_init` (default 1.0) to
`intrinsic_coeff_final` (default 0.01) over `intrinsic_anneal_steps` (default
500,000) steps. This ensures strong early exploration that fades as the policy
converges.

### 7.3. Explicit Loop Penalty

$$r_{loop} = -\lambda_{loop} \cdot \max(0, s_{loop} - \tau)$$

Where $s_{loop}$ is the episodic memory cosine similarity and $\tau = 0.85$ is
the loop detection threshold. Default $\lambda_{loop} = 0.5$.

### 7.4. Configurable Parameters

All reward constants are constructor arguments to `RewardShaper`, enabling
hyperparameter sweeps without code changes.

---

## 8. Training Loop

### 8.1. PpoTrainer — Proximal Policy Optimization

**Module:** `training/ppo_trainer.py`

Synchronous step-mode PPO with BPTT-aware sequential minibatch sampling.

**Training cycle:**

1. Collect $T$ steps of experience in a `TrajectoryBuffer`, storing
   `(obs, action, log_prob, value, reward, done)` tuples per step.
2. Compute advantages via Generalized Advantage Estimation: $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$
   where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
3. For $K$ optimization epochs over $M$ sequential minibatches:
   - Evaluate actions under current policy: `(new_log_prob, new_value, entropy)`.
   - PPO clipped surrogate loss:
     $\mathcal{L}_{policy} = -\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)$
     where $r_t = \exp(\log\pi_{new} - \log\pi_{old})$.
   - Value loss with clipping.
   - Entropy bonus: $\mathcal{L}_{entropy} = -c_{ent} \cdot H[\pi]$.
   - RND distillation loss: $\mathcal{L}_{RND}$.
4. Publish 13 telemetry metrics via `telemetry_event_v2`.

**Telemetry metrics published per update:**

| Metric | Description |
|--------|-------------|
| `policy_loss` | Clipped surrogate loss |
| `value_loss` | Clipped value function loss |
| `entropy` | Policy entropy |
| `rnd_loss` | RND distillation loss |
| `mean_reward` | Mean episode reward in buffer |
| `mean_intrinsic` | Mean RND intrinsic reward |
| `mean_loop_sim` | Mean episodic memory similarity |
| `clip_fraction` | Fraction of clipped ratios |
| `approx_kl` | Approximate KL divergence |
| `explained_var` | Value function explained variance |
| `mean_advantage` | Mean GAE advantage |
| `grad_norm` | Global gradient norm |
| `beta` | Current RND annealing coefficient |

### 8.2. OnlineSphericalTrainer — Lightweight REINFORCE

**Module:** `training/online.py`

A simpler REINFORCE-style policy gradient trainer for `LearnedSphericalPolicy`.
Uses the 17-dimensional hand-crafted spherical feature vector (see §9.2) and
trains linear weight vectors via policy gradient — no neural network overhead.

### 8.3. TrajectoryBuffer

**Module:** `rollout_buffer.py`

Stores complete trajectory sequences for BPTT. Supports sequential minibatch
sampling that preserves temporal ordering (critical for Mamba2 hidden state
propagation). GAE(λ) advantage computation is performed in-buffer before
optimization begins.

---

## 9. Alternative Policies

### 9.1. ShallowPolicy — Rule-Based Reactive Control

**Module:** `policy.py`

A hand-coded depth-reactive policy that requires no training:

- Splits the depth panorama into left / right azimuth sectors.
- Computes mean depth per sector (valid bins only).
- Steers toward the sector with greater clearance:
  `yaw = clip((right_mean - left_mean) * gain, -1.5, 1.5)`.
- Reduces forward speed when mean occupancy is low (close to walls).
- Default forward velocity: 1.2, gain: 1.8.

Useful for smoke testing, dashboard verification, and establishing baseline
navigation behaviour before training.

### 9.2. LearnedSphericalPolicy — Feature-Based Linear Policy

**Module:** `policy.py`, `spherical_features.py`

A lightweight deterministic policy driven by 17 hand-crafted spherical features
extracted from the full 360° depth matrix:

| Features 1–5 | Directional clearance |
|--------------|----------------------|
| Front min depth | Minimum depth in the forward azimuth band |
| Front mean depth | Mean depth in the forward band |
| Rear min depth | Minimum depth in the backward band |
| Left mean depth | Mean depth in the left hemisphere |
| Right mean depth | Mean depth in the right hemisphere |

| Features 6–13 | 8-sector azimuth means |
|---------------|----------------------|
| Sector $k$ mean | Mean of `per_az_min` for sector $k \in [0, 7]$ |

| Features 14–17 | Elevation-aware 3D control |
|----------------|--------------------------|
| Floor proximity | Min depth in lower elevation bins |
| Ceiling proximity | Min depth in upper elevation bins |
| Vertical clearance ratio | `ceil_min / floor_min` (clamped) |
| Near-object fraction | Fraction of valid bins with depth < 0.15 |

**Action computation:** Four independent linear heads with sigmoid (forward) or
tanh (yaw, vertical, lateral) activation, scaled by configurable maximums.
Checkpoint format: compressed NPZ with weight vectors + bias + scale parameters.

### 9.3. CognitiveMambaPolicy — Full Neural Pipeline

**Module:** `cognitive_policy.py`

The sacred end-to-end neural policy described in §1–§6. Used for PPO training
with the full Mamba2 temporal pipeline. Supports:

- `act(obs, step_id, hidden)` — inference-mode action selection.
- `forward(obs_tensor, hidden)` — training forward pass (sample + value).
- `evaluate(obs_tensor, actions, hidden)` — PPO loss evaluation.
- `evaluate_sequence(obs_seq, actions_seq, hidden)` — BPTT sequence evaluation.
- `encode(obs_tensor)` — extract $z_t$ for RND / episodic memory.
- Checkpoint save/load via `state_dict`.

---

## 10. Roadmap

### 10.1. GPU FAISS Episodic Memory

Migration from CPU `faiss.IndexFlatIP` to `faiss.StandardGpuResources` for
VRAM-locked KNN lookups. Eliminates PCIe bus latency during rollout, achieving
the "no stall" guarantee for memory queries.

### 10.2. Ray-ViT Encoder

Replace `FoveatedEncoder` (CNN) with a Vision Transformer that natively ingests
foveated (variable-density) ray sequences. Each ray or angular cluster becomes
a sequence token with absolute $(\theta, \phi)$ positional encoding. Multi-head
attention resolves spatial topology without the rigid 2D grid assumption of CNNs.

### 10.3. Async Double-Buffered Training

Dual-`TrajectoryBuffer` implementation with thread bifurcation — optimization
thread runs BPTT while the simulation thread continues rollout on a secondary
buffer with zero wait.

### 10.4. FlashAttention IO Fusion for Mamba2

Hardware-aware SRAM IO fusion for the Mamba2 core: load hidden state $h_t$ from
GPU SRAM, compute the selective state transition, and write back to SRAM without
touching global GPU memory. This eliminates memory bandwidth bottlenecks in the
temporal pipeline.
