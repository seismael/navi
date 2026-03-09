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
in `environment/backends/` that transform raw observations *to* the
engine's canonical `(B, 3, Az, El)` input format.

The active performance migration for Navi does **not** replace this brain.
Compiler/runtime upgrades are directed below the actor boundary:

- `projects/voxel-dag` compiles source meshes into `.gmdag` caches.
- `projects/torch-sdf` executes batched CUDA sphere tracing against those caches.
- `projects/environment` adapts the result back into canonical `DistanceMatrix`
  tensors that the Actor already consumes.

### 1.1. Canonical Launch Commands

Actor runtime and training entrypoints are standardized:

```bash
# Runtime service (step mode)
uv run navi-actor serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Shortcut command (equivalent to serve)
uv run brain

# Canonical training on the compiled-path runtime
uv run navi-actor train
```

Canonical repository-root wrappers mirror the same runtime:

```powershell
./scripts/train.ps1
./scripts/train-all-night.ps1
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

By default, canonical training discovers all available dataset scenes,
prepares the required compiled `.gmdag` corpus, and runs continuously until
the user explicitly requests a scene-specific or time-bounded override.

See [TRAINING.md](TRAINING.md) for the overnight training and dashboard attach workflow.

### 1.1.1. Canonical Training Surface

`train` is the single canonical actor training entrypoint. It means direct in-process stepping of
`SdfDagBackend` for the fastest deterministic rollout path. Alternate training
architectures are intentionally removed rather than preserved as equal modes.

The canonical training default is corpus-driven, not sample-driven: if the user
does not request a specific scene or subset, `train` must use the full
discovered dataset corpus.

The remaining performance work is now entirely inside this canonical path:
reduce the remaining CPU episodic-memory boundary and replace per-actor Python
transition assembly with indexed rollout storage in the canonical hot loop.

### 1.2. Pipeline Dataflow

```text
DistanceMatrix v2
  │
  ├── depth (B, 1, Az, El) ──┐
  ├── semantic (B, 1, Az, El)──┼── stack → (B, 3, Az, El)
  └── valid_mask (B, 1, Az, El)┘
          │
          ▼
  ┌──────────────────────────┐
  │   1. Ray-ViT Encoder     │  (B, 3, Az, El) → (B, 128) z_t
  │      Vision Transformer  │
  └──────────┬───────────────┘
             │ z_t
     ┌───────┼────────┐
     ▼       ▼        ▼
  ┌──────┐ ┌──────┐ ┌───────────────────┐
  │2. RND│ │3. Ep.│ │                   │
  │Module│ │Memory│ │4. Temporal Core     │
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
episodic memory operate externally on $z_t$ during training. The policy API is
intentionally insulated from simulator upgrades: diagnostic voxel/mesh/habitat
paths and the canonical SDF/DAG backend all converge on the same
`DistanceMatrix` contract.

---

## 2. Ray-ViT Encoder — Spatial Perception

**Module:** `perception.py`

Vision Transformer (ViT) that treats patches of the spherical grid as tokens 
with fixed sin/cos spherical positional encodings.

- **Input:** `(B, 3, Az, El)` float tensor. Channel 0 = depth, channel 1 = 
  semantic, channel 2 = valid_mask.
- **Output:** $z_t \in \mathbb{R}^{B \times D}$, where $D = 128$ (configurable 
  `embedding_dim`).
- **Mechanism:** Implements §8.4 of ARCHITECTURE.md. Uses a [CLS] token for 
  global spatial aggregation.

The encoder is resolution-agnostic and exploits the structured nature of the 
spherical input (center=front, edges=back).

---

## 3. RND Curiosity Module — Intrinsic Exploration

**Module:** `rnd.py`

Random Network Distillation (RND) forces the agent into mathematically unmapped
regions by rewarding high prediction errors in novel spatial embeddings.

**Architecture:**

- **Target network** (frozen, randomly initialized):
  `Linear(D, 128) → ReLU → Linear(128, 64)`
- **Predictor network** (trainable):
  `Linear(D, 128) → ReLU → Linear(128, 64)`

The predictor mirrors the target’s architecture so it cannot trivially
replicate the target from different random initialisation — the residual
prediction error serves as the novelty signal.

> **Note:** An earlier 3-layer predictor (`128→128→128→64`) was found to
> over-fit the target, collapsing the intrinsic reward signal. Reduced to
> 2 layers matching the target per the Feb 2026 training correctness pass
> (see TODO.md Issue 3).

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

**Data structure:** Canonical training uses a tensor-native cosine-similarity
ring buffer. Spatial embeddings remain on the policy device, the newest
`exclusion_window` entries are skipped, and the oldest entries are overwritten
in fixed-capacity FIFO order.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 128 | Dimensionality of stored embeddings |
| `capacity` | 100,000 | Maximum embeddings retained per episode |
| `exclusion_window` | 50 | Most recent entries skipped when querying (avoids trivial self-match) |
| `similarity_threshold` | 0.85 | Cosine similarity above which a loop is declared |

**API:**

- `reset()` — Clear all stored embeddings (called at episode start).
- `add(embedding)` — Store an L2-normalized embedding. Evicts oldest entry
  when capacity is reached (FIFO). Capacity enforcement must remain amortized;
  the canonical runtime may rebuild or compact lazily, but not on every insert
  after the buffer is full.
- `query(embedding) → (similarity, matched_embedding, is_loop)` — Search
  the first `N - exclusion_window` entries. Returns max cosine similarity,
  the closest historical embedding (or `None`), and a boolean loop flag.

### 4.1. Runtime Performance Rule

When the environment is accelerated by compiled SDF/DAG stepping, episodic
memory and rollout bookkeeping can become the dominant bottleneck. Canonical
actor training therefore treats episodic memory as a hot-path subsystem:

- capacity eviction must stay amortized,
- queries must operate on the active searchable window without per-step full
  structure rebuilds,
- trainer telemetry should separate memory, transport, and transition overhead
  so regressions can be attributed quickly.

**Loop penalty integration:** When `is_loop` is `True`, the `RewardShaper`
applies a penalty proportional to `max(0, similarity - threshold)`. This
instantly repels the agent from cyclical paths.

---

## 5. Temporal Core — Sequence Engine

**Module:** `mamba_core.py`

Current canonical runtime uses `mambapy` Mamba for temporal sequence modeling on
native Windows CUDA in this workstation profile.

**Current canonical runtime (Mar 2026):**

- `mambapy.mamba.Mamba` with parameters `d_model=128`, `d_state=64`, `d_conv=4`, `expand=2`.
- Sequence residual + normalization: `LayerNorm(core(z_seq) + z_seq)`.
- `forward_step` wraps single tokens as sequence length 1 for online inference.

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

**Migration note (active):**

- Canonical temporal backend is now selected by benchmark, not by optional runtime fallback.
- Candidate bake-off harnesses are allowed during migration, but production runtime remains single-path canonical.

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
  | Forward | 1.0 |
  | Vertical | 1.0 |
  | Lateral | 1.0 |
  | Yaw | 1.0 |

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
| Collision penalty | $r_{col}$ (applied on `done`) | 0.0 | Disabled by default (see TODO.md Issue 2) |
| Existential tax | $r_{tax}$ (per step) | -0.01 | Forces temporal efficiency |
| Velocity bonus | $w_v \cdot (\max(0, v_{fwd}) - 0.5 \cdot |v_{ang}|)$ | $w_v = 0.0$ | Disabled — speed is not a training signal |

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

### 8.1. PpoTrainer — Canonical PPO Runtime

**Module:** `training/ppo_trainer.py`

Single canonical PPO runtime with direct in-process `sdfdag` stepping and
BPTT-aware sequential minibatch sampling.

Current canonical runtime status:

- initial rollout seeding uses tensor-native `reset_tensor()` observations
- rollout stepping prefers tensor-native `batch_step_tensor_actions()`
- reward shaping, episodic memory, and rollout buffer appends stay batched and
  tensor-native on the canonical path
- CPU `DistanceMatrix` or action materialization remains only for low-volume
  dashboard publication and telemetry
- PPO updates run inline at rollout boundaries; the old async optimizer path
  is no longer part of the canonical runtime

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
4. Publish 13 actor telemetry metrics plus coarse canonical runtime perf via
  `environment.sdfdag.perf`.

**High-Fidelity Performance Metrics:**

| Metric | Description |
|--------|-------------|
| `sps` | Steps Per Second (throughput) |
| `fwd_ms` | Inference latency (Ray-ViT forward pass) |
| `env_ms` | Simulation latency (Batch raycasting) |
| `mem_ms` | Episodic memory query latency |
| `zw_ratio` | Coordination indicator; canonical inline PPO updates should keep this near `0%` |

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

### 8.2. TrajectoryBuffer

**Module:** `rollout_buffer.py`

Stores complete trajectory sequences for BPTT. Supports sequential minibatch
sampling that preserves temporal ordering (critical for Mamba2 hidden state
propagation). GAE(λ) advantage computation is performed in-buffer before
optimization begins.

---

## 9. Single Pipeline Invariant

The canonical trainer (`train`) uses one end-to-end pipeline:
**CLI-level `SdfDagBackend` wiring -> `PpoTrainer` -> `CognitiveMambaPolicy`**.

There are no alternative trainer modes or policy implementations. The sacred
cognitive pipeline is the only policy in the actor package:

- `act(obs, step_id, hidden)` — inference-mode action selection.
- `forward(obs_tensor, hidden)` — training forward pass (sample + value).
- `evaluate(obs_tensor, actions, hidden)` — PPO loss evaluation.
- `evaluate_sequence(obs_seq, actions_seq, hidden)` — BPTT sequence evaluation.
- `encode(obs_tensor)` — extract $z_t$ for RND / episodic memory.
- Checkpoint save/load via `state_dict`.

Feature extraction (`spherical_features.py`) provides 17 hand-crafted
spherical features used for reward shaping diagnostics but never for
action selection.

**Module:** `cognitive_policy.py`

---

## 10. Roadmap

### 10.1. Tensor-Native Episodic Memory

Canonical training now keeps episodic-memory query/add operations on tensors so
loop-detection embeddings stay on the same device as the policy rollout.
Remaining memory work is limited to profiling and scaling, not FAISS migration.

### 10.0. Tensor-Native Canonical Rollout (Completed)

The canonical trainer now keeps the rollout path tensor-native in the intended
order:

1. tensor-native environment runtime seam
2. tensor-native trainer observation path
3. tensor-native action stepping
4. batched reward shaping and rollout storage
5. tensor-native episodic memory

Low-volume `DistanceMatrix` publication remains available for dashboard and
telemetry observability, but it is no longer part of the canonical rollout hot
path.

### 10.2. Ray-ViT Encoder

Uses `RayViTEncoder` (Vision Transformer) that natively ingests
foveated (variable-density) ray sequences. Each ray or angular cluster becomes
a sequence token with absolute $(\theta, \phi)$ positional encoding. Multi-head
attention resolves spatial topology without the rigid 2D grid assumption of CNNs.

### 10.3. Canonical Inline PPO Updates

Canonical training now performs PPO updates inline at rollout boundaries. This
keeps one measured runtime only, preserves the tensor-native rollout path, and
avoids the coordination regressions that appeared once the background optimizer
thread stopped being net beneficial.

### 10.4. FlashAttention IO Fusion for Mamba2

Hardware-aware SRAM IO fusion for the Mamba2 core: load hidden state $h_t$ from
GPU SRAM, compute the selective state transition, and write back to SRAM without
touching global GPU memory. This eliminates memory bandwidth bottlenecks in the
temporal pipeline.

---

## 11. Operational Validation Baseline (Mar 2026)

This section records durable runtime guarantees for the current canonical actor
path on native Windows CUDA.

- **Canonical temporal runtime:** `Mamba2TemporalCore` with `mambapy` is the active production path.
- **Canonical startup evidence:** actor logs emit `Mamba2TemporalCore: canonical mambapy runtime active`.
- **CUDA-only training policy:** PPO trainer fails fast if CUDA is unavailable or CUDA kernel preflight fails.
- **CUDA startup evidence:** trainer logs explicit preflight details (`device`, `capability`, CUDA version).
- **Launcher lifecycle policy:** `run-ghost-stack -Train -NoDashboard` waits for natural train completion and no longer exits after startup-only logs.
- **Training readiness policy:** training socket readiness wait uses a 60-second timeout window in stack launcher checks.
- **Checkpoint schedule robustness:** checkpoint saves are interval-crossing based (not modulo-only), avoiding missed saves when step increments skip exact boundaries.
