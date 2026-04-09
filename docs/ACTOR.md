- keep PPO update attribution on the canonical learner path with explicit sub-stage means for minibatch fetch, minibatch prep, policy evaluation, backward, gradient clip, optimizer step, RND step, and progress-callback overhead
- when diagnostic CUDA event profiling is enabled, canonical PPO learner stage timings must synchronize around eval, backward, clip, optimizer, and RND so device time is attributed truthfully rather than inferred from host dispatch latency
# ACTOR.md - Canonical Cognitive Actor Architecture

**Subsystem:** Brain Layer - Sacred Cognitive Engine
**Package:** `navi-actor`
**Status:** Active canonical specification
**Policy:** See `AGENTS.md` for implementation rules and non-negotiables

## 1. Executive Summary

The actor is the sacred cognitive core of Navi. It transforms spherical
observations into continuous 4-DOF motion commands while maintaining temporal
context and exploration pressure.

The core design rule is unchanged:

- environment and compiler work may improve the runtime beneath the actor
- but the actor-side cognitive pipeline itself remains fixed

The active production architecture therefore improves the system below the actor
boundary while preserving the actor's observation and action semantics.

## 2. Canonical Responsibilities

The actor domain owns:

- the Ray-ViT perception stack
- temporal sequence modeling
- actor-critic action and value heads
- intrinsic curiosity via RND
- episodic memory and loop avoidance
- rollout buffer management
- PPO update logic
- coarse actor telemetry publication

The actor domain does not own:

- corpus compilation
- raw dataset normalization
- world stepping implementation
- dashboard rendering

## 3. Canonical Launch Surfaces

```bash
# Runtime service
uv run navi-actor serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Shortcut
uv run brain

# Canonical training
uv run navi-actor train
```

Repository wrappers mirror the same surfaces:

```powershell
./scripts/train.ps1
./scripts/train-all-night.ps1
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

By default, canonical training:

- discovers the full available corpus
- prepares or reuses compiled `.gmdag` assets
- uses the canonical `256x48` observation contract
- runs continuously until explicitly stopped or bounded by user override

## 4. Canonical Training Surface

`train` is the single canonical actor training entrypoint.
It means:

- direct in-process stepping of `SdfDagBackend`
- batched observation and action flow
- tensor-native preference when runtime seams are available
- preallocated batched rollout storage on the active device when the canonical trainer has a fixed rollout horizon
- masked batched hidden-state, auxiliary-state, and episode bookkeeping instead of per-actor Python state in the rollout hot path
- one production rollout/update loop

Alternate training architectures are intentionally not preserved as equal modes.

## 5. Input And Output Contracts

### 5.1 Public Observation Boundary

The actor consumes `DistanceMatrix` as the public observation contract.
That contract preserves:

- spherical depth
- semantic ids
- validity mask
- pose metadata for diagnostics and replay

### 5.2 Internal Observation Tensor

The canonical training path stacks public observation semantics into a tensor of
shape `(B, 3, Az, El)`:

- channel `0`: normalized depth
- channel `1`: semantic ids
- channel `2`: valid mask

This is the actor's true runtime input for the hot path.

Canonical default resolution for production training, benchmarks, wrappers, and
diagnostic entrypoints is `Az=256`, `El=48`. Lower-resolution defaults are not
part of the active production surface.

### 5.3 Action Boundary

The actor internally reasons in 4-DOF form:

- forward
- vertical
- lateral
- yaw

The public `Action` model remains stable for service mode and diagnostics.

## 6. Cognitive Pipeline

```text
DistanceMatrix semantics
  -> stacked observation tensor
  -> Ray-ViT encoder
  -> latent embedding z_t
  -> RND and episodic-memory side channels
  -> temporal core
  -> actor-critic heads
  -> action sample and value estimate
```

This pipeline is sacred and immutable at the architectural level.
Compiler/runtime improvements must preserve it rather than pressure it into new
sensor-specific branches.

## 7. Perception: Ray-ViT Encoder

**Module:** `perception.py`

The Ray-ViT encoder treats spherical observations as structured patches with
fixed positional meaning.

Current properties:

- input shape `(B, 3, Az, El)`
- output latent embedding `z_t` with configurable embedding dimension
- strided `Conv2d` patch projection with `patch_size=8`
- `nn.TransformerEncoder` over patch tokens plus one `[CLS]` token
- fixed spherical positional structure rather than ad hoc flattened features

Architectural consequence:

- runtime upgrades must preserve spherical observation semantics
- imported ideas such as foveation remain optional future work and do not alter
  the current production contract

### 7.1 Resolution Scaling Boundary

The current encoder is not resolution-linear.

Patch-token count grows as:

`(Az / 8) * (El / 8)`

and the encoder then applies full self-attention over those tokens. That means
encoder cost grows much faster than raw ray count once the profile is widened.

Concrete token counts on the active benchmark profiles are:

- `256x48` -> `192` patch tokens
- `384x72` -> `432` patch tokens
- `512x96` -> `768` patch tokens
- `768x144` -> `1728` patch tokens

That is why the environment runtime and the actor do not scale the same way.
The environment mostly tracks ray-count growth; the actor additionally pays the
transformer token-growth penalty during PPO evaluation.

### 7.2 Current Measured Consequence

The March 2026 resolution sweep established the following on the active
4-actor canonical trainer surface:

- `256x48`: about `49.6 SPS`, `ppo_update_ms` about `1019.68`
- `384x72`: about `49.34 SPS`, `ppo_update_ms` about `1249.87`
- `512x96`: about `43.96 SPS`, `ppo_update_ms` about `17731.88`
- `768x144`: trainer OOM on the active MX150 surface inside transformer
  self-attention during PPO update

Those results come from the canonical bounded trainer artifacts under
`artifacts/benchmarks/resolution-compare/` and they matter architecturally:

- the environment runtime is no longer the only performance story
- the actor encoder is already a first-class scaling limit
- future temporal-core promotion alone will not make the full trainer scale
  linearly with observation resolution

## 8. Curiosity: RND Module

**Module:** `rnd.py`

RND provides intrinsic novelty pressure by comparing a frozen target projection
against a trainable predictor.

Current role in the production system:

- encourage exploration in novel latent regions
- remain separate from the public wire model
- contribute to the total reward formulation without widening service contracts

The predictor architecture was already simplified in earlier repo work to avoid
collapsing the intrinsic signal.

## 9. Loop Avoidance: Episodic Memory

**Module:** `memory/episodic.py`

The current canonical system uses a tensor-native cosine-similarity memory
buffer with fixed-capacity overwrite behavior.

Architecturally important rules:

- embeddings stay on the active policy device
- exclusion windows avoid trivial self-matches
- capacity enforcement must stay amortized
- full rebuilds on every post-capacity insert are not acceptable on the
  canonical training path
- when the same rollout embedding batch is both queried and inserted in one
  trainer tick, normalization must be computed once and reused across both
  memory operations
- when sparse step telemetry is enabled, action and done mirrors should be
  batch-copied once for the selected telemetry tick rather than extracted via
  per-actor device synchronization

This subsystem is now part of the performance conversation, not just the
behavior conversation.

## 9.1 Rollout Storage Direction

The current production direction for trainer-side throughput is:

- keep rollout observations, actions, values, rewards, dones, and auxiliary tensors in reusable batched device storage
- batch hidden-state carry and reset with done masks
- reserve host extraction for sparse telemetry, checkpoints, and passive observation publication only
- when completed episodes must be published, actor id, return, and length should be batch-copied once for the done set rather than mirrored through fragmented per-field host transfers
- keep aggregate reward accounting on-device across rollout ticks and materialize it on the host only when producing final training metrics
- when PPO update summaries are needed, learner loss metrics should be materialized through one packed epoch-end host copy rather than repeated scalar extraction calls
- keep PPO update attribution on the canonical learner path with explicit sub-stage means for minibatch prep, policy evaluation, backward, gradient clip, optimizer step, and RND step
- when `seq_len > 0`, canonical PPO minibatches must already carry sequence-native observation, action, and auxiliary views rather than asking the learner to rebuild them from flat tensors
- canonical rollout buffers must normalize advantages once per finalized rollout and reuse that cached tensor across PPO epochs rather than re-normalizing during each sampling pass or leaving the batched path raw
- canonical PPO learner minibatch prep must not rebuild or copy hidden-state batches on the update path while the production trainer keeps one hidden-free sequence seam across supported temporal backends
- canonical trainer rollout code should keep hidden-state storage out of the hot path unless a benchmark-proven production path requires it for correctness
- batched rollout buffers and minibatches should avoid hidden-state slabs on the canonical path while still preserving the recurrent API seam for supported alternate backends and future promotions
- when `MultiTrajectoryBuffer` is running in its canonical batched mode (`capacity` set), it must not also allocate per-actor fallback `TrajectoryBuffer` wrappers; the production rollout store is the batched slab itself
- completed-episode host extraction must happen only for actors whose sparse episode telemetry will actually be published; canonical reset bookkeeping for done actors stays on-device
- initial tensor resets and live runtime steps must materialize `DistanceMatrix` observations only for actors whose dashboard observation stream will actually publish; when observation streaming is disabled, canonical training must request no published observations at all

This preserves the sacred actor topology while removing avoidable Python
orchestration above the already-batched CUDA environment step.

## 10. Temporal Core

**Module:** `cognitive_policy.py`

Canonical runtime now exposes one controlled selector on the same sacred actor topology: `mamba2` is the default backend, with `gru` and `mambapy` available as supported comparison backends.

Current architectural status:

- one canonical actor topology and one canonical trainer surface in production
- all production training and inference surfaces share the same temporal-core selector contract
- `mamba2` is the default backend, proven by 25K-step training comparison to deliver significantly better learning quality
- `gru` and `mambapy` are supported alternate backends for controlled comparisons on that same surface
- fused Mamba-2 (`mamba-ssm`) remains a future hardware-fused upgrade target, not the current production dependency

### 10.1 Why Mamba2 SSD Is Canonical

The temporal core sits directly on the actor's hottest remaining wall-clock
path: `CognitiveMambaPolicy.evaluate_sequence()` during PPO BPTT updates.

A controlled 25K-step head-to-head training comparison on the active MX150
machine established the decision:

| Metric | GRU (cuDNN) | Mamba2 (Pure-PyTorch SSD) |
|---|---|---|
| Final reward_ema | -1.48 | **-0.88** |
| Rollout SPS | ~100 | ~72 |
| Forward pass | 6–8 ms | 13–18 ms |
| Wall-clock (25K steps) | 7m 28s | 9m 37s |

Mamba2 SSD's selective state-space mechanism (8,192-dim effective state vs
GRU's 128-dim hidden) provides meaningfully better long-range situational
awareness across partial observability, which shows up as more stable and
higher-quality late-game reward.

The throughput gap is modest (1.29x wall-clock) because PPO optimizer cost
dominates total training time regardless of temporal core choice.

The pure-PyTorch SSD implementation requires no Triton, no causal-conv1d
wheels, and no custom C++ extensions — it uses only standard PyTorch ops
(einsum, cumsum, exp, tril) and works on any CUDA-capable PyTorch install.

### 10.2 Canonical Enforcement

The selector contract is now the architecture policy, not ad hoc tuning.

- canonical actor environments must expose one temporal-core selector across config, CLI, and wrappers
- `mamba2` is the default backend and must continue to work with no extra extension build requirement
- `gru` and `mambapy` are supported only through that same selector on the same canonical trainer and serve surfaces
- fused `mamba-ssm` remains a migration target and may be benchmarked, but it is not part of the current supported production selector set

### 10.3 How To Switch Back Later

If a future temporal core candidate proves better learning quality or throughput
in a controlled head-to-head comparison, the upgrade checklist is:

- run a bounded training comparison (25K+ steps) on the canonical trainer surface
- compare final reward_ema, throughput SPS, and wall-clock time
- verify the candidate works on the active training machine with no extra build requirements
- update config defaults, wrappers, tests, and docs together in one pass
- update AGENTS.md to codify the new canonical runtime

The temporal core remains critical because it gives the actor memory across
partial observability without reopening transformer-scale quadratic costs.

## 11. Decision Layer: Actor-Critic Heads

**Module:** `actor_critic.py`

The actor-critic heads project temporal features into:

- Gaussian action distribution parameters
- scalar value estimates

Architectural consequence:

- action semantics remain stable even if the environment stepping seam becomes
  more tensor-native underneath
- PPO remains able to evaluate log-probabilities and clipped importance ratios
  over the same action meaning

## 12. Reward Formulation In Actor Context

The total training signal combines:

- environment-side reward and penalties
- intrinsic novelty pressure from RND
- loop-avoidance penalties from episodic memory

The environment and actor each own different parts of the shaping problem.
That split is intentional:

- environment owns geometry-derived shaping directly tied to observations
- actor owns latent novelty and memory-derived shaping

### 12.1 Environment-Side Components (9 terms)

The environment reward engine computes nine tensor components per step:

1. **Exploration reward** — decays with spatial visit count; includes heading
   novelty and frontier adjacency bonuses.
2. **Progress reward** — proportional to displacement, discounted by proximity
   ratio so approaching walls yields diminishing credit.
3. **Clearance-delta reward** — positive when the actor increases free space
   while within the obstacle clearance window (`3.0 m`).
4. **Starvation penalty** — fires when horizon-saturated rays dominate the
   spherical observation.
5. **Proximity penalty** — scales with the fraction of near-field valid hits
   below the proximity threshold (`2.0 m`).
6. **Structure-band reward** — rewards stable mid-range geometry visibility.
7. **Forward-structure reward** — rewards informative geometry in the forward
   sector.
8. **Inspection reward** — rewards orientation changes that gain structure
   information.
9. **Collision penalty** — velocity-scaled: `penalty * (1 + speed_norm)` so
   fast crashes are punished more severely than gentle grazing.

All terms are batched CUDA tensors derived from the existing spherical
observation. They do not require a second sensing pipeline.

Exploration rewards are additionally clearance-gated: rewards are multiplied by
`clamp(current_clearance, 0, 1)` so exploring into tight spaces yields
diminishing exploration credit.

### 12.2 Actor-Side Components (5 terms)

The actor's `RewardShaper` adds:

1. **Existential tax** — constant per-step cost (`-0.02`).
2. **Velocity reward** — disabled by default (`weight=0.0`) to remove forward
   approach bias near obstacles.
3. **Intrinsic novelty** — RND prediction-error signal, annealed over training.
4. **Loop penalty** — cosine-similarity detection against episodic memory.
5. **Loop temporal decay** — exponential half-life weighting on loop detection.

### 12.3 Action Space

The actor reasons in 4-DOF continuous form:

- `forward` — forward/backward translation
- `vertical` — up/down translation
- `lateral` — left/right strafe
- `yaw` — horizontal rotation

This covers all six movement directions (forward, backward, up, down, left,
right) plus heading control through yaw. Pitch and roll are intentionally
excluded to keep the action space simple and the observation contract stable.

## 13. Canonical PPO Runtime

**Module:** `training/ppo_trainer.py`

The current production trainer is a direct in-process PPO runtime with a strong
preference for runtime tensor seams.

Current production traits:

- tensor-native reset seeding when available
- tensor-native batched stepping when available
- selective publication only for actors that need passive observation
- inline PPO updates at rollout boundaries
- coarse dashboard heartbeat publication during optimizer windows
- configurable passive dashboard observation cadence with a canonical default of
  about `10 Hz` for the selected actor

The previous async optimizer-worker design is no longer the production runtime.

## 14. Rollout Buffer Architecture

**Module:** `rollout_buffer.py`

The rollout buffer stores trajectory information for PPO and sequential
minibatching.

Current production rules:

- keep actor trajectories logically separate for PPO and BPTT correctness
- cache stacked rollout tensors once per PPO update when practical
- keep minibatch shuffle indices on the same device as sampled tensors
- avoid rebuilding Python transition objects unnecessarily when batched tensor
  fields already exist

## 15. Actor-Side Performance Reality

The current actor bottleneck map includes:

- remaining host extraction inside rollout steps
- observation or action materialization performed only for passive publication
- episodic-memory query and append cost
- PPO minibatch assembly and optimizer wall time
- telemetry publication overhead

This means actor-side runtime code must now be treated like systems code.
The environment is no longer the only place where performance decisions matter.

## 16. Actor Telemetry And Observability

The actor publishes coarse telemetry for:

- policy and value optimization metrics
- reward summaries
- throughput and timing summaries
- dashboard heartbeats when optimizer windows would otherwise starve the UI

Canonical observer rule:

- passive matrix publication stays low-volume and actor-filtered
- the default cadence is intentionally much lower than UI render cadence but
  high enough to feel live during training inspection

Important rule:

- observability is allowed
- but it must remain coarse, safe, and subordinate to rollout throughput

## 17. Single-Pipeline Invariant

The canonical trainer uses one end-to-end pipeline only:

- one compiled-scene backend
- one sacred actor pipeline
- one PPO update surface
- one benchmarked production runtime

There are no equal-status alternate production trainers.

## 18. Behavioral Cloning Pre-Training

**Module:** `training/bc_trainer.py`

Behavioral cloning provides a supervised pre-training path that trains the same
sacred `CognitiveMambaPolicy` from human navigation demonstrations.  The BC
trainer shares the `evaluate_sequence()` forward pass with PPO but uses
maximum-likelihood loss instead of the clipped surrogate objective.

Architectural properties:

- trains the **identical** pipeline: `RayViTEncoder` → `TemporalCore` →
  `ActorCriticHeads`
- uses BPTT sequences to preserve temporal-core context
- freezes `log_std` during BC to preserve exploration capacity for subsequent
  PPO fine-tuning
- produces standard v3 checkpoints loadable by `PpoTrainer.load_training_state()`
- supports `--checkpoint` for incremental improvement across scenes

Demonstration capture occurs in the auditor project (`DemonstrationRecorder`),
not in the actor.  This preserves the boundary: the auditor records passively,
the actor trains.  The data exchange format is `.npz` archives with observations
matching `(N, 3, Az, El)` and actions matching `(N, 4)` in normalised policy
space.

### 18.1 BC Training Algorithm

1. Load all `.npz` demonstration files from the demonstrations directory.
2. Chunk concatenated observations and actions into BPTT sequences.
3. Shuffle and iterate in minibatches through the full dataset per epoch.
4. Compute loss: `L = -E[log pi(a|o)] - beta * H(pi)` where `H` is entropy.
5. Gradient clip and Adam optimiser step.
6. Save v3 checkpoint with fresh RND weights.

### 18.2 BC Commands

```powershell
# Single-scene workflow
uv run --project projects/auditor explore --record --gmdag-file <scene.gmdag>
uv run --project projects/actor brain bc-pretrain

# Incremental multi-scene
uv run --project projects/actor brain bc-pretrain --checkpoint artifacts/checkpoints/bc_base_model.pt

# Multi-scene exploration then training
./scripts/run-explore-scenes.ps1
./scripts/run-bc-pretrain.ps1
```

## 19. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/SIMULATION.md`
- `docs/DATAFLOW.md`
- `docs/PERFORMANCE.md`
- `docs/TRAINING.md`
