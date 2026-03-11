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
- fixed spherical positional structure rather than ad hoc flattened features

Architectural consequence:

- runtime upgrades must preserve spherical observation semantics
- imported ideas such as foveation remain optional future work and do not alter
  the current production contract

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
- the active `Mamba2TemporalCore` ignores hidden state, so canonical PPO learner minibatch prep must not rebuild or copy hidden-state batches on the update path
- the active `Mamba2TemporalCore` also ignores rollout hidden state, so the canonical trainer must not carry, reset, store, or bootstrap `hidden_batch` tensors on the rollout/update hot path
- because the canonical temporal core does not consume hidden starts, batched rollout buffers and minibatches must not allocate hidden-state slabs or emit hidden-state minibatch fields on the canonical PPO path
- when `MultiTrajectoryBuffer` is running in its canonical batched mode (`capacity` set), it must not also allocate per-actor fallback `TrajectoryBuffer` wrappers; the production rollout store is the batched slab itself
- completed-episode host extraction must happen only for actors whose sparse episode telemetry will actually be published; canonical reset bookkeeping for done actors stays on-device
- initial tensor resets and live runtime steps must materialize `DistanceMatrix` observations only for actors whose dashboard observation stream will actually publish; when observation streaming is disabled, canonical training must request no published observations at all

This preserves the sacred actor topology while removing avoidable Python
orchestration above the already-batched CUDA environment step.

## 10. Temporal Core

**Module:** `mamba_core.py`

Current canonical runtime uses `mambapy` Mamba on the active Windows CUDA
profile.

Current architectural status:

- one canonical temporal backend in production
- benchmark-gated comparison tooling is allowed outside the production path
- no runtime fallback branch on the production trainer

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

## 18. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/SIMULATION.md`
- `docs/DATAFLOW.md`
- `docs/PERFORMANCE.md`
- `docs/TRAINING.md`
