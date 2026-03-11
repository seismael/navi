# PERFORMANCE.md — Throughput And Runtime Gates

This document defines Navi's performance acceptance rules.
Correctness is mandatory, but canonical status is decided by measured end-to-end
training throughput.

## 1. Priority Order

1. rollout throughput on the canonical trainer
2. no-stall execution across rollout and PPO update phases
3. GPU residency for already-batched world-query and rollout data
4. contract preservation at service and diagnostic boundaries

## 2. Current Production Truth

The canonical production runtime is:

1. `.gmdag` compilation in `projects/voxel-dag`
2. CUDA sphere tracing in `projects/torch-sdf`
3. `SdfDagBackend` adaptation in `projects/environment`
4. direct in-process trainer execution in `projects/actor`

The imported docs were right about one key systems rule: once the environment
kernel is fast, the bottleneck moves to dataflow above it.

That is the current state of Navi.

## 3. Adopted Performance Rules

### 3.1 Measure End To End

Microkernel speedups are not enough.

- benchmark claims must be made against the real trainer when the change affects the hot path
- service-mode ZMQ measurements do not replace in-process rollout measurements
- unverified headline numbers are not carried into canonical docs
- bounded canonical benchmark summaries must treat the unified actor log at `logs/navi_actor_train.log` as the authoritative source for final optimizer and checkpoint timing when redirected stdout artifacts omit tail completion lines

### 3.2 Prefer Tensor-Native Seams

If the backend already produces batched CUDA tensors, the next optimization is
not more environment experimentation. It is removal of needless staging.

Forbidden patterns on the canonical hot path:

- GPU -> CPU -> GPU observation bounces
- per-actor Python object rebuilding when batched tensors already exist
- per-actor hidden-state or episode-state dictionaries when the rollout horizon is fixed and a batched tensor can carry the same state
- re-normalizing the same embedding batch separately for episodic-memory query and episodic-memory add in one rollout tick
- per-actor device synchronization for sparse telemetry fields when one batched CPU mirror can serve the same selected-actor publish tick
- fragmented completed-episode host copies when one packed done-event mirror can carry actor id, episode return, and episode length for sparse publication
- host copies of rollout scalar tensors that are not consumed on the current tick, including shaped-reward mirrors when step telemetry is disabled
- per-tick reward-sum device synchronization when aggregate reward accounting can stay on-device until final metric materialization
- repeated learner metric `.item()` extraction at PPO epoch end when one packed metric mirror can serve debug logging and returned `PpoMetrics`
- new transport hops inserted into direct trainer stepping
- per-step allocation churn when reusable buffers are practical

### 3.3 Treat PPO Update As Part Of The Hot Path

The imported project framed performance mostly around the ray kernel.
Navi now documents a stricter rule: PPO update cost is part of production
throughput, not a secondary concern.

Current high-value work therefore includes:

- minimizing device copies during minibatch assembly
- stacking rollout tensors once per PPO update and reusing them across epochs
- writing rollout ticks into reusable `(actors, time, ...)` device slabs instead of appending actor-by-actor into Python-managed buffers
- passing cached BPTT sequence views directly into learner sequence evaluation instead of flattening and reshaping them again inside the PPO epoch loop
- keeping shuffle indices on the same device as the sampled tensors
- materializing PPO epoch summary metrics through one packed host transfer and reusing that mirror for debug logging and returned metrics
- attributing PPO update wall time on the canonical learner path into minibatch prep, policy evaluation, backward, gradient clip, optimizer step, and RND step means
- requiring sequence-native minibatches on the canonical BPTT path instead of flattening and reshaping minibatch tensors again inside the learner
- normalizing rollout advantages once per finalized PPO update buffer and reusing the cached normalized tensor across epochs instead of recomputing normalization during each sampling pass
- skipping hidden-state minibatch reconstruction on the canonical BPTT path because the active temporal core ignores hidden state
- removing trainer-side rollout hidden carry, reset, and PPO bootstrap plumbing on the canonical path because the active temporal core ignores hidden state during both rollout and sequence execution
- removing batched rollout-buffer hidden allocation, cache population, and minibatch emission on the canonical PPO path because the active temporal core never consumes hidden starts
- removing per-actor fallback `TrajectoryBuffer` allocation from the canonical `MultiTrajectoryBuffer(capacity=...)` path so batched rollout storage does not pay dual-surface Python overhead
- gating completed-episode host extraction behind real sparse episode publication need so done-actor reset bookkeeping stays on-device when episode telemetry is disabled or filtered out
- gating initial and live observation materialization behind real dashboard publication need so the canonical trainer does not ask the runtime to build `DistanceMatrix` objects when no observation stream will publish them
- hardening attribution toggles so perf-only telemetry configurations remain valid

### 3.4 Keep Observability Passive

Human-facing tooling may consume data, but it must not shape the runtime.

- dashboards may receive coarse heartbeat republishing during optimizer windows
- those heartbeats are diagnostic-only, not new environment steps
- frame dropping is allowed; training stall is not

## 4. Current Bottleneck Interpretation

The current architecture has already removed major environment-side waste.
The confirmed remaining bottlenecks are actor/runtime dataflow and optimizer
wall time.

Interpret end-to-end SPS using both environment and actor telemetry:

- if `env_ms` is low but actor transfer or update time rises, the bottleneck is above the simulation layer
- `ppo_update_ms` is the primary wall-clock bottleneck once environment stepping is no longer dominant
- `host_extract_ms` is cleanup work unless it grows into the same order as reward shaping or PPO update cost
- telemetry toggles are mainly robustness and attribution tools at the current cadence

The latest attribution matrix in
`artifacts/benchmarks/attribution-matrix/20260309-095734/summary.csv` supports
that interpretation.

One concrete remaining environment-to-actor seam still worth cleaning is the
MJX speed-throttling dependency on previous depth. On tensor-native trainer
paths, that seam should read the already-kept previous-depth tensor and only pay
for host materialization when a published `DistanceMatrix` is actually required.

## 5. Benchmark Gates

### 5.1 Fleet Throughput

- standard fleet: `4` actors
- acceptance floor: `>= 60 SPS`
- major runtime changes must be compared against the current canonical trainer, not only isolated kernels

### 5.2 Runtime Proof Requirements

Any change to the canonical `sdfdag` stack must continue to demonstrate all of
the following:

1. `.gmdag` compilation remains reproducible from Navi workflows
2. `torch_sdf.cast_rays()` executes correctly on CUDA with batched actor rays
3. observation semantics remain contract-correct
4. end-to-end training throughput improves or materially advances the repository toward the fleet floor
5. the hot path reduces `.cpu()`, `.numpy()`, scalar extraction, or Python object assembly relative to the previous baseline

### 5.3 Benchmark Artifact Recovery

Bounded benchmark artifacts may be collected from different capture surfaces during
investigation. Only one rule matters for canonical comparison:

- rollout cadence metrics such as `sps`, `env_ms`, `trans_ms`, `host_extract_ms`, and `telemetry_publish_ms` may be computed from the bounded run artifact log itself
- final optimizer timing such as `ppo_update_ms` must be recovered from the same artifact when present, or from `logs/navi_actor_train.log` for the matching trainer start window when the artifact tail is incomplete
- comparison notes must state which source supplied `ppo_update_ms` whenever fallback recovery was needed

Environment-layer benchmark captures should prefer the structured `uv run navi-environment bench-sdfdag --json ...` summary so actor count, warmup, resolution, and measured throughput are recorded explicitly in one machine-readable artifact.
When run-to-run variance is non-trivial on the local machine, canonical environment comparison should prefer `bench-sdfdag --repeats N --json` and compare the reported median `measured_sps` rather than a single run.

### 5.4 Experimental Work Classification

`TSDF.md` and the imported docs describe multiple classes of change. Navi does
not treat them equally.

Accepted and already integrated:

- fixed horizon alignment between `EnvironmentConfig.max_distance` and CUDA tracing termination
- starvation and proximity shaping derived from existing spherical observations

Benchmark-gated only:

- compiler-side truncation metadata
- alternate leaf-distance storage policies
- Morton or other layout redesigns
- stream-overlap or double-buffering changes that complicate the single canonical trainer

### 5.5 Nightly Validation Thresholds

The canonical overnight validation flow should treat these as different classes of signal:

- hard failures: process crash, non-finite training metrics, failed bounded qualification, failed checkpoint resume proof, or stalled checkpoint production
- soft warnings: attach instability, environment benchmark drift, and degraded but still finite rollout throughput

Nightly summaries should compare against the last accepted baseline instead of treating one run in isolation.

## 6. Current Reference Baselines

These values are current reference points, not final goals.

| Profile | Resolution | Steps | Mean SPS | Mean Zero-Wait Ratio | Soft Warnings |
| --- | --- | --- | --- | --- | --- |
| Baseline long run | `256x48` | 29,600 | `17.32` | `19.16%` | `269` |

## 7. Current Implementation Priorities

1. optimize canonical PPO learner and minibatch flow before further environment micro-optimization
2. keep tensor-native observation and action stepping on the direct trainer path
3. remove remaining avoidable host staging inside the rollout tick, including
	tensor-path kinematic seams that still force previous-depth NumPy materialization
4. preserve coarse, safe observability while keeping dashboards passive

## 8. Canonical Decision Rule

No design becomes canonical because it is elegant on paper.
It becomes canonical when it is contract-correct and faster in real training.
