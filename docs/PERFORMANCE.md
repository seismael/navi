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
- new transport hops inserted into direct trainer stepping
- per-step allocation churn when reusable buffers are practical

### 3.3 Treat PPO Update As Part Of The Hot Path

The imported project framed performance mostly around the ray kernel.
Navi now documents a stricter rule: PPO update cost is part of production
throughput, not a secondary concern.

Current high-value work therefore includes:

- minimizing device copies during minibatch assembly
- stacking rollout tensors once per PPO update and reusing them across epochs
- passing cached BPTT sequence views directly into learner sequence evaluation instead of flattening and reshaping them again inside the PPO epoch loop
- keeping shuffle indices on the same device as the sampled tensors
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

## 6. Current Reference Baselines

These values are current reference points, not final goals.

| Profile | Resolution | Steps | Mean SPS | Mean Zero-Wait Ratio | Soft Warnings |
| --- | --- | --- | --- | --- | --- |
| Baseline long run | `256x48` | 29,600 | `17.32` | `19.16%` | `269` |
| Focused pass | `128x24` | 5,000 | `20.62` | `0.00%` | `2` |

## 7. Current Implementation Priorities

1. optimize canonical PPO learner and minibatch flow before further environment micro-optimization
2. keep tensor-native observation and action stepping on the direct trainer path
3. remove remaining avoidable host staging inside the rollout tick, including
	tensor-path kinematic seams that still force previous-depth NumPy materialization
4. preserve coarse, safe observability while keeping dashboards passive

## 8. Canonical Decision Rule

No design becomes canonical because it is elegant on paper.
It becomes canonical when it is contract-correct and faster in real training.
