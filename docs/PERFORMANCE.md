- attributing PPO update wall time on the canonical learner path into minibatch fetch, minibatch prep, policy evaluation, backward, gradient clip, optimizer step, RND step, post-update stats accumulation, end-of-epoch metric materialization, and progress-callback means
- using synchronized CUDA event timing for diagnostic PPO profiling runs when host timers leave large unattributed optimizer wall time, while keeping the default throughput path free of forced synchronization
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
- GPU -> CPU -> GPU action bounces for already-CUDA policy outputs
- per-actor Python object rebuilding when batched tensors already exist
- per-actor hidden-state or episode-state dictionaries when the rollout horizon is fixed and a batched tensor can carry the same state
- per-actor Python branching over CUDA reset masks when one batched reset-index extraction can drive the same state transition
- re-normalizing the same embedding batch separately for episodic-memory query and episodic-memory add in one rollout tick
- per-actor device synchronization for sparse telemetry fields when one batched CPU mirror can serve the same selected-actor publish tick
- fragmented completed-episode host copies when one packed done-event mirror can carry actor id, episode return, and episode length for sparse publication
- host copies of rollout scalar tensors that are not consumed on the current tick, including shaped-reward mirrors when step telemetry is disabled
- per-tick reward-sum device synchronization when aggregate reward accounting can stay on-device until final metric materialization
- repeated learner metric `.item()` extraction at PPO epoch end when one packed metric mirror can serve debug logging and returned `PpoMetrics`
- new transport hops inserted into direct trainer stepping
- per-step allocation churn when reusable buffers are practical
- collision-triggered subset re-casts when rollback to the previous safe pose already preserves the non-terminal wall-contact signal the actor needs

### 3.3 Treat PPO Update As Part Of The Hot Path

The imported project framed performance mostly around the ray kernel.
Navi now documents a stricter rule: PPO update cost is part of production
throughput, not a secondary concern.

Current high-value work therefore includes:

- minimizing device copies during minibatch assembly
- stacking rollout tensors once per PPO update and reusing them across epochs
- writing rollout ticks into reusable `(actors, time, ...)` device slabs instead of appending actor-by-actor into Python-managed buffers
- passing cached BPTT sequence views directly into learner sequence evaluation instead of flattening and reshaping them again inside the PPO epoch loop
- sampling each canonical BPTT minibatch with one device-side sequence gather per tensor family, then deriving flattened transition views from that selected sequence block instead of gathering both sequence and flat tensors separately
- keeping shuffle indices on the same device as the sampled tensors
- materializing PPO epoch summary metrics through one packed host transfer and reusing that mirror for debug logging and returned metrics
- keeping `actor.training.ppo.update` presence for mode/status detection while treating PPO loss fields as explicit diagnostics, so the canonical default path does not pay an end-of-epoch CUDA-to-host sync for update scalars with no active live consumer
- attributing PPO update wall time on the canonical learner path into minibatch prep, policy evaluation, backward, gradient clip, optimizer step, and RND step means
- requiring sequence-native minibatches on the canonical BPTT path instead of flattening and reshaping minibatch tensors again inside the learner
- normalizing rollout advantages once per finalized PPO update buffer and reusing the cached normalized tensor across epochs instead of recomputing normalization during each sampling pass
- skipping hidden-state minibatch reconstruction on the canonical BPTT path because the production trainer currently evaluates supported temporal cores with `hidden=None`
- skipping done-mask minibatch emission on the canonical batched BPTT path because the active temporal cores ignore sequence done masks during PPO evaluation
- removing trainer-side rollout hidden carry, reset, and PPO bootstrap plumbing on the canonical path because the production trainer currently keeps one hidden-free rollout/update seam across supported temporal backends
- removing batched rollout-buffer hidden allocation, cache population, and minibatch emission on the canonical PPO path because the production trainer does not currently consume hidden starts on the update surface
- removing per-actor fallback `TrajectoryBuffer` allocation from the canonical `MultiTrajectoryBuffer(capacity=...)` path so batched rollout storage does not pay dual-surface Python overhead
- gating completed-episode host extraction behind real sparse episode publication need so done-actor reset bookkeeping stays on-device when episode telemetry is disabled or filtered out
- gating initial and live observation materialization behind real dashboard publication need so the canonical trainer does not ask the runtime to build `DistanceMatrix` objects when no observation stream will publish them
- hardening attribution toggles so perf-only telemetry configurations remain valid

### 3.4 Treat The Temporal Core As Hot-Path Infrastructure

The actor temporal core is not a side dependency. It is now part of the
throughput-critical training infrastructure.

Recent end-to-end profiling established that:

- environment-side CUDA execution is already fast enough that rollout can hold roughly `~110 SPS`
- rollout-buffer sequence view work is not the dominant remaining PPO stall
- synchronized CUDA-event timing still leaves a large host-side optimizer gap when the temporal core runs through Python `mambapy`

That result matters architecturally. It means the dominant remaining cost is not
hidden GPU work. It is host-side interpreter, dispatcher, autograd-graph, and
kernel-launch overhead created by the unfused temporal-core path.

Therefore the canonical performance rule is:

- one canonical trainer surface is mandatory on the production actor hot path
- on the active hardware, that surface now defaults to the cuDNN GRU path because repeated profiled bounded runs beat `mambapy` on `steady_sps`, `ppo_update_ms`, and `gpu_backward_ms`
- `mambapy` may be selected explicitly on that same surface for controlled comparisons, but performance conclusions must compare like-for-like bounded runs with only the temporal-core selector changed
- benchmark-proven end-to-end trainer wins may replace GRU or another supported backend as the canonical default, but only when config defaults, wrappers, validation, and docs are updated together so the repo keeps one performance truth
- throughput attribution must measure the selected runtime on the real trainer surface before assigning the remaining `~100 SPS` ceiling to the temporal core alone

Why this is mandatory in practice:

- the PPO learner calls `evaluate_sequence()` on contiguous BPTT minibatches
- an unfused Python Mamba path decomposes that sequence into many smaller tensor ops
- the CPU pays for every dispatcher entry, argument check, autograd node, and launch submission
- the GPU may still execute only a smaller fraction of the total wall time
- a future fused Mamba-2 build may still collapse that sequence math further, but the current production question is how much of the remaining wall time comes from rollout cadence, environment variance, telemetry, and PPO update structure while the active Mamba path is in place

Future switch-back rule:

- fused Mamba-2 may be introduced later, but only after the supported environment exists and the real bounded trainer proves that it beats the current Windows-friendly Mamba baseline without reopening deployment friction on the active platform

### 3.5 Keep Observability Passive

Human-facing tooling may consume data, but it must not shape the runtime.

- dashboards may receive coarse heartbeat republishing during optimizer windows
- those heartbeats are diagnostic-only, not new environment steps
- frame dropping is allowed; training stall is not

## 4. Current Bottleneck Interpretation

The current architecture has already removed major environment-side waste.
The confirmed remaining bottlenecks are actor/runtime dataflow and optimizer
wall time.

On the environment side, the remaining Python-adjacent cost is no longer the
ray kernel launch itself. It is eager PyTorch dispatch around the kernel
boundary when many small tensor ops prepare directions, normalize outputs, and
assemble reward terms. The canonical fix for that class of overhead is
`torch.compile` on tensor-only helper graphs that stay outside the custom CUDA
extension boundary.

On older GPUs, that fix may be unavailable in practice. The current Windows
MX150 surface is `sm_61`, and PyTorch inductor/triton does not support that GPU
for the compiled CUDA path. Canonical behavior on unsupported hardware is to
warn, report that compile was requested but inactive, and continue on the eager
tensor path instead of crashing the runtime.

Interpret end-to-end SPS using both environment and actor telemetry:

- if `env_ms` is low but actor transfer or update time rises, the bottleneck is above the simulation layer
- `ppo_update_ms` is the primary wall-clock bottleneck once environment stepping is no longer dominant
- `host_extract_ms` is cleanup work unless it grows into the same order as reward shaping or PPO update cost
- telemetry toggles are mainly robustness and attribution tools at the current cadence

The latest attribution matrix in
`artifacts/benchmarks/attribution-matrix/20260309-095734/summary.csv` supports
that interpretation.

Canonical PPO update logs now also report the minibatch `updates=` count plus
both averaged and summed learner-stage timings. That surface now includes
`stats=` for per-minibatch KL, clip-fraction, and metric accumulation work plus
`loop=` for untimed per-update outer-loop overhead, `setup=` for learner setup,
and `finalize=` for the end-of-epoch metric materialization transfer. The total
surface now also reports `learner=` with `learner_gap=` and `trainer_gap=` so the
remaining PPO wall time can be split between untimed work inside the learner and
work outside the learner on the trainer boundary. When setup dominates, the log
also expands that bucket into `setup(policy=... rnd=... mode=... params=... other=...)`
so first-update latency can be attributed before changing the canonical runtime.
The canonical trainer should eagerly prime learner optimizer and parameter-cache
state before the first timed PPO update so one-time optimizer construction does
not pollute the production update surface.

When `--profile-cuda-events` is enabled, the optimizer log also emits a
diagnostic `diag(...)` suffix with GPU execution totals and per-stage host-gap
deltas for eval, backward, clip, optimizer, RND, and stats. Use that surface to
separate Python/autograd orchestration overhead from actual GPU execution time on
the active Mamba learner path.

One concrete remaining environment-to-actor seam still worth cleaning is the
MJX speed-throttling dependency on previous depth. On tensor-native trainer
paths, that seam should read the already-kept previous-depth tensor and only pay
for host materialization when a published `DistanceMatrix` is actually required.

One concrete remaining actor-side eager seam is reward shaping inside the
rollout loop. When that shaping remains a pure tensor helper graph, the
canonical fix is the same as on the environment side: request `torch.compile`
on supported GPU/compiler stacks, report whether compile was merely requested or
actually active, and keep a clean eager fallback when the active hardware stack
cannot compile the graph.

The validated March 13 environment refactor also establishes two concrete rules
for the canonical tensor path:

- batch reset seeding should reuse one batched ray cast plus one batched state write for all reset actors in the tick
- indexed CUDA state updates must use proper indexed assignment semantics rather than `.copy_()` on advanced-indexed temporary tensors
- actor-index routing on the canonical tensor path should stay tensor-native through batch stepping, reset seeding, and observation casting; convert to Python only at the final public result or selected publish seam
- final public `StepResult` materialization on the tensor path should use one packed host mirror for result rows rather than separate per-field `.cpu().tolist()` extractions
- eager tensor micro-kernels around `torch_sdf.cast_rays()` should be fused with `torch.compile` on the canonical path when the helper graph remains pure PyTorch tensor code

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
The same rule now applies to bounded temporal-core comparisons: `scripts/run-temporal-compare.ps1` should prefer repeated direct-trainer runs and compare the emitted median `steady_sps` and median `ppo_update_ms` rather than one wrapper pass.
When `-ProfileCudaEvents` is used on that surface, comparison artifacts should also compare median learner `backward_ms` and median `gpu_backward_ms` before attributing the remaining stall to the active temporal runtime.

### 5.4 Experimental Work Classification

`TSDF.md` and the imported docs describe multiple classes of change. Navi does
not treat them equally.

Accepted and already integrated:

- fixed horizon alignment between `EnvironmentConfig.max_distance` and CUDA tracing termination
- starvation and proximity shaping derived from existing spherical observations
- two-group trainer-side CUDA stream overlap on the canonical PPO rollout tick, with actor subset routing and deferred episodic-memory adds so same-tick memory queries still observe the pre-add state across the full fleet
- macro-cell empty-space caching in `projects/torch-sdf/cpp_src/kernel.cu`, where repeated samples inside the same empty DAG child cell reuse cached bounds and advance to the cell exit without a fresh root descent

Benchmark-gated only:

- compiler-side truncation metadata
- alternate leaf-distance storage policies
- Morton or other layout redesigns

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
