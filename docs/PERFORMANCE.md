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

Current machine-level truth is now sharper:

- the environment-only `bench-sdfdag` surface still scales materially better
	than the full trainer as observation resolution rises
- the canonical 4-actor trainer remains near the production floor at
	`256x48`, degrades moderately at `512x96`, and fails at `768x144` on the
	active MX150 surface because actor-side attention memory exhausts the
	available `2 GB` VRAM

## 3. Adopted Performance Rules

### 3.1 Measure End To End

Microkernel speedups are not enough.

- benchmark claims must be made against the real trainer when the change affects the hot path
- service-mode ZMQ measurements do not replace in-process rollout measurements
- unverified headline numbers are not carried into canonical docs
- bounded canonical benchmark summaries must treat the stable actor log at `logs/navi_actor_train.log` and the matching run-scoped log under the active run root as the authoritative source for final optimizer and checkpoint timing when redirected stdout artifacts omit tail completion lines
- canonical trainer investigations should prefer the machine-readable run metrics under `metrics/` for correlation across rollout, update, checkpoint, and wrapper lifecycle events before falling back to raw log scraping
- canonical training attribution should combine phase wall time with coarse process and CUDA resource snapshots so bottleneck review can distinguish rollout cadence, optimizer cost, checkpoint cost, corpus preparation cost, and orchestration overhead on the same run timeline

### 3.2 Prefer Tensor-Native Seams

If the backend already produces batched CUDA tensors, the next optimization is
not more environment experimentation. It is removal of needless staging.

Forbidden patterns on the canonical hot path:

- GPU -> CPU -> GPU observation bounces
- GPU -> CPU -> GPU action bounces for already-CUDA policy outputs
- pinned-CPU or host-first rollout slabs on the canonical trainer when the same rollout state can remain on CUDA
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
- viewer-driven observation remaps, re-normalization, or contract branching inside environment or actor hot-path code
- dashboard-required CPU materialization or cadence checks that alter rollout math instead of staying inside passive publication seams
- blind removal of validated `torch-sdf` leaf, void, or prefix caches without a reproduced current-branch regression and a benchmark-backed replacement
- per-step direction-norm revalidation inside `cast_rays()` when ray directions are mathematically guaranteed normalized (e.g. yaw-rotated unit vectors); the four GPU→CPU pipeline drains this causes are eliminated by `skip_direction_validation=True` on the hot path

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
- rewriting grouped rollout overlap only when the design preserves actor-local `Obs -> Action -> Next Obs` ordering and shows an end-to-end trainer gain on the canonical surface

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
- on the active hardware, that surface now defaults to the pure-PyTorch Mamba-2 SSD path because a 25K-step head-to-head comparison proved significantly better learning quality (reward_ema -0.88 vs GRU's -1.48) with only a modest throughput trade-off (~72 SPS vs ~100 SPS)
- `gru` and `mambapy` may be selected explicitly on that same surface for controlled comparisons, but performance conclusions must compare like-for-like bounded runs with only the temporal-core selector changed
- benchmark-proven end-to-end trainer wins may replace Mamba2 SSD or another supported backend as the canonical default, but only when config defaults, wrappers, validation, and docs are updated together so the repo keeps one performance truth
- throughput attribution must measure the selected runtime on the real trainer surface before assigning the remaining throughput ceiling to the temporal core alone
- future fused temporal-core work may reduce part of PPO update cost, but it does not remove RayViT token-attention scaling on higher observation profiles

Why this is mandatory in practice:

- the PPO learner calls `evaluate_sequence()` on contiguous BPTT minibatches
- an unfused Python Mamba path decomposes that sequence into many smaller tensor ops
- the CPU pays for every dispatcher entry, argument check, autograd node, and launch submission
- the GPU may still execute only a smaller fraction of the total wall time
- a future fused Mamba-2 build may still collapse that sequence math further, but the current production question is how much of the remaining wall time comes from rollout cadence, environment variance, telemetry, PPO update structure, and the now-dominant RayViT encoder path on the active Mamba2 SSD trainer surface

Future switch-back rule:

- fused Mamba-2 (`mamba-ssm`) may be introduced later, but only after the supported environment exists and the real bounded trainer proves that it beats the current Mamba2 SSD baseline without reopening deployment friction on the active platform

### 3.5 Keep Observability Passive

Human-facing tooling may consume data, but it must not shape the runtime.

- dashboards use a dedicated `zmq.CONFLATE` observation socket so the displayed
  frame is always the latest published observation regardless of UI pauses,
  garbage collection, or window operations
- a separate telemetry socket with `RCVHWM=50` carries ordered action and
  telemetry events without dropping intermediate history
- dashboards may receive coarse heartbeat republishing during optimizer windows
- those heartbeats are diagnostic-only, not new environment steps
- frame dropping is allowed; training stall is not
- rendering occurs only when a genuinely new observation arrives; redundant
  re-renders on unchanged frames are eliminated to keep CPU free for queue
  draining and status updates
- observation publication is optional and passive; canonical training and inference must remain correct and throughput-safe when the auditor is absent
- viewer requirements must be implemented by observer-side transforms over the published spherical contract, not by changing core math or environment semantics
- run-aware manifests, logs, and metrics are required for reviewability, but they must remain append-only side effects around the hot path rather than new gating work inside per-step rollout math
- coarse resource measurement follows the same rule: capture CPU-process and CUDA allocator state at major phase boundaries and the existing logging cadence, but never add per-step polling or repeated external GPU queries inside the rollout loop

## 4. Current Bottleneck Interpretation

The current architecture has already removed major environment-side waste.
The confirmed remaining bottlenecks are actor/runtime dataflow and optimizer
wall time.

### 4.0 GPU Compute Utilization Reality (March 2026)

On the active MX150 (`sm_61`, 3 SMs, 384 CUDA cores), measured GPU compute
utilization during canonical training stays well below 100% despite full VRAM
occupancy. There are three structural causes, none of which are fixable by
removing CPU sync barriers.

**Cause 1: Eager PyTorch dispatcher overhead.** Without `torch.compile` (which
requires `sm_70+`), every PyTorch operation dispatches a separate CUDA kernel
through the Python runtime. The canonical rollout tick launches **~72-90
individual kernels** and each PPO minibatch update launches **~165-376 kernels**.
Each kernel dispatch incurs ~10-100μs of Python-side overhead where the GPU sits
idle waiting for the next launch. This aggregates to **1-5ms of GPU idle time per
rollout tick** and **2-18ms per PPO minibatch** purely from dispatcher gaps.

**Cause 2: Mamba2 SSD kernel count.** The canonical Mamba2 SSD temporal core is
pure-PyTorch and dispatches **55-60 separate CUDA kernels** per forward pass
(`einsum`, `cumsum`, `exp`, segment-sum loops). By comparison, the GRU temporal
core uses a **single fused cuDNN kernel** (2-4 dispatches). This 15x kernel-count
difference is the largest single source of Mamba2's throughput disadvantage
relative to GRU on `sm_61`.

**Cause 3: Rollout-PPO serialization.** The rollout phase and PPO update phase
run fully serially. During the ~1000ms PPO window, the environment GPU kernels
are idle. During the ~6000ms rollout window, the PPO optimizer is idle. No overlap
is possible without a double-buffer architecture that has not yet been implemented.

These three structural causes explain why GPU compute utilization remains low even
though the canonical hot path contains **zero unnecessary GPU→CPU synchronization
barriers**. The direction-norm validation bypass (`skip_direction_validation=True`)
removes four real pipeline drains per `cast_rays()` call, but these were ~2ms
out of a ~45ms tick — a meaningful cleanup but not a visible utilization shift.

**What would actually move the needle:**

| Change | Impact | Status |
| --- | --- | --- |
| `torch.compile` on RayViT + reward helpers | Eliminate ~50% of dispatcher gaps | **Blocked:** requires `sm_70+` |
| `mamba-ssm` fused Triton kernels | Eliminate 55-60 Mamba2 dispatcher gaps | **Blocked:** not available on Windows |
| GRU temporal core | 2-4 dispatches vs 55-60 (Mamba2) | Available but lower learning quality |
| PPO/rollout double-buffer overlap | Eliminate ~1000ms GPU idle per PPO window | **Future:** complex architecture work |
| CUDA graph capture of step path | Replay entire step as single submission | **Infeasible:** data-dependent control flow |

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

### 4.0.1 Apr 2026 Optimization Closeout (sm_61)

The H1/H2/H3 levers tracked in TODO.md Phase H were instrumented and measured
on the active MX150. Bake-off
`artifacts/benchmarks/cadence-compare/cadence-compare-20260420-230927/comparison-summary.json`
recorded mamba2+gru × rollout 256/512 × 3 repeats × 4096 steps, with each run
starting from a fresh policy via the new `--no-auto-resume` CLI flag:

| Configuration                 | Mar 2026 baseline | Apr 2026 mean | Apr 2026 median |
|-------------------------------|------------------:|--------------:|----------------:|
| mamba2 + 256 (current default)| ~72 SPS           | 68.5          | 74.4            |
| mamba2 + 512                  | not measured      | 72.4          | 75.3            |
| gru + 256                     | ~100 SPS          | 69.4          | 79.6            |
| gru + 512                     | not measured      | 87.3          | 92.7            |

**Conclusions:**

- Cadence tuning (lever A) on the canonical mamba2 core is **not a win** on
  this hardware (+6%, within noise). `rollout_length` was left at 256.
- Throughput-only winner is gru + 512 (+21–27% vs mamba2+256), but it was
  **not promoted** because the original Mar 2026 25K-step decision was made
  on learning quality (mamba2 reward_ema −0.88 vs GRU −1.48) per §2.7
  Promotion Rule. A short bake-off cannot disprove that.
- The PPO sub-attribution timers (`loss_build_ms`, `zero_grad_ms`) added by
  H1 are in place but were not active in this run because
  `--profile-cuda-events` was not enabled. Re-running the wrapper with
  `-ProfileCudaEvents` is the cheap next step to populate the new fields.
- The GRU **regression vs Mar 2026** (~100 → ~87 SPS even at 2× cadence) is
  an unexplained signal that should be diagnosed before reading too much
  into Apr 2026 throughput numbers. Suspected sources are the Apr-only
  reward-shaping additions (info foraging, structure seeking, void grace
  period per §3.4), the spawn quality gate, and the mesh-repair pipeline.

This closeout confirms the §3.1.2 GPU Compute Utilization Standard
diagnosis: **no Python-side micro-optimization will materially move
throughput on sm_61.** Real gains require either sm_70+ hardware
(`torch.compile`), Linux + hardware-fused `mamba-ssm`, or PPO/rollout
double-buffer overlap. See TODO.md Phase J for the full pickup checklist on
new hardware.

## 4.1 Observation-Resolution Scaling

March 2026 bounded canonical trainer sweeps produced the current reference
results for the active 4-actor canonical trainer surface:

| Profile | Rays / Actor | Steady SPS | `env_ms` | `ppo_update_ms` | Outcome |
| --- | --- | --- | --- | --- | --- |
| `256x48` | `12,288` | `49.6` | `36.50` | `1,019.68` | healthy bounded run |
| `384x72` | `27,648` | `49.34` | `39.96` | `1,249.87` | healthy bounded run |
| `512x96` | `49,152` | `43.96` | `50.64` | `17,731.88` | runnable, update-dominated |
| `768x144` | `110,592` | n/a | n/a | n/a | trainer OOM in actor attention |

Primary artifact roots:

- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-003916/`
- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-004714/`
- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-002948/768x144/repeat-01/train.log`

Interpretation rules:

- ray-count growth alone does not explain trainer slowdown
- `env_ms` rises with the larger sphere, but `ppo_update_ms` rises much faster
- the active failure boundary at `768x144` is `torch.nn.MultiheadAttention`
	allocation, which localizes the current limit to the actor encoder path

The actor-side reason is straightforward. `RayViTEncoder` uses `patch_size=8`,
so token count grows from `192` at `256x48` to `768` at `512x96` and `1728` at
`768x144`. Full self-attention then grows roughly with the square of that token
count.

### 4.2 SphericalCNN Encoder Baselines (Apr 2026)

The `spherical_cnn` encoder replaces RayViT's quadratic self-attention
($O(N^2)$ on 192 tokens) with linear convolutions ($O(H \\cdot W \\cdot C)$)
on the spherical distance matrix treated as a 2D image with circular azimuth
wrapping.

**Architecture:** 3-ch input (depth, semantic, valid) → Stem → 2× depthwise-
separable blocks → 1× standard conv refinement → global average pool →
small projection MLP.

**Estimated costs (hardware-independent):**

| Metric | RayViT (baseline) | SphericalCNN | Ratio |
|---|---|---|---|
| Encoder FLOPs (forward) | ~150M | ~29M | 5.1× |
| Encoder parameters | ~306K | ~225K | 1.36× |

**Measured results — MX150 (`sm_61`, no `torch.compile`):**

| Metric | RayViT (median) | SphericalCNN (median) | Ratio |
|---|---|---|---|
| Encoder forward (4×256×48) | 5.0 ms | 4.3 ms | 1.2× |
| Encoder fwd+bwd (4×256×48) | 11.1 ms | 10.8 ms | 1.0× |
| Full-policy forward (4×256×48) | 17.0 ms | 15.8 ms | 1.1× |
| PPO minibatch BPTT (8×8 seq) | 46.9 ms | 44.8 ms | 1.0× |
| Training SPS (4096 steps, 4 actors) | 52.8 | 46.4 | 0.9× |
| Early reward_ema (2K steps) | −0.68 | −2.59 | — |

**Interpretation:** On `sm_61` without `torch.compile`, the SphericalCNN's
depthwise-separable convs dispatch many small CUDA kernels whose Python-side
eager-dispatch overhead consumes the 5.1× FLOP advantage.  The actual SPS
is comparable or slightly *worse* than RayViT, and early learning quality
favours RayViT.  The `encoder_backend` selector is integrated and tested,
but `rayvit` remains the canonical default per the §2.9 Default Rule.

**Predicted on `sm_70+` with `torch.compile`:** When the compiler fuses
the CNN's depthwise and pointwise conv dispatches into single kernels,
encoder forward latency is expected to drop to ~0.5 ms and full-policy
forward to ~12 ms, providing a 1.2–1.4× training SPS improvement.
A bounded 25K-step bake-off on `sm_70+` hardware is required before the
default encoder can be promoted.

Environment-only comparison remains important because it shows the CUDA runtime
is not the same ceiling. The environment path still remained benchmark-viable
above the trainer limit on the same machine, which means future work must keep
separating environment wins from actor-side bottlenecks.

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
the active learner path. That diagnostic surface now also splits `eval` into
encoder, temporal-core, and heads buckets so investigation can decide
whether the remaining wall time belongs to RayViT encoding, sequence-core work,
or actor/critic head evaluation before changing the canonical runtime.
The canonical RayViT encoder patch-token path should use direct strided patch
projection rather than explicit `view -> permute -> contiguous -> linear`
materialization when the same tokenization can be expressed as one convolution.

GPU-sampled `512x96` evidence reinforces the same conclusion.
`artifacts/benchmarks/resolution-compare/resolution-compare-20260317-004714/`
and `artifacts/benchmarks/resolution-compare/gpu-sample-512x96-20260317.csv`
show the active MX150 spending the PPO update window near full GPU utilization
while VRAM climbs to roughly `1963 MiB` out of `2048 MiB`. On this machine,
`512x96` is therefore a diagnostic comparison surface, not a new production
default.

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
- canonical trainer-facing tensor steps should not request `StepResult` materialization at all; reward, truncation, env-id, and episode-id columns must stay on-device and drive rollout bookkeeping directly unless a non-training caller explicitly needs public result objects
- eager tensor micro-kernels around `torch_sdf.cast_rays()` should be fused with `torch.compile` on the canonical path when the helper graph remains pure PyTorch tensor code
- canonical environment helper compilation should target pure tensor functions with explicit scalar parameters rather than bound instance methods so the compiled surface does not depend on Python object state capture in the rollout hot path

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
The same repeated-run rule applies to observation-resolution scaling: `scripts/run-resolution-compare.ps1` should compare median `steady_sps` and median `ppo_update_ms` across repeated bounded trainer runs for each `AzimuthxElevation` profile rather than judging one pass.
When `-ProfileCudaEvents` is used on that surface, comparison artifacts should also compare median learner `backward_ms` and median `gpu_backward_ms` before attributing the remaining stall to the active temporal runtime.
Observation-resolution comparison notes must also say whether the result came
from the environment-only `bench-sdfdag` surface or the full trainer surface.
Mixing those two interpretations is a documentation error.

### 5.4 Experimental Work Classification

`TSDF.md` and the imported docs describe multiple classes of change. Navi does
not treat them equally.

Accepted and already integrated:

- fixed horizon alignment between `EnvironmentConfig.max_distance` and CUDA tracing termination
- starvation and proximity shaping derived from existing spherical observations
- two-group trainer-side CUDA stream overlap on the canonical PPO rollout tick, with actor subset routing and deferred episodic-memory adds so same-tick memory queries still observe the pre-add state across the full fleet
- macro-cell empty-space caching in `projects/torch-sdf/cpp_src/kernel.cu`, where repeated samples inside the same empty DAG child cell reuse cached bounds and advance to the cell exit without a fresh root descent
- `cast_rays()` direction-norm validation bypass (`skip_direction_validation` parameter in C++/Python/sdfdag layers), eliminating four GPU→CPU synchronization barriers per step on the hot path; probe and inspection calls retain validation
- configurable rollout overlap group count (`rollout_overlap_groups` in `ActorConfig`), defaulting to `1` on the active MX150 hardware; `2` is available for larger GPUs with enough SMs to benefit from concurrent kernel execution

Those integrated surfaces carry explicit promotion constraints:

- the current grouped rollout implementation is only partial overlap until a benchmark-proven ping-pong rewrite lands; preserving per-actor ordering is mandatory
- on MX150 (3 SMs), 2-group rollout overlap causes ~47% throughput regression (43 SPS vs 82 SPS single-group) because halved batch size underutilizes the limited SM count; overlap groups > 1 are only beneficial on GPUs with enough SMs for concurrent kernel execution
- the current `torch-sdf` cache path is part of the validated runtime baseline and must not be removed based on imported theory alone
- CUDA graph capture of the full environment step path is not feasible due to data-dependent control flow (`nonzero()`, `.item()`, dynamic allocation, tensor-dependent loop bounds, `.tolist()`, advanced indexing with computed indices)

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
5. pursue `torch.compile` kernel fusion when hardware with `sm_70+` becomes available — this is the highest-ROI path to GPU compute utilization improvement
6. keep rollout overlap infrastructure (`rollout_overlap_groups`) ready for larger GPUs where multi-group dispatch can actually benefit from concurrent SM execution

## 8. Canonical Decision Rule

No design becomes canonical because it is elegant on paper.
It becomes canonical when it is contract-correct and faster in real training.

## 9. Optimization Roadmap and Blocked Proposals

This section records optimization proposals that are commonly suggested but
**blocked on the active MX150 (`sm_61`) machine** — either by hardware
capability or by canonical architecture rules in [AGENTS.md](../AGENTS.md).
It exists so future contributors do not re-propose, re-investigate, or
prematurely retry blocked work.

### 9.1 Blocked on Active Hardware (Future-Hardware Roadmap)

For implementation-ready specs (anchors, capability gates, validation gates,
rejection conditions, and recommended execution order) for every item in this
table, see [FUTURE_HARDWARE_ROADMAP.md](FUTURE_HARDWARE_ROADMAP.md).

| Proposal | Blocker | Hardware Unlock | Expected Gain |
|----------|---------|-----------------|---------------|
| `torch.compile` on PPO learner / sequence path | Triton requires `sm_70+`; MX150 is `sm_61` | RTX 2060 / Volta+ | Eliminates dispatcher gaps on PPO update (~1020 ms today) |
| `torch.compile` on RayViTEncoder + reward shaping | Already implemented and capability-gated; auto-disables on `sm_61` and falls back to eager / `jit.script` | RTX 2060 / Volta+ | Eliminates ~50% of remaining dispatcher gaps |
| `bfloat16` distance output from `cast_rays` kernel + tensor-core ingestion | MX150 has zero tensor cores (Pascal `sm_61`); native `bfloat16` requires `sm_80+` | RTX 3060 / Ampere+ | Halves output bandwidth; enables tensor-core MatMul in RayViT/heads |
| Hardware-fused `mamba-ssm` Triton kernels for the temporal core | Not available on Windows; requires `sm_70+` Triton | Linux + Volta+ | Collapses Mamba2's 55–60 dispatches/forward to a fused scan |
| Multi-group rollout overlap (`rollout_overlap_groups > 1`) | MX150 has 3 SMs — overlap causes ~47% throughput regression from SM starvation | Larger SM count (≥ 30 SMs typical) | Hides PPO backward behind environment stepping |

### 9.2 Permanently Blocked by Architecture (Not Hardware-Dependent)

| Proposal | Blocker | Authority |
|----------|---------|-----------|
| CUDA Graph capture of the rollout loop | Data-dependent control flow: `.nonzero()`, `.item()`, dynamic spawn-grid sizing, per-actor collision response, episode reset, scene rotation. None of this can be statically recorded. | [AGENTS.md §3.1.2](../AGENTS.md#3-performance-mandates), repeated in [SDFDAG_RUNTIME.md](SDFDAG_RUNTIME.md), [SIMULATION.md](SIMULATION.md) |
| Alternate trainer modes / shadow benchmark entrypoints | Forbidden by Controlled Selector Rule and Hot-Path Discipline — investigations must add attribution on the existing `navi-actor train` surface | [AGENTS.md §2.7](../AGENTS.md#27-temporal-core-standard-mar-2026), [AGENTS.md §3.3.1](../AGENTS.md#331-actor-hot-path-discipline) |
| Pinned-CPU / host-staged rollout slabs on the canonical path | Forbidden — canonical rollout storage must remain GPU-resident | [AGENTS.md §2.8](../AGENTS.md#28-sdfdag-integration-standard) |

### 9.3 Active-Hardware Levers (Real Next Jumps)

The dominant cost on this machine is the **PPO update (~1020 ms)** versus the
**environment step (~36.5 ms)** — a roughly 28× imbalance. Bandwidth and
graph-capture proposals do not address it. The actionable levers are:

1. **PPO update cadence tuning** — explicitly authorised by
   [AGENTS.md §2.7 Update-Frequency Rule](../AGENTS.md#27-temporal-core-standard-mar-2026):
   reduce optimizer update frequency on the canonical trainer surface. Must be
   benchmark-gated against learning quality.
2. **Temporal-core re-bake-off** — the Mar 2026 25K-step bake-off chose
   Mamba2 for learning quality (final reward_ema −0.88 vs GRU −1.48). On a
   throughput-constrained `sm_61` machine where backward dispatcher cost
   dominates, the trade-off space may have shifted. Re-validate via
   [scripts/run-temporal-bakeoff.ps1](../scripts/run-temporal-bakeoff.ps1)
   under current PPO cadence settings.
3. **PPO backward sub-attribution** — current profiling attributes ~1129 ms
   to the backward pass broadly. Per-section attribution (Mamba2 SSD scan vs.
   policy heads vs. critic vs. KL/clip math) is needed before further
   optimization can be targeted intelligently.

### 9.4 Why The Common HPC Playbook Doesn't Apply Here

The standard HPC optimization playbook (CUDA graphs → reduce dispatcher
overhead, half-precision → bandwidth, `torch.compile` → fusion) assumes a
modern data-center or workstation GPU. On `sm_61`:

- No tensor cores → `bfloat16` saves bandwidth but cannot accelerate compute
- No Triton support → `torch.compile` falls back to eager
- Per-actor episodic reset and collision recovery → graph capture is incompatible
  with the canonical Pipeline-A invariants documented in
  [SIM_TO_REAL_PARITY.md](SIM_TO_REAL_PARITY.md)

These same proposals **should be the canonical roadmap on `sm_70+` /
`sm_80+` hardware** and should be the first wave of work after a hardware
upgrade.

## 10. Related Docs

- [ACTOR.md](ACTOR.md)
- [TRAINING.md](TRAINING.md)
- [VERIFICATION.md](VERIFICATION.md)
- [RESOLUTION_BENCHMARKS.md](RESOLUTION_BENCHMARKS.md)
- [SDFDAG_RUNTIME.md](SDFDAG_RUNTIME.md)
- [SIM_TO_REAL_PARITY.md](SIM_TO_REAL_PARITY.md)
