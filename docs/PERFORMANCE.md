# PERFORMANCE.md — Throughput And Runtime Gates

This document defines the performance-first acceptance criteria for Navi.
Correctness is mandatory, but performance decides which runtime path becomes
canonical for training.

---

## 1. Performance Priority Order

1. **Rollout throughput:** actor + environment must keep training moving.
2. **No-stall execution:** optimizer work must not starve simulation.
3. **GPU residency:** canonical high-performance paths keep world query buffers
   and reusable outputs on CUDA.
4. **Contract preservation:** performance upgrades must still emit canonical
   `DistanceMatrix` tensors for the sacred actor.

---

## 2. Canonical Runtime Paths

### 2.1 Canonical Production Path

The repository's canonical production training path is:

1. `projects/voxel-dag` compiles source meshes to `.gmdag`.
2. `projects/torch-sdf` executes batched CUDA sphere tracing against the DAG.
3. `projects/environment` adapts `[B, Rays]` outputs back into canonical
   `(1, Az, El)` `DistanceMatrix` tensors.
4. `projects/actor` remains unchanged and consumes the same observation contract.

For canonical training internals specifically, this contract preservation rule
does not require the rollout hot path to rebuild Python `DistanceMatrix`
objects on every step. The contract must remain externally true, but the
fastest path is allowed to keep equivalent observation tensors on CUDA and only
materialize wire objects for coarse diagnostics.

### 2.3 Canonical Throughput Direction

The repository keeps one canonical training runtime only: direct in-process
backend stepping over `sdfdag`. If the environment backend is already batched
and GPU-backed, the next required optimization is removal of per-step transport
and host staging, not continued support for alternate training architectures.

The confirmed remaining bottlenecks are now actor/runtime dataflow:

1. CUDA ray outputs are converted to CPU `numpy` arrays and Python
  `DistanceMatrix` objects before training consumes them.
2. The trainer converts those objects back into CUDA tensors every step.
3. Actions, embeddings, values, and intrinsic rewards are still extracted to
  CPU inside the rollout loop.
4. The remaining actor-side hot-path work is dominated by coordination and
  coarse telemetry costs now that reward shaping, episodic-memory query/add,
  and rollout-buffer writes are tensor-native or batched.

The current implementation priority is attribution-first work on this same
canonical surface:

1. add explicit diagnostic toggles for telemetry, episodic memory, and reward
  shaping on `navi-actor train`
2. extend perf telemetry with host-extraction and publication cost visibility
3. remove avoidable device -> host -> device bounces that still remain inside
  the actor rollout tick

The latest full attribution matrix on
`artifacts/benchmarks/attribution-matrix/20260309-095734/summary.csv` showed:

1. episodic memory and reward shaping are the only diagnostic toggles with a
  consistently material effect on steady-state rollout SPS
2. observation stream and training/perf telemetry have comparatively small
  rollout impact at the current coarse cadence
3. PPO rollout-boundary optimization time remains the dominant wall-clock cost

The next required implementation pass is therefore:

1. optimize the canonical PPO learner/update path before further environment
  micro-optimisation
2. remove redundant optimizer-side device copies and allocator churn on CUDA
3. harden perf-only telemetry publication so any attribution toggle
  combination fails safely instead of crashing the trainer

### 2.2 Diagnostic Reference Paths

- `VoxelBackend`: procedural regression and correctness reference.
- `MeshSceneBackend`: CPU-side regression comparison for compiled-scene behavior.
- `HabitatBackend`: external adapter and sensor validation path.

---

## 3. Non-Negotiable Runtime Rules

- Canonical high-performance runtime must not silently fall back to CPU.
- Batched stepping is mandatory for the SDF/DAG path.
- Per-step allocation churn in the hot path is forbidden when reusable buffers
  are practical.
- Observation adaptation must be vectorized and must preserve actor contracts.
- Benchmark claims are valid only when measured end-to-end, not just in a microkernel.
- Actor-side rollout code must be profiled and optimized as soon as environment
  stepping stops dominating end-to-end SPS.
- Episodic-memory capacity enforcement must remain amortized; full index rebuilds
  on each post-capacity insert are not acceptable on the canonical training path.
- Per-step CUDA-to-host synchronization in the actor hot loop must be kept to the
  minimum needed for wire transport and logging.
- Canonical training must not pay a GPU -> CPU -> GPU bounce for observations
  once the environment backend is already producing batched CUDA results.
- Canonical training must not create one Python `Action` object per actor per
  step in the rollout hot path once actions already exist as batched CUDA
  tensors.
- Canonical throughput work must not add new ZMQ or serialization hops inside the
  rollout loop when an in-process backend step is available.
- Attribution tooling must stay on the canonical trainer surface and be framed as
  diagnostic controls, not as alternate training modes.
- Attribution controls must be benchmark-safe in every supported combination;
  a perf-only telemetry configuration is valid diagnostics, not an exceptional
  mode.
- Auditor/dashboard observability during canonical training must remain passive:
  actor-stream subscription is allowed, but environment control sockets and
  environment-stream dependencies must not be required on the training path.

---

## 4. Benchmark Gates

### 4.1 Fleet Throughput

- Standard fleet: `4` actors.
- Acceptance floor: `>= 60 SPS` for canonical training runtime.
- Any major SDF/DAG runtime change should still be compared against the current
  mesh/voxel references on equivalent scenes and resolution profiles.

### 4.2 Latency Budgets

- Environment latency target for 4 actors remains bounded by the current
  Ghost-Matrix performance mandates.
- SDF/DAG runtime work must lower or at minimum not regress per-step environment
  cost relative to the mesh baseline.
- Startup and preflight latency are secondary to training throughput, but
  canonical runtime must fail fast.

### 4.3 Ongoing Proof Requirements For Canonical SDF/DAG

Now that `sdfdag` is the canonical training surface, changes to that path must
continue to demonstrate all of the following:

1. `.gmdag` compilation is reproducible from Navi workflows.
2. `torch_sdf.cast_rays()` executes correctly on CUDA with batched actor rays.
3. The backend preserves `DistanceMatrix` semantics and shapes.
4. End-to-end training throughput improves or materially advances the repo toward
   the `>= 60 SPS` fleet target.
5. The canonical training hot path reduces `.cpu()`, `.numpy()`, scalar
  extraction, and Python object assembly relative to the previous baseline.

---

## 5. Current Baselines

These numbers remain useful as current reference points, not as final goals.

| Profile | Resolution | Steps | Mean SPS | Mean Zero-Wait Ratio | Soft Warnings |
| --- | --- | --- | --- | --- | --- |
| Baseline long run | `256x48` | 29,600 | `17.32` | `19.16%` | `269` |
| Focused pass | `128x24` | 5,000 | `20.62` | `0.00%` | `2` |

These values establish the current bar that SDF/DAG work must beat or clearly
surpass in scaling potential.

## 5.1 Bottleneck Interpretation Rule

End-to-end SPS must be interpreted using both environment and actor telemetry.
If environment batch-step time remains low while actor `trans` time grows, the
active bottleneck has moved above the simulation layer and fixes must target the
actor runtime before further environment micro-optimisation.

During the 400-SPS gap investigation, actor perf interpretation should also use:

- `host_extract_ms`: batched device -> host extraction cost needed for coarse
  accounting and telemetry
- `telemetry_publish_ms`: cost of publishing optional observation/training
  telemetry on the canonical path
- explicit ablation runs where episodic memory, reward shaping, and telemetry
  emission are disabled one class at a time

The current interpretation priority after the March 2026 full matrix is:

1. treat `ppo_update_ms` as the primary wall-clock bottleneck once `env_ms`
  stays in the low-millisecond range
2. treat `host_extract_ms` as secondary cleanup work unless it grows into the
  same order of magnitude as reward shaping or transition assembly
3. treat telemetry toggles as robustness and observability concerns first,
  because their measured steady-state SPS effect is small at the current cadence

---

## 6. Performance-Focused Integration Sequence

1. Add monorepo-native `.gmdag` compiler orchestration.
2. Add CUDA preflight and `torch-sdf` runtime validation.
3. Implement batched `sdfdag` backend in `projects/environment`.
4. Emit coarse `environment.sdfdag.perf` telemetry from the canonical batched path.
5. Benchmark against mesh and voxel via `uv run navi-environment bench-sdfdag ...`.
6. Keep the canonical actor training surface on direct in-process `sdfdag` stepping.
7. Keep canonical launch defaults on `sdfdag`; do not reopen legacy defaults.
8. Replace observation and action host staging with tensor-native runtime and
  trainer integration.
9. Batch reward shaping and rollout-buffer writes before revisiting smaller
  Python-side optimisations.
10. After tensor-native action stepping lands, canonical actor work moves to
  rollout-buffer indexed writes, tensor-native episodic memory, and then
  optimizer-coordination reassessment rather than observation or Action
  transport.

---

## 7. Canonical Decision Rule

No runtime path becomes canonical because it is mathematically elegant or easy to
explain. It becomes canonical only when it is contract-correct and measurably
faster in end-to-end training.
