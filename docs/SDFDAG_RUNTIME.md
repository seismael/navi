# SDFDAG_RUNTIME.md — Low-Level CUDA And DAG Runtime Notes

## 1. Purpose

This document captures the low-level implementation truths that matter for
performance work on the canonical `.gmdag` plus `torch-sdf` path.

It is intentionally narrower than `docs/ARCHITECTURE.md` and more concrete than
`docs/PERFORMANCE.md`.

## 2. Current Verified Runtime Shape

The current production path is:

1. `projects/voxel-dag` produces compiled `.gmdag` assets
2. `projects/torch-sdf` loads the DAG into CUDA-visible memory
3. `projects/environment` prepares batched ray origins and directions
4. `torch_sdf.cast_rays()` fills preallocated distance and semantic outputs
5. `SdfDagBackend` adapts those outputs into either:
   - tensor-native trainer observations, or
   - materialized `DistanceMatrix` objects for service and diagnostics

## 3. CUDA Boundary Contract

The current binding enforces the following before launching the kernel:

- `origins` must be CUDA and contiguous
- `dirs` must be CUDA and contiguous
- output tensors must be CUDA
- `origins` must have rank `3` with trailing dimension `3`

The binding also releases the Python GIL during kernel execution.

This matters because the boundary is a systems seam, not a convenience wrapper.

## 4. Current Tensor Shapes

### 4.1 Ray Tensors

| Tensor | Shape | Dtype | Device |
| --- | --- | --- | --- |
| `origins` | `[B, R, 3]` | `float32` | CUDA |
| `dirs` | `[B, R, 3]` | `float32` | CUDA |
| `out_distances` | `[B, R]` | `float32` | CUDA |
| `out_semantics` | `[B, R]` | `int32` | CUDA |

### 4.2 Trainer Observation Tensor

Current canonical trainer observation shape:

- `(B, 3, Az, El)` on CUDA
- channel `0`: normalized depth
- channel `1`: semantic ids cast to `float32`
- channel `2`: valid mask cast to `float32`

The external `DistanceMatrix` contract remains the same; this tensor seam exists
so the hot path can stay on device.

## 5. Current DAG Layout Notes

The current CUDA kernel already assumes a compact `uint64` node layout.
Observed implementation facts in the current code include:

- one contiguous `uint64` DAG memory block
- a type bit in the high bit position
- internal nodes carrying an 8-bit child mask and child-base pointer
- leaf nodes carrying packed distance and semantic payloads
- `__restrict__` pointers used in the kernel for DAG memory and ray buffers

These low-level choices are useful to document because they explain why the
runtime is shaped around contiguous buffers and alias-safe traversal code.

Important precision note:

- `__restrict__` is an aliasing promise to the compiler, not a guarantee that
   data will be routed through a specific cache tier
- cache residency and read-only behavior depend on the full kernel, the target
   architecture, and compiler code generation rather than on `__restrict__`
   alone

## 6. Execution Boundaries

The canonical runtime is bounded, not literally constant-time.

Execution cost depends on:

- DAG traversal depth
- configured ray horizon
- hit epsilon
- maximum march iteration count `MAX_STEPS`
- scene geometry and ray distribution

The stable architectural guarantee is therefore bounded batched execution, not
literal constant-time cost per ray.

## 6.1 Runtime Ceiling vs. Trainer Ceiling

The current repository now documents these as different limits.

- the low-level runtime ceiling is measured with `bench-sdfdag`
- the full-trainer ceiling is measured with the canonical actor training surface

March 2026 resolution sweeps on the active MX150 machine showed that the
runtime remained benchmark-viable at profiles above the current full-trainer
limit. The trainer then failed later at `768x144` inside actor-side transformer
self-attention during PPO update.

That distinction matters for low-level CUDA work:

- a trainer OOM at high resolution is not evidence that `torch_sdf.cast_rays()`
   failed
- low-level runtime changes must still be benchmarked on `bench-sdfdag` even
   when the actor currently hits the first wall in end-to-end runs

## 7. Boundary Semantics

The runtime and the environment boundary have distinct responsibilities.

Runtime-level expectations:

- hit, miss, and maximum-iteration termination must be explicit
- inside-solid behavior must be deterministic and documented
- the kernel may expose negative or near-zero distances when the underlying
   field supports that interpretation

Environment-level expectations:

- the published `DistanceMatrix` contract defines how horizon saturation,
   invalidity, and observation normalization are surfaced to the actor
- out-of-domain handling for public observations is an environment contract,
   not a promise of globally correct analytic SDF continuation beyond the
   compiled asset domain

### 7.1 Runtime Correctness Targets

The runtime validation target is not “kernel launches without crashing.”
Canonical tests must continue to expand around these behaviors:

- direct hits, grazing hits, and bounded misses against small analytic fixtures
- exact or tolerance-bounded behavior for `max_steps`, hit epsilon, and outside-domain miss semantics
- rejection of non-finite bounds and malformed tensor inputs before kernel launch
- repeated batched execution with preallocated tensors to guard against silent shape or buffer-regression bugs
- shared oracle-view fixtures that pin known spherical depth profiles to named poses so environment, actor, and auditor layers consume the same expected ray truth

Where a test can use an analytic reference or an independent small-scene evaluator, that oracle should be preferred over matching one Navi implementation against another.

## 8. What Is Stable vs. What Is Not

Stable architectural expectations:

- CUDA-only canonical runtime
- batched actor-ray execution
- one configured environment horizon shared by tracing and normalization
- optional publish-time materialization instead of unconditional wire-object rebuilding

Not yet frozen as public format guarantees:

- exact future leaf payload precision policy
- alternative storage layouts such as Morton or Z-order variants
- any truncation-metadata scheme beyond the currently accepted fixed-horizon runtime contract

Those remain benchmark-gated experiments.

## 8.1 Runtime Load Validation Boundary

Canonical runtime startup is not the same thing as corpus qualification.

The current policy is:

- `load_gmdag_asset()` always enforces header correctness, finite bounds, payload length, and trailing-byte integrity
- full DAG graph traversal and pointer-layout validation remain mandatory on explicit integrity surfaces such as `check-sdfdag`, promoted-corpus validation, and dedicated loader tests
- ordinary `SdfDagBackend` scene loads may skip that repeated deep traversal for already-qualified canonical assets so environment startup and benchmarks measure runtime behavior instead of re-qualifying the binary format on every load

## 9. Hot-Path Rules

Performance work on this stack should assume:

- reusable buffers are preferred over per-step allocation
- `torch_sdf.cast_rays()` is the CUDA boundary; tensor-only direction prep, output normalization, kinematics, and reward helper graphs around that boundary are valid `torch.compile` targets on the canonical path
- canonical action-tensor stepping must keep command rows on device rather than bouncing CUDA actions through NumPy
- canonical trainer-facing tensor steps should return tensor bookkeeping as the primary result surface and treat Python `StepResult` reconstruction as an opt-in diagnostic/public seam rather than unconditional per-step work
- reset handling should batch state reinitialization and initial observation seeding for all selected actors instead of walking the CUDA reset mask actor-by-actor
- canonical reset pose selection may spend extra one-time scene-load work on low-resolution spherical spawn probes so spawn positions and initial yaws favor navigable, structure-visible interior views instead of pure clearance maxima
- collision rollback on the canonical tensor path should restore the previous safe pose without launching a second collided-subset ray cast in the same tick
- materialization for dashboards and telemetry should be selective
- host extraction should be batched and coarse when unavoidable
- any optimization that helps only the kernel but regresses trainer dataflow is incomplete

Important limit:

- `torch.compile` does not optimize the internal DAG traversal inside `projects/torch-sdf/cpp_src/kernel.cu`; deeper traversal caching remains a separate CUDA/runtime project with its own correctness and benchmark gates
- the compiled tensor-helper path also depends on the active GPU/compiler stack; on unsupported devices such as the current `sm_61` MX150 surface, Navi now warns and keeps the eager tensor helper path active

Current traversal reuse notes:

- the CUDA kernel now checks a cached resolved leaf cell before any DAG descent so repeated march samples that stay inside one deep leaf can return the same packed distance and semantic payload immediately
- when the next sample leaves that leaf cell, the kernel still falls back to the existing fixed-depth internal-prefix cache before resuming full descent from the root
- these reuse layers are runtime-only hot-path optimizations; they do not change the `.gmdag` binary contract or Python binding surface

## 10. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/PERFORMANCE.md`
- `docs/SIMULATION.md`
- `docs/VERIFICATION.md`
