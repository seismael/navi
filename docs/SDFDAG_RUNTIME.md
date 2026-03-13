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

## 9. Hot-Path Rules

Performance work on this stack should assume:

- reusable buffers are preferred over per-step allocation
- canonical action-tensor stepping must keep command rows on device rather than bouncing CUDA actions through NumPy
- reset handling should batch state reinitialization and initial observation seeding for all selected actors instead of walking the CUDA reset mask actor-by-actor
- collision rollback on the canonical tensor path should restore the previous safe pose without launching a second collided-subset ray cast in the same tick
- materialization for dashboards and telemetry should be selective
- host extraction should be batched and coarse when unavoidable
- any optimization that helps only the kernel but regresses trainer dataflow is incomplete

## 10. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/PERFORMANCE.md`
- `docs/SIMULATION.md`
- `docs/VERIFICATION.md`
