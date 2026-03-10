# SDFDAG_RUNTIME.md - Low-Level CUDA And DAG Runtime Notes

## 1. Purpose

This document captures the low-level implementation truths that matter for
performance work on the canonical `.gmdag` plus `torch-sdf` path.

It is intentionally narrower than `docs/ARCHITECTURE.md` and more concrete than
`docs/PERFORMANCE.md`.

## 2. Current Verified Runtime Shape

The current production path is:

1. `projects/voxel-dag` produces compiled `.gmdag` assets
2. `projects/environment` validates and loads asset metadata plus contiguous node payloads
3. `projects/torch-sdf` receives DAG memory and batched ray tensors on CUDA
4. `torch_sdf.cast_rays()` fills preallocated distance and semantic outputs
5. `SdfDagBackend` adapts those outputs into either:
   - tensor-native trainer observations, or
   - materialized `DistanceMatrix` objects for service and diagnostics

## 3. Binary Artifact Contract

### 3.1 Current Header Layout

The environment integration code currently parses a 32-byte header with:

| Offset | Type | Field |
| --- | --- | --- |
| `0x00` | `char[4]` | magic `GDAG` |
| `0x04` | `uint32` | format version |
| `0x08` | `uint32` | grid resolution |
| `0x0C` | `float32` | `bbox_min.x` |
| `0x10` | `float32` | `bbox_min.y` |
| `0x14` | `float32` | `bbox_min.z` |
| `0x18` | `float32` | voxel size |
| `0x1C` | `uint32` | node count |

The payload that follows is a contiguous `uint64` node array.

### 3.2 Current Load Semantics

`load_gmdag_asset()` currently validates:

- header size
- `GDAG` magic
- node payload length consistency
- derived `bbox_max` from `bbox_min + resolution * voxel_size`

Those checks make the file format a real runtime contract rather than a vague
compiler-side convention.

## 4. Current Node Layout Observations

The current CUDA kernel already assumes a compact `uint64` node layout.
Observed implementation facts include:

- the high bit acts as a node-type discriminator
- internal nodes carry an 8-bit child mask and a child-base pointer
- leaf nodes carry packed distance and semantic payloads
- DAG memory is traversed as one contiguous `uint64` array
- `__restrict__` pointers are used for DAG memory and ray buffers

These details matter because they explain the current cache-friendly traversal
shape and why the runtime is built around contiguous buffers instead of rich
object graphs.

## 5. CUDA Boundary Contract

### 5.1 Current Input Tensors

| Tensor | Shape | Dtype | Device | Requirement |
| --- | --- | --- | --- | --- |
| `origins` | `[B, R, 3]` | `float32` | CUDA | contiguous |
| `dirs` | `[B, R, 3]` | `float32` | CUDA | contiguous |
| `out_distances` | `[B, R]` | `float32` | CUDA | preallocated |
| `out_semantics` | `[B, R]` | `int32` | CUDA | preallocated |

### 5.2 Binding Enforcement

The current binding enforces before kernel launch:

- CUDA placement for input and output tensors
- contiguity on ray tensors
- rank-3 input with trailing dimension `3`

The binding also releases the Python GIL during kernel execution.

This is one of the most important imported ideas that is now fully documented in
repo-local terms.

## 6. Kernel Execution Sequence

At a high level the current kernel path is:

1. flatten batched rays into `num_rays = B * R`
2. map one thread to one ray
3. read origin and direction triples
4. query the stackless DAG for local distance and semantic payload
5. advance along the ray by sampled distance
6. terminate on hit or horizon exhaustion
7. write the resulting distance and semantic id to the output tensors

The important runtime consequence is that output buffers can be reused across
steps and only need reshaping and adaptation afterward.

## 7. Environment Adaptation Boundary

The environment currently reshapes runtime outputs from `[B, R]` into
`(B, Az, El)` depth, semantic, and validity tensors. It then chooses one of two
surfaces:

- tensor-native trainer observation batches
- materialized `DistanceMatrix` publication surfaces

This is the critical seam where performance work now lives. The kernel can be
fast and still lose end-to-end throughput if this boundary bounces data to host
unnecessarily.

## 8. Stable Expectations vs. Benchmark-Gated Ideas

### 8.1 Stable Expectations

The following are current architectural truths:

- CUDA-only canonical runtime
- batched actor-ray execution
- one configured environment horizon shared by tracing and normalization
- reusable buffers preferred over per-step allocation
- selective publish-time materialization instead of unconditional wire-object rebuilding

### 8.2 Benchmark-Gated Ideas

The following remain candidates, not locked guarantees:

- exact future leaf-distance precision policy
- truncation metadata schemes beyond the fixed-horizon runtime contract
- alternate storage layouts such as Morton or Z-order variants
- deeper cache-tuned node layouts that change the current on-disk contract

## 9. Hot-Path Rules

Performance work on this stack should assume:

- reusable buffers are preferred over per-step allocation
- host extraction should be batched and coarse when unavoidable
- materialization for dashboards and telemetry should be selective
- any optimization that helps only the kernel but regresses trainer dataflow is
  incomplete

## 10. Related Docs

- `docs/COMPILER.md`
- `docs/ARCHITECTURE.md`
- `docs/SIMULATION.md`
- `docs/DATAFLOW.md`
- `docs/PERFORMANCE.md`
- `docs/VERIFICATION.md`
