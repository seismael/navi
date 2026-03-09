# COMPARISON.md — Runtime Path Comparison

This document compares the runtime paths that still matter inside Navi:

- `VoxelBackend`
- `MeshSceneBackend`
- `HabitatBackend`
- the integrated `SdfDagBackend` path built around `projects/voxel-dag` and `projects/torch-sdf`

The goal of this comparison is practical: define the role of each path now that
the SDF/DAG runtime is the canonical performance-sensitive training surface.

---

## 1. Backend Comparison Matrix

| Path | Current Role | Strengths | Weaknesses | Canonical Status |
| --- | --- | --- | --- | --- |
| `VoxelBackend` | Procedural regression reference | Simple, deterministic, sparse local worlds | Not the canonical compiled-scene runtime | Diagnostic only |
| `MeshSceneBackend` | CPU regression reference | Batched raycasting, no Habitat dependency | CPU-side mesh traversal limits ultimate scaling | Diagnostic only |
| `HabitatBackend` | Sensor/adapter validation | Rich external scenes and semantics | External simulator overhead | Diagnostic only |
| `SdfDagBackend` | Canonical training runtime | `.gmdag` compression, CUDA sphere tracing, reusable GPU buffers, coarse perf telemetry, direct benchmark CLI | Requires CUDA/toolchain readiness and compiled assets | Canonical |

---

## 2. Why SDF/DAG Is The Target

The imported monorepo domains provide the intended long-term advantages:

- `projects/voxel-dag` reduces large scenes into compressed `.gmdag` caches.
- `projects/torch-sdf` exposes a CUDA-only batched ray API through
  `torch_sdf.cast_rays()`.
- This allows Navi to pursue GPU-resident world queries without changing the
  actor's sacred input contract.

---

## 3. Decision Rule

The canonical `SdfDagBackend` remains accepted only while all of the following stay true:

1. `.gmdag` compilation is owned by Navi workflows.
2. `torch_sdf.cast_rays()` is validated on the active CUDA stack.
3. `DistanceMatrix` compatibility is preserved without actor changes.
4. End-to-end training throughput beats or materially advances the current
   mesh and voxel baselines.

---

## 4. Actor Boundary Reminder

This comparison does not reopen actor architecture selection. The actor remains
RayViT + Mamba. The migration contest is exclusively about the environment
runtime that feeds the same sacred observation contract.
