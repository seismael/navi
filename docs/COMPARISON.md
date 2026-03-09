# COMPARISON.md — Canonical Runtime Summary

The repository now keeps one active environment runtime only:

| Runtime | Role | Status |
|---------|------|--------|
| `SdfDagBackend` | GPU-resident batched ray execution over compiled `.gmdag` scenes | Canonical |

Supporting domains:

- `projects/voxel-dag` compiles source scenes into `.gmdag` assets
- `projects/torch-sdf` executes CUDA sphere tracing
- `projects/environment` preserves the actor-facing `DistanceMatrix` contract

Historical mesh, voxel, habitat, and sparse-Zarr paths have been removed from the active repository runtime.
