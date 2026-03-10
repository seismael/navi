# SIMULATION.md - Canonical Environment Runtime

## 1. Executive Summary

The environment layer is the spatial truth layer of Navi. In the current repo it
has one active production runtime:

- compiled `.gmdag` assets under `artifacts/gmdag/corpus/`
- CUDA sphere tracing through `projects/torch-sdf`
- one batched `SdfDagBackend`
- contract-preserving `DistanceMatrix` publication for service and diagnostics
- optional tensor-native seams for the canonical trainer

The environment does not own the sacred actor. It owns the world query,
kinematic update, reward seam, and contract-preserving adaptation layer.

## 2. Canonical Commands

```bash
# Environment service
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file ./artifacts/gmdag/corpus/replicacad/frl_apartment_stage.gmdag

# Shortcut
uv run environment

# Prepare or refresh the full corpus
uv run navi-environment prepare-corpus --force-recompile

# Compile one explicit source scene
uv run navi-environment compile-gmdag --source ./data/scenes/replicacad/frl_apartment_stage.glb --output ./artifacts/gmdag/corpus/replicacad/frl_apartment_stage.gmdag --resolution 512

# Runtime preflight and throughput validation
uv run navi-environment check-sdfdag --gmdag-file ./artifacts/gmdag/corpus/replicacad/frl_apartment_stage.gmdag
uv run navi-environment bench-sdfdag --gmdag-file ./artifacts/gmdag/corpus/replicacad/frl_apartment_stage.gmdag --actors 4 --steps 200
```

## 3. Runtime Architecture

`SimulatorBackend` remains the abstract environment boundary, but only one
production implementation is active:

```text
SimulatorBackend
  -> SdfDagBackend
     -> loaded DAG asset metadata
     -> contiguous DAG tensor on CUDA
     -> reusable ray origin and direction buffers
     -> reusable distance and semantic output buffers
     -> optional materialization for publish surfaces
```

Key runtime properties:

- one canonical compiled-scene execution path
- CUDA-only execution on the production path
- batched actor stepping is mandatory
- the CUDA ray tracer stops at the configured environment horizon
- the actor may consume equivalent CUDA tensors directly on the canonical path
- service mode and diagnostics may still materialize `DistanceMatrix` objects

## 4. Corpus Preparation And Promotion

### 4.1 Source Discovery

`prepare-corpus` discovers scenes from:

- an explicit `--scene` override
- an explicit manifest
- or canonical source discovery roots

When the user does not narrow the corpus, full discovered corpus coverage is the
default production training dataset.

### 4.2 Compilation Policy

Canonical compile policy is:

- default resolution `512`
- overwrite-first refresh when explicitly requested
- automatic replacement of compiled assets with mismatched stored resolution
- promotion only after a successful staged rebuild

### 4.3 Live Artifact Policy

The persistent runtime depends on compiled assets, not retained raw downloads.

After a successful corpus refresh:

- transient source downloads may be removed
- live manifests must point at promoted `.gmdag` assets
- scratch download paths must not remain in promoted metadata

Canonical roots:

- transient source staging: `artifacts/tmp/corpus-refresh/downloads/`
- compiled corpus: `artifacts/gmdag/corpus/`

## 5. Step Lifecycle

A single batched environment step on the canonical runtime does the following:

1. receive actions or action-equivalent tensors for `B` actors
2. integrate body-frame motion through `MjxEnvironment`
3. update actor origins and world-frame directions
4. invoke `torch_sdf.cast_rays()` on preallocated CUDA buffers
5. reshape batched outputs into `(Az, El)` depth, semantic, and validity grids
6. compute environment-side reward terms
7. either:
   - return tensor-native observation batches to the trainer, or
   - materialize `DistanceMatrix` objects for publication or service replies

The important architectural point is that steps 4 through 6 already have the
information needed for most production training logic. Rebuilding Python objects
is therefore optional on the fast path, not required.

## 6. Observation Contract And Geometry Conventions

### 6.1 Public Observation Contract

The environment publishes the actor's required fields unchanged:

- `depth`
- `delta_depth`
- `semantic`
- `valid_mask`

Wire arrays remain shaped `(1, Az, El)` per environment.

### 6.2 Tensor-Native Trainer Seam

The canonical trainer may consume the equivalent CUDA tensor with channels:

- channel `0`: normalized depth
- channel `1`: semantic ids cast to `float32`
- channel `2`: validity mask cast to `float32`

Per-actor shape is `(3, Az, El)`. Batched trainer shape is `(B, 3, Az, El)`.

### 6.3 Spherical Convention

The canonical spherical convention is:

- `matrix_shape == (azimuth_bins, elevation_bins)`
- forward ray is azimuth bin `0`
- forward direction is local `-Z`
- dashboard forward-FOV extraction must roll the azimuth seam before cropping

## 7. Fixed-Horizon Policy

The canonical environment horizon is `EnvironmentConfig.max_distance`.
That same value controls:

- CUDA ray termination
- depth normalization
- horizon-saturation semantics
- invalidity of rays that exceed the configured horizon

Dynamic per-step trace radius is not part of the canonical runtime.

## 8. Reward And Episode Semantics

### 8.1 Persistence-First Collision Handling

Canonical training is persistence-first:

- collision is non-terminal
- invalid motion is reverted
- a penalty is applied
- the episode continues

### 8.2 Scene Residency

Canonical scene rotation is throughput-aware:

- hard truncation defaults to `2000` steps
- scene-pool rotation defaults to `16` completed episodes per scene across the fleet
- actors stay on a scene long enough to recover, explore, and exploit local
  geometry rather than rotating after trivial failure events

### 8.3 Environment-Side Shaping Terms

The environment reward seam includes:

- obstacle-clearance reward for increasing free space while near geometry
- starvation penalty when horizon-saturated observations dominate the sphere
- proximity penalty when very near valid hits dominate the sphere

These terms are derived from already-produced depth and validity tensors.
No second observation pipeline is required.

## 9. External Data And Sim-to-Real Boundary

The imported OmniSense material is adopted in a narrow, production-safe form.

`DatasetAdapter` remains the only valid place for:

- axis transpose
- coordinate normalization
- semantic remapping
- external sensor projection into the spherical contract

This is a boundary rule, not a license to reopen multiple production runtime
paths.

### 9.1 Domain Randomization Direction

Noise injection and hardware-robustness transformations remain valid design
ideas, but they belong after mathematical world query and before training or
inference consumption. They should be treated as optional environment-side
post-processing, not as new compiler or actor responsibilities.

### 9.2 Sensor Bridge Direction

Real-world hardware compatibility remains conceptually simple:

- point clouds or stitched depth sensors must be projected into the canonical
  spherical contract
- the actor should not need to know whether those inputs came from simulation or
  real hardware

That remains a design direction, not a second canonical runtime.

## 10. Service Surfaces

### 10.1 Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` and `step_result_v2`

### 10.2 Service-Mode Validity

Service mode remains valid for:

- integration testing
- manual stepping
- dashboard teleoperation
- recorder and rewinder workflows

Service mode is not the canonical throughput path, but it remains an important
operational and validation surface.

## 11. Dataset QA Direction

The imported dashboard proposal for a dataset-auditor pinhole camera is kept as
an important diagnostic direction:

- render compiled scenes through the same mathematical runtime used for training
- avoid geometry export or browser-only surrogate visualizations
- prefer proof through the real runtime over convenience rendering

This is a passive validation surface and should remain optional.

## 12. Validation

Core validation commands:

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

Important current checks include:

- fixed-horizon clamp and validity contract tests
- live compiled-corpus validation
- live `SdfDagBackend` step validation against real `.gmdag` assets
- runtime readiness checks via `check-sdfdag`
- environment throughput attribution via `bench-sdfdag`
