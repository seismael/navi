# SIMULATION.md — Canonical Environment Runtime

## 1. Overview

The environment layer now exposes one active production runtime:

- compiled `.gmdag` assets
- CUDA sphere tracing through `projects/torch-sdf`
- `DistanceMatrix v2` preserved for external service and diagnostic boundaries
- coarse `environment.sdfdag.perf` telemetry
- staged corpus refresh that removes transient raw downloads after promotion

The environment remains the sole spatial truth layer for Navi.

## 2. Canonical Commands

```bash
# Environment service
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file ./artifacts/gmdag/corpus/apartment_1.gmdag

# Shortcut
uv run environment

# Prepare or refresh the full corpus
uv run navi-environment prepare-corpus --force-recompile

# Compile one explicit source scene
uv run navi-environment compile-gmdag --source ./data/scenes/hssd/102343992.glb --output ./artifacts/gmdag/corpus/hssd/102343992.gmdag --resolution 512

# Runtime preflight and throughput validation
uv run navi-environment check-sdfdag
uv run navi-environment check-sdfdag --json
uv run navi-environment check-sdfdag --gmdag-file ./artifacts/gmdag/corpus/apartment_1.gmdag
uv run navi-environment bench-sdfdag --gmdag-file ./artifacts/gmdag/corpus/apartment_1.gmdag --actors 4 --steps 200
uv run navi-environment bench-sdfdag --gmdag-file ./artifacts/gmdag/corpus/apartment_1.gmdag --actors 4 --steps 200 --json
uv run navi-environment bench-sdfdag --gmdag-file ./artifacts/gmdag/corpus/apartment_1.gmdag --actors 4 --steps 200 --repeats 5 --json
```

## 3. Runtime Architecture

`SimulatorBackend` remains the environment abstraction boundary, but only one
production implementation remains active:

```text
SimulatorBackend
  -> SdfDagBackend
     -> preloaded DAG tensor on CUDA
     -> reusable ray origin/direction buffers
     -> reusable output tensors
     -> optional publish-time materialization
```

Key runtime properties:

- one canonical compiled-scene execution path
- no CPU fallback in the compiled runtime
- batched actor stepping is mandatory
- the CUDA ray tracer stops at the configured environment horizon
- `DistanceMatrix` materialization is preserved for service mode and passive tooling
- the canonical trainer may consume equivalent CUDA tensors directly

## 4. Corpus Preparation

`prepare-corpus` discovers source scenes, compiles missing or stale `.gmdag`
assets, and writes live manifests for the compiled corpus.

Canonical refresh policy:

- default compile resolution is `512`
- compiled assets with mismatched stored resolution are rebuilt automatically
- `scripts/refresh-scene-corpus.ps1` stages downloads and promotes only after a successful rebuild
- transient source downloads are removed after successful promotion
- live `gmdag_manifest.json` entries must point at promoted compiled assets, not scratch paths
- promoted manifest metadata must stay in sync with the live corpus: `scene_count`, `gmdag_root`, `requested_resolution`, and `compiled_resolutions` are validated against the promoted assets

Canonical roots:

- transient source staging: `artifacts/tmp/corpus-refresh/downloads/`
- compiled corpus: `artifacts/gmdag/corpus/`

## 5. Observation And Reward Semantics

### 5.1 Observation Contract

The environment publishes the actor's required contract unchanged:

- `depth`
- `delta_depth`
- `semantic`
- `valid_mask`

Wire arrays remain shaped `(1, Az, El)`.
The tensor-native trainer seam may use the equivalent CUDA observation tensor
`[depth, semantic, valid]` with shape `(3, Az, El)` per actor.

### 5.2 Fixed-Horizon Policy

The canonical environment horizon is `EnvironmentConfig.max_distance`.

That same value now controls:

- CUDA ray termination
- depth normalization
- horizon saturation behavior
- validity semantics for rays that exceed the configured horizon

Dynamic per-step trace radius is not part of the canonical runtime.

### 5.3 Training Persistence Policy

Canonical training keeps actors in-scene and learning:

- collision remains non-terminal
- invalid motion is reverted and penalized
- hard truncation defaults to `2000` steps
- scene rotation defaults to `16` completed episodes per scene across the fleet

### 5.4 Shaping Policy

The environment reward seam now includes:

- positive clearance-delta reward when an actor increases free space near geometry
- starvation penalty when horizon-saturated observations dominate the sphere
- proximity penalty when very near valid hits dominate the sphere
- positive structure-band reward when stable mid-range geometry stays visible
- positive forward-structure reward when informative geometry remains in the forward sector
- inspection reward when turning or repositioning reveals more usable structure instead of less

These terms are derived from already-produced batched depth and validity data.
They do not require a second sensing pipeline.

## 6. External Data Boundary

The imported OmniSense documents were strongest on one point: external data
mess belongs outside the training hot loop.

Navi expresses that through the adapter boundary.

`DatasetAdapter` is the only valid place for:

- axis transpose
- coordinate normalization
- semantic remapping
- external sensor projection into the spherical contract

The first canonical raw-adapter surface is an explicit equirectangular fixture:

- raw `equirect_depth` and `equirect_semantic` grids arrive as `(El, Az)`
- adapters convert them to canonical `(1, Az, El)` arrays
- adapted arrays materialize through one shared `DistanceMatrix` bridge so dataset and runtime publication paths preserve the same public contract
- fixture QA may materialize single frames or ordered frame sequences through `navi_environment.integration.adapt_fixture_frame(...)` and `navi_environment.integration.adapt_fixture_sequence(...)` without introducing a second runtime backend
- Habitat camera fixtures currently use the explicit `habitat_camera_transform_spec()` preset: right-handed, `+Y` up, forward `-Z`, which matches Navi's current canonical local frame and therefore materializes as an identity transform instead of an implied one
- per-dataset rigid transforms must be carried as explicit `4x4` matrices, not prose

This remains a boundary concern, not a reason to reopen alternate canonical
runtime paths.

## 7. Service And Auditor Surfaces

### 7.1 Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` and `step_result_v2`

### 7.2 Passive Observer Rules

- training-time dashboard usage must remain actor-stream-first and passive
- environment control sockets are for service mode and explicit diagnostics, not canonical training dependency
- frame dropping is acceptable if it preserves rollout throughput

### 7.3 Dataset QA Direction

The imported dashboard proposal for a pinhole dataset auditor is useful as a
diagnostic idea: validate compiled scenes through the same mathematical runtime
used for training instead of re-exporting geometry for browser rendering.

This is a passive validation surface, not a new runtime dependency.

## 8. Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

Important current checks include:

- fixed-horizon clamp and validity contract tests
- live compiled-corpus validation
- live `SdfDagBackend` step validation against real `.gmdag` assets
- `check-sdfdag --json` machine-readable runtime and corpus validation summaries for promoted-corpus preflight capture
- `check-sdfdag` corpus validation against the promoted manifest, live promoted assets, and canonical compile resolution when run without `--gmdag-file`
- `bench-sdfdag --json` structured summaries for repeatable environment benchmark capture and live promoted-corpus benchmark smoke proof
| `max_distance` | `float` | 30.0 | Maximum observation distance (metres) |
| `habitat_scene` | `str` | `""` | Path to Habitat `.glb` scene (Habitat only) |
| `habitat_dataset_config` | `str` | `""` | Path to PointNav episode JSON (Habitat only) |
| `seed` | `int` | 42 | Random seed for deterministic reset selection |
