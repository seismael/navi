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
- observer requirements do not change the environment's tensor math, observation normalization, or step semantics
- canonical hot-path `cast_rays()` calls use `skip_direction_validation=True` because yaw-rotated unit vectors are mathematically guaranteed normalized, eliminating four GPU→CPU pipeline drains per call; probe and inspection calls retain validation
- CUDA graph capture of the full step path is not feasible due to data-dependent control flow

## 3.1 Runtime-Only Scaling Interpretation

`bench-sdfdag` is the environment-layer throughput surface, not the full-trainer
throughput surface.

Use it to answer questions like:

- does the CUDA runtime still scale acceptably when ray count increases?
- did a compiler or kernel change regress stepping throughput?
- does a higher observation profile remain viable before the actor is involved?

Do not use it to claim end-to-end trainer viability at the same resolution.
The active March 2026 benchmark work showed that the environment runtime
remained usable above the current full-trainer ceiling on the MX150 machine.
That means a trainer failure at high resolution is not automatically an
environment regression.

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

This contract is viewer-invariant. Dashboard FOV choice, half-sphere extraction,
palette, or other observer presentation preferences must be applied after
publication and must not alter the environment contract or tensor stepping path.

Real-scene interpretation note:

- not every approved dataset scene has a closed ceiling shell
- several ReplicaCAD stage assets are open-top shells, so upward-looking rays in
  those scenes may legitimately saturate at the configured horizon
- ceiling visibility therefore depends on source geometry and is not by itself a
  dashboard or projection bug

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
- maximum drone speed defaults to `5.0 m/s` so the proximity speed limiter
  has enough reaction time before geometry contact
- velocity smoothing defaults to `0.15` (fast-tracking) for responsive
  obstacle avoidance maneuvers

### 5.4 Shaping Policy

The environment reward engine combines nine tensor-computed components per step.
All terms are derived from already-produced batched depth and validity data and
do not require a second sensing pipeline.

**Navigation and progress:**

- spatial exploration reward decaying with visit count, heading novelty, and
  frontier adjacency bonuses
- proximity-discounted progress reward: forward displacement is scaled by
  `(1 - proximity_ratio)` so approaching walls yields diminishing progress
  credit instead of a flat distance reward

**Collision avoidance (March 2026 revision):**

- velocity-scaled collision penalty: fast crashes incur `penalty * (1 + speed)`
  so high-speed wall contact is punished more severely than gentle grazing
- positive clearance-delta reward when an actor increases free space while
  within the obstacle clearance window (default `3.0 m`)
- proximity penalty scaling with the fraction of near-field valid hits below
  the proximity threshold (default `2.0 m`)
- clearance-gated exploration: exploration rewards are multiplied by
  `clamp(current_clearance, 0, 1)` so pushing into tight geometry yields
  diminishing exploration credit

**Observation-quality shaping:**

- starvation penalty when horizon-saturated observations dominate the sphere
- positive structure-band reward when stable mid-range geometry stays visible
- positive forward-structure reward when informative geometry remains in the
  forward sector
- inspection reward when turning or repositioning reveals more usable structure
  instead of less

### 5.5 Canonical Shaping Parameters (March 2026)

| Parameter | Default | Env Var | Purpose |
|---|---|---|---|
| `proximity_distance_threshold` | `2.0` m | `NAVI_PROXIMITY_DISTANCE_THRESHOLD` | Near-field classification radius |
| `obstacle_clearance_window` | `3.0` m | `NAVI_OBSTACLE_CLEARANCE_WINDOW` | Escape incentive activation radius |
| `drone_max_speed` | `5.0` m/s | — | Maximum translation speed |
| `collision_penalty` | `-2.0` | — | Base collision penalty (velocity-scaled) |
| `progress_reward_scale` | `0.8` | — | Displacement reward scale (proximity-discounted) |
| `exploration_reward` | `0.3` | — | Base cell-visit exploration reward |
| `starvation_penalty_scale` | `1.5` | `NAVI_STARVATION_PENALTY_SCALE` | Horizon saturation penalty scale |
| `proximity_penalty_scale` | `0.8` | `NAVI_PROXIMITY_PENALTY_SCALE` | Near-field penalty scale |
| `structure_band_reward_scale` | `0.35` | — | Mid-range geometry visibility reward |
| `forward_structure_reward_scale` | `0.2` | — | Forward-sector geometry reward |
| `inspection_reward_scale` | `0.25` | — | Structure-gaining look-around reward |
| `obstacle_clearance_reward_scale` | `0.6` | `NAVI_OBSTACLE_CLEARANCE_REWARD_SCALE` | Clearance delta reward scale |

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
- environment publication is a passive seam; canonical reset and batch-step logic must remain correct with no attached observer

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
- deterministic oracle-fixture checks that preserve exact `(1, Az, El)` depth, delta-depth, validity, and movement-induced view changes before those observations reach the actor or auditor
- canonical `SdfDagBackend` reset selection now scores candidate spawn poses from low-resolution spherical probes so actors start in structure-visible interior views rather than clearance-only void-facing poses
| `max_distance` | `float` | 30.0 | Maximum observation distance (metres) |
| `habitat_scene` | `str` | `""` | Path to Habitat `.glb` scene (Habitat only) |
| `habitat_dataset_config` | `str` | `""` | Path to PointNav episode JSON (Habitat only) |
| `seed` | `int` | 42 | Random seed for deterministic reset selection |
