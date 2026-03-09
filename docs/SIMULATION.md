# SIMULATION.md — Canonical Environment Runtime

## 1. Overview

The environment layer now exposes one active runtime only:

- compiled `.gmdag` assets prepared from source scenes
- CUDA sphere tracing through `projects/torch-sdf`
- `DistanceMatrix v2` observations preserved for the actor contract
- coarse `environment.sdfdag.perf` telemetry for throughput tracking
- staged corpus refresh that removes transient downloads after successful integration

The environment remains the sole spatial truth layer for Navi.

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

`SimulatorBackend` remains the environment abstraction boundary, but the repository now keeps only one production implementation:

```text
SimulatorBackend
  └── SdfDagBackend — batched CUDA sphere tracing against compiled `.gmdag` caches
```

Key runtime properties:

- one canonical compiled-scene execution path
- no procedural generators
- no mesh, habitat, or voxel fallback runtime
- no sparse-Zarr world format in the active environment surface
- the CUDA ray tracer must stop at the configured environment horizon rather than a separate hardcoded kernel limit

## 4. Corpus Preparation

`prepare-corpus` discovers source scenes, compiles missing or stale `.gmdag` assets, and writes:

- source manifest under the source root
- compiled manifest under the compiled corpus root

Canonical refresh policy:

- default compile resolution is `512`
- compiled assets with the wrong stored resolution are rebuilt automatically during refresh
- `scripts/refresh-scene-corpus.ps1` stages source downloads and only promotes the compiled corpus after a full successful rebuild
- transient source downloads are removed after successful promotion so the persistent runtime depends on the compiled corpus, not raw meshes
- promoted `gmdag_manifest.json` files rewrite `source_path` to the live compiled asset path; deleted scratch download paths must not remain in the live corpus metadata

Canonical scene/episode policy:

- collision during training remains non-terminal; the backend reverts the pose, applies a penalty, and continues the episode
- hard truncation remains `2000` steps by default
- scene-pool rotation defaults to `16` completed episodes per scene across the fleet, reducing expensive scene swaps and giving actors more time to recover and explore locally
- near-obstacle reward includes a positive clearance-delta signal when an actor increases free space after getting close to geometry
- horizon-saturated observations now incur a starvation penalty once invalid rays dominate the sphere beyond the configured threshold
- very near valid hits now incur a proximity penalty so persistent wall-hugging is discouraged without changing collision non-terminal behavior

TSDF-derived integration policy:

- the first accepted TSDF-derived runtime change is horizon alignment between `EnvironmentConfig.max_distance` and `torch-sdf` kernel termination
- starvation/proximity reward shaping is implemented at the environment reward seam using already-produced batched depth/validity tensors
- compiler-side truncation radius and Morton/Z-order storage remain benchmark-gated experiments until they prove better than the current `.gmdag` path

Canonical roots:

- transient source staging: `artifacts/tmp/corpus-refresh/downloads/`
- compiled corpus: `artifacts/gmdag/corpus/`

## 5. Observation Contract

The environment still publishes the actor’s required contract unchanged:

- `depth`
- `delta_depth`
- `semantic`
- `valid_mask`

All arrays remain shaped `(1, Az, El)`.

## 6. Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` / `step_result_v2`

## 7. Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

---

## 3. DatasetAdapter Protocol

**Module:** `backends/adapter.py`

Formalizes the boundary between external data sources and the training engine's
canonical DistanceMatrix contract. The adapter is the **only** place where raw
external observations are transformed into the engine's format.

```python
@runtime_checkable
class DatasetAdapter(Protocol):
    @property
    def metadata(self) -> AdapterMetadata: ...

    def adapt(self, raw_obs: dict[str, Any], step_id: int) -> dict[str, NDArray]: ...

    def reset(self) -> None: ...
```

### 3.1. AdapterMetadata

| Field | Type | Description |
|-------|------|-------------|
| `azimuth_bins` | `int` | Output azimuth resolution |
| `elevation_bins` | `int` | Output elevation resolution |
| `max_distance` | `float` | Maximum depth in metres (normalization divisor) |
| `semantic_classes` | `int` | Number of semantic classes produced (0…N-1) |

### 3.2. Adapt Transform Chain

The `adapt()` method performs the following transformations in order:

1. **Axis transpose:** Raw `(El, Az)` → canonical `(Az, El)`.
2. **Depth normalization:** Metres → `[0, 1]` via `clip(d, 0, max_dist) / max_dist`.
3. **Semantic remapping:** External instance/category IDs → Navi's `[0, 10]`
   range via `HabitatSemanticLUT`.
4. **Delta-depth computation:** Frame difference `depth_t - depth_{t-1}`.
5. **Valid-mask computation:** `raw_depth > 0` (marks bins with actual ray hits).
6. **Overhead minimap generation:** Top-down projection from depth data with
   Turbo-style distance colormap.
7. **Env-dimension insertion:** `(Az, El)` → `(1, Az, El)`.

### 3.3. Implemented Adapters

**HabitatAdapter** (`backends/habitat_adapter.py`):

Converts Habitat equirectangular sensor output to canonical arrays. Uses
`HabitatSemanticLUT` for semantic remapping — a uint8 lookup table built from
`habitat_sim.SemanticScene.objects` that maps instance IDs to Navi's 0–10
semantic range via the `REPLICACAD_CATEGORY_MAP`.

**Navi semantic ID table:**

| ID | Category | Examples |
|----|----------|----------|
| 0 | Air / empty | Void, unknown |
| 1 | Floor / ground | Floor surfaces |
| 2 | Wall / barrier | Walls |
| 3 | Ceiling / overhead | Ceiling |
| 4 | Furniture / obstacle | Chair, table, sofa, bed, cabinet |
| 5 | Object / interactable | Lamp, book, cushion, appliance |
| 6 | Structure | Door, window, column, stairs |
| 7 | Vegetation / organic | Plants |
| 8 | Water / liquid | — |
| 9 | Hazard / dynamic obstacle | — |
| 10 | Target / goal | Navigation targets |

### 3.4. SOLID Adapter Ecosystem (Roadmap)

Following the Open/Closed Principle, additional adapters can be implemented
without modifying the training engine:

- **PointCloudAdapter** (planned): Ingests raw `.las` / `.ply` LiDAR scans,
  computing continuous surface normals and projecting into spherical depth.
- **ProceduralAdapter** (planned): Wraps procedural generation algorithms,
  translating algorithmic structures into geometric boundaries.
- **IsaacAdapter** (planned): Bridge to NVIDIA Isaac Sim for high-fidelity
  robot simulation.

---

## 4. World Compilation Pipeline

Two world-compilation paths currently coexist in Navi:

- **Sparse Zarr voxel worlds** for the current `VoxelBackend` flow.
- **`.gmdag` worlds** for the internal `projects/voxel-dag` + `projects/torch-sdf`
  performance path.

The performance-first direction of the repository is to promote `.gmdag` to the
canonical compiled-world format for high-throughput training once the runtime
backend is validated.

### 4.1. WorldModelCompiler / PlyWorldCompiler

**Module:** `transformers/compiler.py`

Offline tool that converts mesh assets into the canonical sparse Zarr chunk
format for `VoxelBackend` consumption.

This compiler remains part of the repository for the current voxel path, but it
is no longer the sole strategic compilation story.

**Supported input formats:** PLY, OBJ, ASCII STL.

**Output:** Zarr archive with sparse chunk layout:
- `chunk_index` — array mapping chunk coordinates to data offsets.

### 4.2. Internal `.gmdag` Compiler Path

The imported `projects/voxel-dag` project is now the in-repo compiler domain
for the high-performance Ghost-Matrix path.

- **Input:** `.glb`, `.obj`, `.ply`, and related mesh assets.
- **Output:** `.gmdag` binary caches with cubic world alignment and DAG-compressed
  distance payloads.
- **Entry points:** `projects/voxel-dag/src/main.cpp` and the package entry
  point exposed by `projects/voxel-dag/pyproject.toml`.

Navi surfaces this flow through `navi-environment compile-gmdag` so the
Environment project can own orchestration without reimplementing the compiler.
`navi-environment check-sdfdag` is the paired preflight command for validating
compiler/runtime readiness and `.gmdag` metadata before rollout.
`navi-environment bench-sdfdag` runs the same canonical batched backend without
the actor layer so throughput can be measured directly before stack cutover.
- `chunks/<cx>_<cy>_<cz>` — per-chunk voxel arrays `(N, 5)`.

**CLI:**

```bash
uv run navi-environment compile-world \
  --source ./worlds/city.ply \
  --source-format ply \
  --output ./worlds/city.zarr
```

### 4.2. World Generators

**Module:** `generators/`

Procedural world generators implementing `AbstractWorldGenerator`:

| Generator | Description |
|-----------|-------------|
| `ArenaGenerator` | Simple walled arena with optional pillars/obstacles |
| `CityGenerator` | Multi-block urban layout with buildings and streets |
| `MazeGenerator` | Recursive maze with configurable complexity |
| `RoomsGenerator` | Connected room layouts with doorways |
| `Open3DVoxelGenerator` | Open3D-based voxelization of mesh files |
| `FileLoaderGenerator` | Loads pre-compiled Zarr world files |

Each generator implements:
- `generate_chunk(cx, cy, cz) → NDArray` — produce voxel data for a chunk.
- `spawn_position() → (x, y, z)` — deterministic starting position.

---

## 5. Raycasting Engine

### 5.1. RaycastEngine

**Module:** `raycast.py`

Vectorized spherical-projection Z-buffer that projects local voxels into
discretized azimuth/elevation bins.

**Algorithm:**

1. Compute relative positions of all voxels to the robot.
2. Convert to spherical coordinates:
   - Azimuth: $\theta = \text{atan2}(z, x)$ (horizontal XZ plane).
   - Elevation: $\phi = \text{atan2}(y, \sqrt{x^2 + z^2})$ (vertical lift).
3. Discretize into bin indices: $\theta \to [0, \text{Az})$,
   $\phi \to [0, \text{El})$.
4. Normalize distances: $d_{norm} = \text{clip}(d / d_{max}, 0, 1)$.
5. **Scatter-reduce:** `np.minimum.at(depth_flat, flat_indices, normalized)` —
   nearest hit per bin.
6. **Semantic assignment:** Sort by distance, scatter closest voxel's semantic
   ID per bin.
7. **Valid mask:** Mark any bin that received at least one hit.

**Performance:** Scales to 256×128+ bins and tens of thousands of voxels
without measurable overhead. Fully vectorized — no Python for-loops in the
hot path.

### 5.2. DistanceMatrixBuilder

**Module:** `distance_matrix_v2.py`

Wraps `RaycastEngine` and adds:

- **Heading-relative rotation:** Voxel positions are rotated into the robot's
  forward-facing frame before projection, so the depth panorama is centered
  on the robot's heading direction.
- **Delta-depth:** Temporal velocity awareness channel — per-bin change since
  the last step.
- **Overhead minimap:** 256×256 BGR top-down view centered on the robot with
  semantic colormap and heading arrow.

### 5.3. MjxEnvironment

**Module:** `mjx_env.py`

Body-frame kinematic pose stepping. Integrates velocity commands from `Action`
into pose updates using configurable time delta (default `dt=1.0`).

### 5.4. Chunk Pruning

**Module:** `pruning.py`

Utilities for memory-efficient chunk lifecycle management. Prunes chunks that
fall outside the sliding window radius, reducing memory footprint during
long-running training sessions in large environments.

---

## 6. Shape Convention

All backends produce arrays with the following canonical conventions:

| Property | Value |
|----------|-------|
| `matrix_shape` | `(azimuth_bins, elevation_bins)` — default `(256, 48)` |
| Axis ordering | `(n_envs, azimuth, elevation)` |
| Single-env shape | `n_envs = 1` → `(1, Az, El)` |
| Depth range | `[0, 1]` — normalized by `max_distance` (default 30.0 m) |
| Semantic range | `int32` in `[0, 10]` |
| Overhead minimap | `(256, 256, 3)` BGR uint8 or float32 |

---

## 7. Roadmap: SDF Compiler & DAG Engine

### 7.1. SDF Compiler

Offline mathematical distance field evaluation from adapted geometry. The
compiler calculates the exact distance to the nearest geometric boundary for
all points in simulated space, yielding a continuous Signed Distance Field.

### 7.2. Sparse Voxel Octree (SVO) Compression

The environment is recursively subdivided into an octree. If a large region
contains no geometry and a uniform SDF gradient, subdivision halts — empty
space is represented by a single high-level node. This reduces memory footprint
by orders of magnitude compared to dense grids.

### 7.3. DAG Deduplication

The octree is folded into a **Directed Acyclic Graph** by identifying
mathematically identical SDF branches across the environment and merging their
memory pointers. For procedural environments with repetitive structures
(corridors, rooms), this achieves further orders-of-magnitude compression,
ensuring the environment fits within GPU L1/L2 caches for bandwidth-optimal
parallel raycasting.

### 7.4. `.gmdag` Binary Format

The compiler outputs `.gmdag` (Ghost-Matrix DAG) binary cache files. Leaf nodes
contain the tuple `[float32 distance, int32 semantic_id]`. The runtime engine
operates exclusively on compiled `.gmdag` caches — it possesses no knowledge of
the original dataset format.

### 7.5. CUDA/Triton Sphere Tracing Kernel

A GPU-optimized kernel executes sphere tracing across the DAG structure.
Memory access patterns are localized to exploit the DAG's deduplication. The
kernel outputs structured float tensors directly in PyTorch's memory space
(zero CPU-GPU transfer). See [ARCHITECTURE.md §8.3](ARCHITECTURE.md) for
full specification.

---

## 8. Configuration Reference

**Module:** `config.py` — `EnvironmentConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pub_address` | `str` | `tcp://*:5559` | PUB socket bind address |
| `rep_address` | `str` | `tcp://*:5560` | REP socket bind address |
| `action_sub_address` | `str` | `tcp://localhost:5557` | SUB socket for async mode |
| `mode` | `str` | `step` | `step` (REQ/REP) or `async` (SUB/PUB) |
| `backend` | `str` | `sdfdag` | Backend selection: `sdfdag` (canonical). `voxel`, `habitat`, and `mesh` remain diagnostic-only. |
| `world_source` | `str` | `procedural` | `procedural` or `file` |
| `world_file` | `str` | `""` | Path to Zarr world file (when `file` source) |
| `generator` | `str` | `arena` | World generator: `arena`, `city`, `maze`, `rooms`, `open3d` |
| `chunk_size` | `int` | 16 | Voxel chunk edge length |
| `window_radius` | `int` | 2 | Sliding window radius in chunks |
| `lookahead_margin` | `int` | 8 | Predictive prefetch distance |
| `barrier_distance` | `float` | 0.0 | Minimum standoff from occupied voxels |
| `collision_probe_radius` | `float` | 1.5 | Radius for collision constraint check |
| `azimuth_bins` | `int` | 256 | Azimuth resolution |
| `elevation_bins` | `int` | 128 | Elevation resolution |
| `max_distance` | `float` | 30.0 | Maximum observation distance (metres) |
| `habitat_scene` | `str` | `""` | Path to Habitat `.glb` scene (Habitat only) |
| `habitat_dataset_config` | `str` | `""` | Path to PointNav episode JSON (Habitat only) |
| `seed` | `int` | 42 | Random seed for procedural generation |
