# navi-environment

Simulation Layer for Ghost-Matrix runtime.

**Full specification:** [docs/SIMULATION.md](../../docs/SIMULATION.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Responsibilities

- Executes headless simulation steps via pluggable `SimulatorBackend` implementations.
- Receives `Action v2` commands.
- Publishes `DistanceMatrix v2` observations in canonical `(1, Az, El)` shape.
- Supports step-mode and async mode over ZMQ.

## Backend Architecture

The server delegates physics and sensing to a `SimulatorBackend` ABC:

```
SimulatorBackend (ABC)
  ├── VoxelBackend       — procedural voxel worlds + RaycastEngine (default)
  ├── HabitatBackend     — Meta habitat-sim with DatasetAdapter
  └── MeshSceneBackend   — trimesh ray-mesh intersection for .glb/.obj/.ply
```

External backends (Habitat, etc.) use a `DatasetAdapter` (Protocol) internally
to convert raw sensor data into canonical DistanceMatrix format:

```
DatasetAdapter Protocol
  .adapt(raw_obs, step_id) → dict[str, NDArray]   # canonical (1, Az, El) arrays
  .reset()                                          # clear frame-diff state
  .metadata → AdapterMetadata                       # az/el bins, LUT info
```

The adapter handles axis transpose, depth normalisation to `[0, 1]`, semantic
class remapping, delta-depth computation, valid-mask, and env-dimension insertion.
The training engine never changes — adapters always transform *to* its format.

## World Generators

Procedural world generators implementing `AbstractWorldGenerator`:

- `ArenaGenerator` — simple walled arena with optional obstacles
- `CityGenerator` — multi-block urban layout with buildings
- `MazeGenerator` — recursive maze with configurable complexity
- `RoomsGenerator` — connected room layouts with doorways
- `Open3DVoxelGenerator` — Open3D-based voxelization of mesh files
- `FileLoaderGenerator` — loads pre-compiled Zarr world files

## Usage

```bash
cd projects/environment
uv sync

# Default voxel backend
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Habitat backend (requires habitat-sim)
uv run navi-environment serve --backend habitat --habitat-scene /path/to/scene.glb --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Mesh backend (trimesh — no habitat-sim required)
uv run navi-environment serve --backend mesh --mesh-scene /path/to/scene.glb --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Compile model assets (.ply/.obj/.stl) into sparse chunked world format
uv run navi-environment compile-world --source ./worlds/city.ply --output ./worlds/city.zarr
uv run navi-environment compile-world --source ./worlds/city.obj --source-format obj --output ./worlds/city.zarr

# Run with file-backed world
uv run navi-environment serve --world-source file --world-file ./worlds/city.zarr --mode step --pub tcp://*:5559 --rep tcp://*:5560
```

## World Format

- Canonical file-backed world format is Zarr only.
- Supports sparse chunk layout (`chunk_index` + `chunks/<cx>_<cy>_<cz>`) for efficient storage.
- Legacy dense `voxels` dataset remains readable for compatibility.
- `compile-world` supports source model ingestion from PLY, OBJ, and ASCII STL.

## Shape Convention

All backends produce arrays with axis ordering `(n_envs, azimuth, elevation)`:
- `matrix_shape[0]` = azimuth bins (rows)
- `matrix_shape[1]` = elevation bins (columns)
- Single-env backends use `n_envs = 1` → shape `(1, Az, El)`
- Depth is normalised to `[0, 1]`

## Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` / `step_result_v2`

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
