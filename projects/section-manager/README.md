# navi-section-manager

Simulation Layer for Ghost-Matrix runtime.

## Responsibilities

- Executes headless simulation steps.
- Receives `Action v2` commands.
- Publishes `DistanceMatrix v2` observations.
- Supports step-mode and async mode over ZMQ.

## Usage

```bash
cd projects/section-manager
uv sync
uv run navi-section-manager serve --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Compile model assets (.ply/.obj/.stl) into sparse chunked world format
uv run navi-section-manager compile-world --source ./worlds/city.ply --output ./worlds/city.zarr
uv run navi-section-manager compile-world --source ./worlds/city.obj --source-format obj --output ./worlds/city.zarr

# Run with file-backed world
uv run navi-section-manager serve --world-source file --world-file ./worlds/city.zarr --mode step --pub tcp://*:5559 --rep tcp://*:5560
```

## World Format

- Canonical file-backed world format is Zarr only.
- Supports sparse chunk layout (`chunk_index` + `chunks/<cx>_<cy>_<cz>`) for efficient storage.
- Legacy dense `voxels` dataset remains readable for compatibility.
- `compile-world` supports source model ingestion from PLY, OBJ, and ASCII STL.

## Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` / `step_result_v2`
