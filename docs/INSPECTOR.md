# GMDAG 3D Inspector

Interactive viewer and mesh exporter for `.gmdag` (Ghost-Matrix DAG) files.

Extracts SDF isosurfaces via marching cubes and presents them as navigable 3D meshes using PyVista. Multi-resolution support enables instant coarse previews (128³) with on-demand refinement to full detail (512³).

## Quick Start

```powershell
# View a gmdag file interactively
uv run --project projects/inspector inspector view artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\apartment_1.gmdag

# Get file metadata
uv run --project projects/inspector inspector info artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\apartment_1.gmdag

# Export to PLY mesh
uv run --project projects/inspector inspector export scene.gmdag output.ply --resolution 256

# List all gmdag files in the corpus
uv run --project projects/inspector inspector corpus
```

Or via the wrapper script:

```powershell
.\scripts\run-inspector.ps1 -GmdagFile artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\apartment_1.gmdag
.\scripts\run-inspector.ps1 -Action info -GmdagFile scene.gmdag
.\scripts\run-inspector.ps1 -Action corpus
```

## CLI Commands

### `view` — Interactive 3D Viewer

```
navi-inspector view <gmdag_path> [--resolution 128] [--level 0.0]
```

Opens a PyVista interactive window with the extracted isosurface mesh.

**Options:**
- `--resolution, -r` — Initial extraction resolution (32–512, default: 128)
- `--level, -l` — Isosurface level (default: 0.0 = SDF zero-crossing)

### `info` — File Metadata

```
navi-inspector info <gmdag_path>
```

Prints a table with resolution, voxel size, bounding box, node count, and file size.

### `export` — Extract and Save Mesh

```
navi-inspector export <gmdag_path> <output_path> [options]
```

**Options:**
- `--resolution, -r` — Extraction resolution (default: 512 for export)
- `--level, -l` — Isosurface level (default: 0.0)
- `--format, -f` — Output format: `ply`, `obj`, `stl` (default: `ply`)
- `--simplify, -s` — Mesh simplification ratio 0.0–1.0 (default: 1.0 = no simplification)

### `corpus` — List GMDAG Files

```
navi-inspector corpus [corpus_dir]
```

Scans a directory (default: `artifacts/gmdag/corpus`) and lists all `.gmdag` files with metadata.

## Keyboard Shortcuts (Interactive Viewer)

| Key | Action |
|-----|--------|
| **W** | Toggle wireframe / solid rendering |
| **H** | Toggle SDF distance heatmap colouring |
| **B** | Toggle bounding box wireframe |
| **C** | Toggle interactive clipping plane |
| **R** | Refine to full DAG resolution (512³) |
| **I** | Toggle info HUD overlay |
| **E** | Export current mesh to PLY |

**Mouse:**
- Left drag — Orbit
- Right drag — Pan
- Scroll — Zoom

## Architecture

```
projects/inspector/
├── pyproject.toml
├── src/navi_inspector/
│   ├── cli.py           # Typer CLI (view, info, export, corpus)
│   ├── config.py        # Pydantic settings (NAVI_INSPECTOR_*)
│   ├── gmdag_io.py      # Standalone .gmdag binary reader
│   ├── dag_extractor.py # DAG → dense SDF grid extraction
│   ├── mesh_builder.py  # SDF grid → triangle mesh (marching cubes)
│   ├── viewer.py        # PyVista interactive 3D viewer
│   └── cache.py         # Disk cache for extracted meshes
└── tests/
    ├── unit/
    │   ├── test_gmdag_io.py
    │   ├── test_dag_extractor.py
    │   └── test_mesh_builder.py
    └── integration/
```

## Pipeline

```
.gmdag file
    │
    ▼
gmdag_io.load_gmdag()     # Parse 32-byte header + uint64 node pool
    │
    ▼
dag_extractor.extract_sdf_grid()  # Stack-based DAG traversal → dense float32 grid
    │                                  Multi-resolution: 128³ (instant) → 512³ (full)
    ▼
mesh_builder.build_mesh()   # scikit-image marching_cubes → PyVista PolyData
    │
    ├──▶ viewer.launch_viewer()  # Interactive 3D window
    └──▶ mesh_builder.export_mesh()  # Save to PLY/OBJ/STL
```

## Caching

Extracted meshes are cached to disk under `artifacts/inspector/cache/` keyed by:
- Source `.gmdag` file path (SHA-256 hash)
- Extraction resolution
- Isosurface level

Cache invalidation: if the source `.gmdag` modification time is newer than the cached file, the cache is regenerated. Set `NAVI_INSPECTOR_CACHE_DIR` to override the cache location.

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `NAVI_INSPECTOR_DEFAULT_RESOLUTION` | `128` | Default viewer resolution |
| `NAVI_INSPECTOR_EXPORT_RESOLUTION` | `512` | Default export resolution |
| `NAVI_INSPECTOR_CACHE_DIR` | `artifacts/inspector/cache` | Mesh cache directory |

## Project Isolation

The inspector is fully self-contained — it does **not** import from `navi-environment`, `navi-contracts`, `navi-actor`, or any other navi project. It includes its own standalone `.gmdag` binary reader that replicates the canonical format exactly.

## Workflow: Edit and Recompile

The inspector is read-only for `.gmdag` files. To edit geometry:

1. **Export** mesh from inspector: `navi-inspector export scene.gmdag scene.ply`
2. **Edit** in Blender, MeshLab, or other 3D editor
3. **Recompile** back to `.gmdag`: `navi-environment compile-gmdag --input edited.obj --output scene.gmdag --resolution 512`

## Multi-Resolution Strategy

| Resolution | Voxels | Typical Time | Use Case |
|------------|--------|-------------|----------|
| 64³ | ~262K | <0.5s | Quick structural overview |
| **128³** | ~2M | <1s | **Default preview** |
| 256³ | ~16M | 2–5s | Mid-detail inspection |
| **512³** | ~134M | 10–30s | **Full detail / export** |
