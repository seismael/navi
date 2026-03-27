# navi-inspector

**GMDAG 3D Inspector** — interactive viewer and mesh exporter for `.gmdag` files produced by the Navi voxel-dag compiler.

Load any `.gmdag` file, explore the compiled SDF octree as a 3D voxel structure with flight-style navigation (arrow keys / WASD to fly, mouse to look around), inspect metadata, and export to standard mesh formats (PLY, OBJ, STL) for use in external tools like Blender, MeshLab, or CloudCompare.

The default **voxel** mode renders near-surface cells as solid coloured blocks, so walls and floors are clearly visible while architectural openings (doors, windows, corridors) appear as natural gaps. Press **M** to toggle to a traditional marching-cubes surface mesh.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Commands](#cli-commands)
  - [view — Interactive 3D Viewer](#view--interactive-3d-viewer)
  - [info — File Metadata](#info--file-metadata)
  - [export — Mesh Export](#export--mesh-export)
  - [corpus — Corpus Discovery](#corpus--corpus-discovery)
- [Viewer Keyboard Shortcuts](#viewer-keyboard-shortcuts)
- [Viewer Mouse Controls](#viewer-mouse-controls)
- [PowerShell Wrapper Script](#powershell-wrapper-script)
- [Configuration](#configuration)
- [Pipeline Architecture](#pipeline-architecture)
- [GMDAG Binary Format](#gmdag-binary-format)
- [Multi-Resolution Extraction](#multi-resolution-extraction)
- [Mesh Caching](#mesh-caching)
- [Unsigned Distance Field Handling](#unsigned-distance-field-handling)
- [Export Workflow (Blender Integration)](#export-workflow-blender-integration)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Quick Start

```powershell
# From the repository root
cd projects/inspector
uv sync

# View a .gmdag file interactively
uv run navi-inspector view artifacts/gmdag/corpus/ai-habitat_habitat_test_scenes/van-gogh-room.gmdag

# Print file metadata
uv run navi-inspector info artifacts/gmdag/corpus/ai-habitat_habitat_test_scenes/van-gogh-room.gmdag

# Export to PLY at full resolution
uv run navi-inspector export artifacts/gmdag/corpus/ai-habitat_habitat_test_scenes/van-gogh-room.gmdag output.ply

# List all .gmdag files in the corpus
uv run navi-inspector corpus artifacts/gmdag/corpus
```

---

## Installation

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/) package manager.

```powershell
cd projects/inspector
uv sync              # Install all dependencies
uv sync --all-extras # Include dev tools (pytest, ruff, mypy)
```

### Dependencies

| Package           | Purpose                              |
|-------------------|--------------------------------------|
| `numpy`           | Node pool and SDF grid arrays        |
| `pyvista`         | Interactive 3D rendering (VTK-based) |
| `scikit-image`    | Marching cubes isosurface extraction  |
| `typer`           | CLI framework                        |
| `rich`            | Terminal formatting and tables        |
| `pydantic-settings` | Configuration management           |

---

## CLI Commands

The tool provides four commands accessible via `uv run navi-inspector <command>` or the shortcut `uv run inspector <command>`.

### `view` — Interactive 3D Viewer

Opens a PyVista 3D window for interactive exploration of the compiled SDF octree.

```
navi-inspector view <GMDAG_PATH> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--resolution` | `-r` | `128` | Initial extraction resolution (32–512). Lower values load faster. |
| `--level` | `-l` | auto | Isosurface level. Omit for auto-detection (recommended). |

**Examples:**

```powershell
# Quick preview at 64³ (instant)
uv run navi-inspector view scene.gmdag -r 64

# Mid-detail preview
uv run navi-inspector view scene.gmdag -r 256

# Full resolution (slower, more detail)
uv run navi-inspector view scene.gmdag -r 512

# Explicit iso-level override
uv run navi-inspector view scene.gmdag --level 0.05
```

The viewer opens at a coarse resolution first for instant feedback. Press **R** inside the viewer to refine to full DAG resolution without restarting.

---

### `info` — File Metadata

Prints a formatted table with header metadata from a `.gmdag` file. Does not load the full node pool — reads only the 32-byte header.

```
navi-inspector info <GMDAG_PATH>
```

**Example output:**

```
┌──────────────┬────────────────────────────────────────────────┐
│ Field        │ Value                                          │
├──────────────┼────────────────────────────────────────────────┤
│ Path         │ artifacts/gmdag/corpus/.../van-gogh-room.gmdag │
│ Version      │ 1                                              │
│ Resolution   │ 512                                            │
│ Voxel Size   │ 0.017128 m                                     │
│ BBox Min     │ (-1.78, -4.48, -2.48)                          │
│ BBox Max     │ (6.99, 4.29, 6.29)                             │
│ Node Count   │ 885,522                                        │
│ File Size    │ 6.76 MB (7,084,208 B)                          │
│ World Extent │ 8.77 m                                         │
└──────────────┴────────────────────────────────────────────────┘
```

---

### `export` — Mesh Export

Extracts the SDF isosurface and writes a standard mesh file.

```
navi-inspector export <GMDAG_PATH> <OUTPUT_PATH> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--resolution` | `-r` | `0` (config default: 512) | Extraction resolution. `0` uses the config export default. |
| `--level` | `-l` | auto | Isosurface level. Omit for auto-detection. |
| `--format` | `-f` | `ply` | Output format: `ply`, `obj`, or `stl`. |
| `--simplify` | `-s` | `1.0` | Triangle reduction ratio. `0.5` = keep ~50% of faces. |

**Examples:**

```powershell
# Export full-resolution PLY
uv run navi-inspector export scene.gmdag scene.ply

# Export simplified OBJ at 256³
uv run navi-inspector export scene.gmdag scene.obj -r 256 -f obj -s 0.3

# Export STL for 3D printing / CAD tools
uv run navi-inspector export scene.gmdag scene.stl -f stl

# Export at quick preview resolution
uv run navi-inspector export scene.gmdag preview.ply -r 64
```

---

### `corpus` — Corpus Discovery

Recursively scans a directory for `.gmdag` files and prints a summary table with metadata for each file.

```
navi-inspector corpus [CORPUS_DIR]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `CORPUS_DIR` | `artifacts/gmdag/corpus` | Directory to scan recursively. |

**Example:**

```powershell
uv run navi-inspector corpus artifacts/gmdag/corpus
```

```
┌───┬──────────────────────────────────────────────────────┬─────┬──────────┬───────────┐
│ # │ File                                                 │ Res │    Nodes │ Size (MB) │
├───┼──────────────────────────────────────────────────────┼─────┼──────────┼───────────┤
│ 1 │ ai-habitat_habitat_test_scenes\apartment_1.gmdag     │ 512 │  2,880k  │     22.00 │
│ 2 │ ai-habitat_habitat_test_scenes\skokloster-castle.gmdag │ 512 │  3,472k │     26.53 │
│ 3 │ ai-habitat_habitat_test_scenes\van-gogh-room.gmdag   │ 512 │    886k  │      6.76 │
│ …                                                                                     │
└───┴──────────────────────────────────────────────────────┴─────┴──────────┴───────────┘
10 files, 142.3 MB total
```

---

## Viewer Keyboard Shortcuts

Once the 3D viewer window is open, the following controls are available:

### Navigation (VTK flight style)

Hold arrow keys or WASD to fly. The camera moves continuously while a key is held.

| Key | Action | Description |
|-----|--------|-------------|
| **↑ Arrow / W** | Fly forward | Move along the camera facing direction. |
| **↓ Arrow / S** | Fly backward | Move opposite to the camera facing direction. |
| **← Arrow** | Turn left | Yaw the camera left while flying. |
| **→ Arrow** | Turn right | Yaw the camera right while flying. |
| **A** | Fly up | Move up along the world vertical axis. |
| **Z** | Fly down | Move down along the world vertical axis. |
| **+** | Speed up | Double the flight speed. |
| **-** | Speed down | Halve the flight speed. |

### Display toggles

| Key | Action | Description |
|-----|--------|-------------|
| **M** | Toggle render mode | Switch between **voxel blocks** (default) and **surface mesh**. Voxels show solid coloured blocks; surface uses marching-cubes. |
| **F** | Toggle wireframe | Switch between solid surface and wireframe rendering (surface mode only). |
| **H** | Toggle heatmap | Colour by SDF distance using a `turbo` colourmap, or switch to a flat colour. |
| **B** | Toggle bounding box | Show/hide the translucent world-space bounding box wireframe. |
| **I** | Toggle info HUD | Show/hide the information overlay (resolution, block/vertex count). |
| **C** | Toggle clip plane | Add/remove an interactive Z-axis clipping plane widget. |
| **R** | Refine to full resolution | Re-extract at the full DAG resolution (e.g. 512³). Shows a progress message. |
| **E** | Export current mesh | Save the surface mesh as PLY to `artifacts/inspector/exports/`. |
| **Q** | Quit | Close the viewer window. |

---

## Viewer Mouse Controls

In flight mode the mouse controls where you look:

| Input | Action |
|-------|--------|
| **Left-click + drag** | Pitch and yaw the camera (look around) |

> **Tip:** Hold an arrow key or W to fly forward while dragging the mouse to steer. Use **+** and **-** to adjust flight speed.

---

## PowerShell Wrapper Script

A convenience wrapper is provided at `scripts/run-inspector.ps1`:

```powershell
# View (default action)
.\scripts\run-inspector.ps1 -GmdagFile artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\apartment_1.gmdag

# View at higher resolution
.\scripts\run-inspector.ps1 -GmdagFile scene.gmdag -Resolution 256

# Print metadata
.\scripts\run-inspector.ps1 -Action info -GmdagFile scene.gmdag

# Export to PLY
.\scripts\run-inspector.ps1 -Action export -GmdagFile scene.gmdag

# List corpus
.\scripts\run-inspector.ps1 -Action corpus
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-GmdagFile` | (required for view/info/export) | Path to the `.gmdag` file. |
| `-Resolution` | `128` | Extraction grid resolution. |
| `-Action` | `view` | One of: `view`, `info`, `export`, `corpus`. |

---

## Configuration

Settings are managed via `pydantic-settings` and can be configured through environment variables or a `.env` file (searched upward from the project directory).

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `NAVI_INSPECTOR_DEFAULT_RESOLUTION` | `128` | Default resolution for the interactive viewer. |
| `NAVI_INSPECTOR_EXPORT_RESOLUTION` | `512` | Default resolution for the `export` command when `--resolution` is not specified. |
| `NAVI_INSPECTOR_CACHE_DIR` | `artifacts/inspector/cache` | Directory for cached extracted meshes. |

**Example `.env` override:**

```ini
NAVI_INSPECTOR_DEFAULT_RESOLUTION=256
NAVI_INSPECTOR_EXPORT_RESOLUTION=512
NAVI_INSPECTOR_CACHE_DIR=artifacts/inspector/cache
```

---

## Pipeline Architecture

The inspector follows a linear extraction pipeline:

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  .gmdag file │────▶│  DAG Extractor │────▶│ Mesh Builder │────▶│   Viewer /  │
│  (binary)    │     │  (SDF grid)    │     │ (marching    │     │   Exporter  │
│              │     │                │     │  cubes)      │     │             │
└──────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
      │                                          │
      ▼                                          ▼
  gmdag_io.py                              mesh_builder.py
  load_gmdag()                             build_mesh()
  gmdag_info()                             export_mesh()
```

**Stage 1 — GMDAG I/O** (`gmdag_io.py`): Reads and validates the 32-byte header and loads the uint64 node pool into a numpy array. Validates magic bytes, version, resolution, voxel size, bounding box finiteness, and node count vs actual file size.

**Stage 2 — DAG Extraction** (`dag_extractor.py`): Walks the octree from root to leaves using an iterative stack-based traversal (no recursion limits). Produces a dense float32 SDF grid at the requested target resolution. When `target_resolution < asset.resolution`, traversal stops early and broadcasts leaf values to covered cells.

**Stage 3 — Mesh Building** (`mesh_builder.py`): Runs scikit-image `marching_cubes()` on the SDF grid to extract the isosurface as triangles. Transforms vertices from grid-local to world coordinates, attaches SDF distance as a scalar field, and optionally decimates the mesh.

**Stage 4 — Output**: Either renders interactively via PyVista (`viewer.py`) or saves to file (`export_mesh()`).

---

## GMDAG Binary Format

The `.gmdag` format is a compact binary representation of a compiled SDF octree.

### Header (32 bytes, little-endian)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | `char[4]` | magic | `"GDAG"` |
| 4 | `uint32` | version | Format version (currently `1`) |
| 8 | `uint32` | resolution | Grid resolution (e.g. `512`) |
| 12 | `float32` | bmin_x | Bounding box minimum X |
| 16 | `float32` | bmin_y | Bounding box minimum Y |
| 20 | `float32` | bmin_z | Bounding box minimum Z |
| 24 | `float32` | voxel_size | Side length of one voxel in metres |
| 28 | `uint32` | node_count | Number of 64-bit nodes following |

### Node Pool (`node_count` × `uint64`)

Each 64-bit word is either a **leaf** or **inner** node:

**Leaf Node** (bit 63 = 1):
```
┌─────┬──────────────────────────────┬──────────────────┬────────────────────────────────┐
│ 63  │ 62-32                        │ 31-16            │ 15-0                           │
│  1  │ (unused)                     │ semantic (u16)   │ distance (fp16)                │
└─────┴──────────────────────────────┴──────────────────┴────────────────────────────────┘
```

**Inner Node** (bit 63 = 0):
```
┌─────┬──────────┬──────────────────┬──────────────────┬────────────────────────────────┐
│ 63  │ 62-56    │ 55-48            │ 47-32            │ 31-0                           │
│  0  │ (unused) │ mask (8 bits)    │ (unused)         │ child_base (32 bits)           │
└─────┴──────────┴──────────────────┴──────────────────┴────────────────────────────────┘
```

- **mask**: 8-bit occupancy bitmask — one bit per octant (0–7). Set if the octant contains geometry.
- **child_base**: Index into the node pool pointing to a contiguous array of child pointers.
- **child lookup**: For octant `i`, offset = `popcount(mask & ((1 << i) - 1))`. The child's node index is `dag[child_base + offset]`.

---

## Multi-Resolution Extraction

The extractor supports coarse-to-fine multi-resolution:

| Resolution | Grid Size | Typical Time | Use Case |
|------------|-----------|--------------|----------|
| `32` | 32³ = 33K voxels | < 0.5s | Thumbnail / shape check |
| `64` | 64³ = 262K voxels | ~2s | Quick preview |
| `128` | 128³ = 2M voxels | ~10s | Default interactive viewer |
| `256` | 256³ = 17M voxels | ~60s | Mid-detail inspection |
| `512` | 512³ = 134M voxels | several min | Full DAG resolution |

The viewer starts at the coarse resolution for instant feedback. Press **R** to refine to the full DAG resolution in-place.

---

## Mesh Caching

Extracted meshes are cached on disk to avoid redundant recomputation when viewing the same file at the same resolution.

- **Cache location**: `artifacts/inspector/cache/` (configurable via `NAVI_INSPECTOR_CACHE_DIR`)
- **Cache key**: SHA-256 hash of `{absolute_path}::{resolution}::{level}`
- **Invalidation**: Automatic — if the source `.gmdag` file's modification time is newer than the cached PLY, the cache entry is discarded and re-extracted.
- **Format**: Cached meshes are stored as `.ply` files.

---

## Unsigned Distance Field Handling

The Navi voxel-dag compiler produces **unsigned** distance fields (all values ≥ 0; surface is at distance ≈ 0). The inspector auto-detects this:

- When all grid values are ≥ 0, the iso-level is automatically set to `voxel_size / 2`
- When negative values are present (signed SDF), the iso-level defaults to `0.0`
- You can always override with `--level <value>` if the auto-detection doesn't suit your needs

---

## Export Workflow (Blender Integration)

The inspector is designed for a **read-only inspection + export** workflow. To edit geometry:

1. **Export** the mesh:
   ```powershell
   uv run navi-inspector export scene.gmdag scene.ply -r 256 -f ply
   ```

2. **Open in Blender** (or MeshLab, CloudCompare, etc.):
   - File → Import → Stanford PLY (or Wavefront OBJ / STL)
   - The mesh includes SDF distance values as a vertex attribute

3. **Edit** in Blender's sculpt/edit mode

4. **Re-export** from Blender as OBJ/PLY

5. **Recompile** back to `.gmdag` using the voxel-dag compiler

---

## Running Tests

```powershell
cd projects/inspector
uv sync --all-extras

# Run all unit tests
uv run pytest tests/unit -v

# Run with coverage (if installed)
uv run pytest tests/unit -v --tb=short

# Run a specific test module
uv run pytest tests/unit/test_gmdag_io.py -v
uv run pytest tests/unit/test_dag_extractor.py -v
uv run pytest tests/unit/test_mesh_builder.py -v
```

### Test Suite

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_gmdag_io.py` | 12 tests | Header parsing, all validation gates (bad magic, version, resolution, voxel_size, bbox, trailing bytes, truncated nodes) |
| `test_dag_extractor.py` | 5 tests | Single-leaf fill, two-level octant values, void octants, coarse extraction, metadata |
| `test_mesh_builder.py` | 8 tests | Sphere mesh, SDF scalars, world coords, simplification, error cases, PLY/STL export |
| **Total** | **26 tests** | |

### Linting

```powershell
uv run ruff check src/ tests/
uv run mypy src/
```

---

## Troubleshooting

### PyVista window doesn't open

PyVista requires a display. On headless systems, set the backend to `static` or `html`:
```python
import pyvista as pv
pv.set_jupyter_backend('static')
```

### "No isosurface at level=..." error

The SDF grid has no surface crossing at the requested level. Try:
- Omit `--level` to use auto-detection
- Use a smaller level value (e.g. `--level 0.01`)
- Increase resolution to capture finer geometry (`-r 256`)

### Import error for PyVista/mypy

PyVista 0.47+ has a mypy plugin that requires mypy to be installed. This is already handled by the dependencies. If you encounter issues:
```powershell
uv pip install --force-reinstall mypy
```

### Windows pytest temp directory errors

If pytest fails with `make_numbered_dir_with_cleanup` errors on Windows, use a local basetemp:
```powershell
uv run pytest tests/unit --basetemp=tests/.pytest_tmp
```

### Slow extraction at high resolution

Extraction at 512³ processes 134M voxels and can take several minutes. Start with `-r 64` or `-r 128` for quick previews, then use **R** in the viewer to refine on demand.

---

## Project Structure

```
projects/inspector/
├── pyproject.toml                    # Build config, dependencies, entry points
├── README.md                         # This file
├── src/
│   └── navi_inspector/
│       ├── __init__.py               # Package init
│       ├── py.typed                  # PEP 561 marker
│       ├── cli.py                    # Typer CLI (view, info, export, corpus)
│       ├── config.py                 # Pydantic settings configuration
│       ├── gmdag_io.py               # .gmdag binary reader + validator
│       ├── dag_extractor.py          # DAG → dense SDF grid extraction
│       ├── mesh_builder.py           # Marching cubes + mesh export
│       ├── cache.py                  # Disk cache with SHA-256 keying
│       └── viewer.py                 # PyVista interactive 3D viewer
└── tests/
    ├── unit/
    │   ├── test_gmdag_io.py          # 12 tests — I/O + validation
    │   ├── test_dag_extractor.py     # 5 tests — extraction + metadata
    │   └── test_mesh_builder.py      # 8 tests — meshing + export
    └── integration/                  # (placeholder for future tests)
```

---

## See Also

- [docs/INSPECTOR.md](../../docs/INSPECTOR.md) — Architecture documentation
- [docs/GMDAG.md](../../docs/GMDAG.md) — GMDAG format specification
- [docs/COMPILER.md](../../docs/COMPILER.md) — Voxel-DAG compiler documentation
- [projects/voxel-dag/](../voxel-dag/) — The compiler that produces `.gmdag` files
- [projects/torch-sdf/](../torch-sdf/) — CUDA sphere-tracing runtime that consumes `.gmdag` files
