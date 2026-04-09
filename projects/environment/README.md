← [Navi Overview](../../README.md)

# navi-environment

Simulation Layer for the Navi ecosystem. Executes headless batched environment
stepping against compiled `.gmdag` assets using the CUDA `sdfdag` backend and
publishes `DistanceMatrix v2` observations.

**Full specification:** [docs/SIMULATION.md](../../docs/SIMULATION.md)  
**Runtime details:** [docs/SDFDAG_RUNTIME.md](../../docs/SDFDAG_RUNTIME.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Table of Contents

- [Install](#install)
- [Responsibilities](#responsibilities)
- [CLI Reference](#cli-reference)
- [Corpus Preparation](#corpus-preparation)
- [Corpus Scripts](#corpus-scripts)
- [Diagnostics](#diagnostics)
- [Default Ports](#default-ports)
- [Torch Compile](#torch-compile)
- [Validation](#validation)

---

## Install

```bash
cd projects/environment
uv sync
```

---

## Responsibilities

- Batched `sdfdag` stepping against compiled `.gmdag` assets on the GPU
- Publishing `DistanceMatrix v2` observations in `(n_envs, Az, El)` shape
- Corpus preparation: discovery, compilation, qualification, and validation of `.gmdag` assets
- Runtime preflight checks and throughput benchmarking

---

## CLI Reference

**Base command:** `uv run --project projects\environment navi-environment <command>`  
**Shortcut:** `uv run --project projects\environment environment` → `serve`

### `serve` — Start Environment Server

ZMQ-based environment server for the 3-process inference stack.

```powershell
uv run navi-environment serve [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pub` | string | `tcp://localhost:5559` | PUB socket address (observation broadcast) |
| `--rep` | string | `tcp://localhost:5560` | REP socket address (step request/response) |
| `--action-sub` | string | `tcp://localhost:5557` | SUB socket for actor actions |
| `--mode` | string | `step` | Service mode (`step` / `async`) |
| `--gmdag-file` | string | | `.gmdag` scene file to load |
| `--actors` | int | `1` | Number of parallel actors |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |
| `--max-distance` | float | `30.0` | Maximum ray distance (meters) |
| `--sdf-max-steps` | int | `256` | Sphere-tracing max iterations |

### `compile-gmdag` — Compile Mesh to `.gmdag`

```powershell
uv run navi-environment compile-gmdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--source` | string | (required) | Input mesh file (`.glb`/`.obj`/`.ply`/`.stl`) |
| `--output` | string | (required) | Output `.gmdag` file path |
| `--resolution` | int | `512` | Target cubic voxel resolution |
| `--repair` | flag | | Apply mesh repair (scene-graph transforms, merge, fill holes) |

### `prepare-corpus` — Full Corpus Preparation

Discover source scenes, compile missing `.gmdag` assets, emit manifest.

```powershell
uv run navi-environment prepare-corpus [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scene` | string | `""` | Single scene name filter |
| `--manifest` | string | `""` | Existing manifest to update |
| `--corpus-root` | string | `artifacts/gmdag/corpus` | Live corpus root |
| `--gmdag-root` | string | `artifacts/gmdag` | Parent `.gmdag` directory |
| `--resolution` | int | `512` | Compile resolution |
| `--min-scene-bytes` | int | `1000` | Minimum scene file size |
| `--force-recompile` | flag | | Recompile all assets |
| `--json` | flag | | Machine-readable JSON output |

### `check-sdfdag` — Validate Runtime + Assets

```powershell
uv run navi-environment check-sdfdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | Validate single `.gmdag` file |
| `--gmdag-root` | string | | Validate all files in directory |
| `--manifest` | string | | Validate against manifest |
| `--expected-resolution` | int | `512` | Expected compile resolution |
| `--json` | flag | | Machine-readable JSON output |

### `qualify-gmdag` — Observation Quality Gate

Probe spawn candidates via real CUDA ray casting to validate observation quality.
Scenes that fail (0 viable spawn candidates or best starvation ≥ 70%) are rejected.

```powershell
uv run navi-environment qualify-gmdag --gmdag-file <path>
```

### `bench-sdfdag` — Benchmark Compiled Assets

Canonical environment-layer throughput benchmark for the SDF/DAG path.

```powershell
uv run navi-environment bench-sdfdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | (required) | `.gmdag` file to benchmark |
| `--actors` | int | `4` | Parallel actor count |
| `--steps` | int | `200` | Benchmark steps |
| `--warmup-steps` | int | `25` | Warmup steps (excluded from timing) |
| `--repeats` | int | `1` | Benchmark repetitions |
| `--azimuth-bins` | int | `256` | Azimuth resolution |
| `--elevation-bins` | int | `48` | Elevation resolution |
| `--max-distance` | float | `30.0` | Max ray distance |
| `--sdf-max-steps` | int | `256` | Sphere-tracing iterations |
| `--torch-compile` | flag | | Enable `torch.compile` (`sm_70+`) |
| `--json` | flag | | Machine-readable JSON output |

### Windows Wrapper

```powershell
.\scripts\run-environment.ps1
.\scripts\run-environment.ps1 -Mode async
.\scripts\run-environment.ps1 -- --gmdag-file .\scene.gmdag
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Mode` | string | `step` | Service mode (`step` / `async`) |
| `-AzimuthBins` | int | `256` | Observation azimuth resolution |
| `-ElevationBins` | int | `48` | Observation elevation resolution |

---

## Corpus Preparation

The training corpus is a collection of compiled `.gmdag` scene files. Multiple
data sources are supported. All compilation defaults to resolution 512.

### Full Corpus Refresh (HuggingFace Datasets)

Transactional pipeline: download → compile → validate → promote to live corpus.
Staged transaction: stale data is only replaced after successful rebuild.

```powershell
.\scripts\refresh-scene-corpus.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Datasets` | `"ai-habitat/ReplicaCAD_dataset,ai-habitat/ReplicaCAD_baked_lighting,..."` | Comma-separated HuggingFace dataset IDs |
| `-ScenesPerDataset` | `10` | Max scenes to download per dataset |
| `-Resolution` | `512` | `.gmdag` compile resolution |
| `-ForceRecompile` | (switch) | Recompile even if `.gmdag` exists |
| `-KeepScratch` | (switch) | Keep intermediate GLB downloads |
| `-IncludeQuake3` | (switch) | Include Quake 3 maps in the refresh |

```powershell
# Examples:
.\scripts\refresh-scene-corpus.ps1                              # Full default refresh
.\scripts\refresh-scene-corpus.ps1 -ScenesPerDataset 5          # Smaller subset
.\scripts\refresh-scene-corpus.ps1 -ForceRecompile              # Force recompile all
.\scripts\refresh-scene-corpus.ps1 -IncludeQuake3               # Include Q3 maps
```

**Outputs:** `artifacts/gmdag/corpus/<dataset>/*.gmdag` + `artifacts/gmdag/corpus/gmdag_manifest.json`

### Habitat Test Scenes (Bootstrap)

Quick download of 3 test scenes + ReplicaCAD stages for initial setup.

```powershell
.\scripts\download-habitat-data.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-DataDir` | `data/scenes` | Download target directory |
| `-Datasets` | `"test_scenes,replicacad"` | Which datasets to fetch |
| `-PreserveExisting` | (switch) | Skip if files already exist |

### ReplicaCAD Expansion (Incremental)

Add new ReplicaCAD baked-lighting scenes without disturbing the existing corpus.

```powershell
.\scripts\expand-replicacad-corpus.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-ScenesLimit` | `0` (all) | Max new scenes to download |
| `-Resolution` | `512` | Compile resolution |

### Quake 3 Arena Maps

Download community Quake 3 maps from [lvlworld.com](https://lvlworld.com),
extract BSP geometry, and compile to `.gmdag`.

```powershell
.\scripts\download-quake3-maps.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-MapFilter` | `""` (all) | Comma-separated map names to download |
| `-Sources` | `"lvlworld"` | Source filter (`lvlworld`, `openarena`) |
| `-Resolution` | `512` | Compile resolution |
| `-Tessellation` | `4` | BSP bezier patch tessellation level |
| `-ForceRecompile` | (switch) | Overwrite existing `.gmdag` files |
| `-KeepIntermediate` | (switch) | Keep downloaded PK3 and intermediate OBJ |

```powershell
# Examples:
.\scripts\download-quake3-maps.ps1                                             # All manifest maps
.\scripts\download-quake3-maps.ps1 -MapFilter "padshop,aerowalk,simpsons_map"  # Specific maps
.\scripts\download-quake3-maps.ps1 -ForceRecompile                             # Rebuild all
```

Map manifest: `data/quake3/quake3_map_manifest.json` (18 lvlworld maps defined).

**Current compiled Quake 3 maps (10):**

| Map | Author | Style | GMDAG Size |
|-----|--------|-------|------------|
| padshop | ENTE | Giant-scale furniture | 6.1 MB |
| japanese_castles | g1zm0 | Multi-layered architecture | 7.2 MB |
| substation11 | g1zm0 | Complex technical indoor | 3.9 MB |
| edge_of_forever | Sock | Dense interconnected (highest-rated) | 1.0 MB |
| rustgrad | Hipshot | Industrial machinery | 5.1 MB |
| simpsons_map | Maggu | Residential rooms & furniture | 5.1 MB |
| unholy_sanctuary | Martinus | Gothic corridors, vertical | 26.4 MB |
| chronophagia | Obsessed | Artistic sealed corridors | 5.3 MB |
| padkitchen | ENTE | Giant-scale kitchen | 7.1 MB |
| aerowalk | the Hubster | Tight tournament | 3.9 MB |

### Single Asset Compilation

```powershell
# Compile one mesh file to .gmdag
uv run navi-environment compile-gmdag `
    --source path/to/scene.glb --output path/to/scene.gmdag --resolution 512

# With mesh repair (for scenes with scene-graph scale transforms)
uv run navi-environment compile-gmdag `
    --source path/to/scene.glb --output path/to/scene.gmdag --resolution 512 --repair

# Validate a compiled asset
uv run navi-environment check-sdfdag --gmdag-file path/to/scene.gmdag

# Benchmark a compiled asset
uv run navi-environment bench-sdfdag --gmdag-file path/to/scene.gmdag --actors 4 --steps 200
```

---

## Corpus Scripts

| Script | Purpose | Wraps |
|--------|---------|-------|
| `refresh-scene-corpus.ps1` | Full transactional corpus refresh | HuggingFace API + `compile-gmdag` |
| `download-habitat-data.ps1` | Bootstrap Habitat test scenes | HuggingFace downloads |
| `expand-replicacad-corpus.ps1` | Incremental ReplicaCAD growth | HuggingFace + `compile-gmdag` |
| `download-quake3-maps.ps1` | Quake 3 map download + compile | `bsp-to-obj` + `compile-gmdag` |

---

## Diagnostics

### Corpus Diagnostics

```powershell
# Shallow header probe (fast — reads .gmdag headers only)
python scripts\diagnose_gmdag_corpus.py

# Deep DAG structural analysis (walks up to 2M nodes per scene)
python scripts\diagnose_gmdag_deep.py
```

Flags: non-finite values, extreme scales, degenerate scenes (all-void, all-surface),
low compression, coarse/fine voxels, far centers.

### Asset Validation

```powershell
# Validate specific asset
uv run navi-environment check-sdfdag --gmdag-file path/to/scene.gmdag

# Validate entire corpus root
uv run navi-environment check-sdfdag --gmdag-root .\artifacts\gmdag\corpus

# JSON output for automation
uv run navi-environment check-sdfdag --json --gmdag-file path/to/scene.gmdag
```

### Training Log Summarization

```powershell
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1 -OutputJson summary.json
```

Extracts per-step metrics (`sps`, `fwd_ms`, `env_ms`, `opt_ms`) and computes
mean/min/max statistics.

---

## Default Ports

| Port | Topic | Transport |
|------|-------|-----------|
| `5559` | `distance_matrix_v2` | PUB |
| `5560` | `step_request_v2` / `step_result_v2` | REP |

---

## Torch Compile

Environment hot-path fusion is enabled by default (`NAVI_SDFDAG_TORCH_COMPILE=1`).
On GPUs where the inductor/triton stack cannot compile (e.g. `sm_61`), the
runtime logs a warning and continues on the eager tensor path. Use
`--no-torch-compile` or `NAVI_SDFDAG_TORCH_COMPILE=0` for A/B attribution runs.

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
