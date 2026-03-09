# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first, message-driven reinforcement learning stack for
autonomous agentic navigation. The architecture fundamentally decouples
geometric simulation from temporal cognition and asynchronous observability.

```text
┌─────────────────────┐    ZMQ PUB/SUB     ┌──────────────────────┐
│  Simulation Layer   │ ──────────────────▶ │    Brain Layer       │
│  (environment)  │ ◀── REQ/REP ────── │    (actor)           │
│  Canonical SDF/DAG  │                     │  Sacred CNN+Mamba2   │
│  runtime + diag     │                     │  +Memory+PPO engine  │
│  references         │                     └──────────┬───────────┘
└─────────┬───────────┘                                │
          │ PUB                                   PUB  │
          ▼                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Gallery Layer (auditor)                       │
│        Passive: Record · Replay · Live Dashboard                 │
└──────────────────────────────────────────────────────────────────┘
```

The training engine is **sacred and immutable** — external simulators connect
through `DatasetAdapter`s that transform raw observations into the engine's
canonical `(1, Az, El)` DistanceMatrix format. The engine is never modified to
accommodate a new data source.

## Prerequisites

- **Python** 3.12+ (tested on 3.13 / 3.14)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Windows** or Linux (PowerShell scripts target Windows; Makefile targets
  run on both via `make`)
- **Optional:** `habitat-sim` for Habitat adapter diagnostics, `faiss-cpu` for
  fast episodic memory KNN

## Project Layout

| Project | Layer | Description |
|---------|-------|-------------|
| [`projects/contracts`](projects/contracts) | Shared | Wire-format models, serialization, ZMQ topics |
| [`projects/environment`](projects/environment) | Simulation | Pluggable backends, raycasting, world generators |
| [`projects/actor`](projects/actor) | Brain | Sacred cognitive engine, PPO training, policies |
| [`projects/auditor`](projects/auditor) | Gallery | Recording, replay, live PyQtGraph dashboard |

## Quick Start

### Three-Terminal Manual Launch

```bash
# Terminal 1 — Simulation Layer
cd projects/environment
uv sync
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Terminal 2 — Brain Layer
cd projects/actor
uv sync
uv run navi-actor serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Terminal 3 — Gallery Layer (optional — passive observer)
cd projects/auditor
uv sync
uv run navi-auditor record --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr
```

### Per-Project Shortcuts

```bash
# Environment service shortcut (same as: navi-environment serve)
cd projects/environment
uv sync
uv run environment

# Brain service shortcut (same as: navi-actor serve)
cd ../actor
uv sync
uv run brain

# Dashboard shortcut (same as: navi-auditor dashboard)
cd ../auditor
uv sync
uv run dashboard
```

### One-Command Launch (Windows PowerShell)

Repository launchers default to the pinned Python runtime in `.python-version`
(`3.12.11`). `run-ghost-stack.ps1` also accepts `-PythonVersion` when you need
to override it explicitly.

```powershell
# Inference stack with canonical SDF/DAG backend and a precompiled asset
./scripts/run-ghost-stack.ps1 -Backend sdfdag -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag

# Inference stack with canonical SDF/DAG backend and compile-on-demand from a mesh scene
./scripts/run-ghost-stack.ps1 -Backend sdfdag -HabitatScene ./data/scenes/sample_apartment.glb -AutoCompileGmDag

# With a pre-trained checkpoint on the canonical runtime
./scripts/run-ghost-stack.ps1 -Backend sdfdag -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag -ActorPolicyCheckpoint "checkpoints/policy_step_10000.pt"

# Training mode on the canonical SDF/DAG runtime
./scripts/run-ghost-stack.ps1 -Train -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag -TotalSteps 500000
```

### Wrapper Scripts (Windows)

```powershell
# Environment wrapper
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Brain wrapper
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Dashboard wrapper
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

### Compile Mesh Assets

Convert `.ply`, `.obj`, or `.stl` meshes into the canonical sparse Zarr format:

```bash
cd projects/environment
uv run navi-environment compile-world \
    --source ../../data/scenes/world.ply --output ../../data/scenes/world.zarr

# OBJ source
uv run navi-environment compile-world \
    --source ../../data/scenes/world.obj --source-format obj --output ../../data/scenes/world.zarr

# STL source
uv run navi-environment compile-world \
    --source ../../data/scenes/world.stl --source-format stl --output ../../data/scenes/world.zarr
```

## Training

### Canonical SDF/DAG Training

```powershell
./scripts/train.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag
```

### All-Night Continuous Training (Optimized)

Run the canonical long-duration training engine on a compiled `.gmdag` asset with Ghost-Matrix Persistence:

```powershell
./scripts/train-all-night.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag
```

`train` and `train-all-night` now expose one canonical throughput-tuned launch
surface on the compiled-path runtime. Override individual knobs only when doing
targeted experiments.

### Live Dashboard (standalone — decoupled via ZMQ PUB/SUB)

Monitor a single live actor view (actor 0 by default) in real time:

```powershell
./scripts/run-dashboard.ps1
```

The dashboard runs independently from training. Connect it to any running
Environment or Actor to observe the selected live actor depth view.
Use `--enable-actor-selector` if you need to switch actors interactively.
The status bar now shows compact live observability beside mode flags
(`TRAINING`, `WAITING`, etc.), including stream stall time, SPS, reward EMA,
episode count, latest step, optimizer wall-time, and zero-wait ratio.
Full metric history remains available through logs, telemetry events, and
recorder artifacts.
Default reporting/mode detection in training relies on low-volume telemetry
events to avoid rollout-loop stalls.

## Repository Commands

```bash
make sync-all        # Install dependencies in all sub-projects
make test-all        # Run pytest in all sub-projects
make lint-all        # ruff check + format check
make typecheck-all   # mypy --strict
make check-all       # lint + typecheck + test (CI gate)
make clean-all       # Remove .venv, caches
```

## Canonical Wire Topics

All inter-service communication uses v2 contracts over ZeroMQ:

| Topic | Direction |
|-------|-----------|
| `distance_matrix_v2` | Simulation → Brain, Gallery |
| `action_v2` | Brain → Simulation, Gallery |
| `step_request_v2` | Brain → Simulation (REQ/REP) |
| `step_result_v2` | Simulation → Brain (REQ/REP) |
| `telemetry_event_v2` | Any → Gallery |

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System layers, SDF theory, algorithmic design decisions |
| [docs/ACTOR.md](docs/ACTOR.md) | Cognitive engine specification (sacred, immutable) |
| [docs/TRAINING.md](docs/TRAINING.md) | Canonical overnight training, checkpointing, dashboard attach, and recovery flow |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Simulation layer, backends, raycasting, world generators |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Canonical wire format — models, serialization, ZMQ topics |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Theoretical performance baselines & throughput targets (roadmap) |
| [AGENTS.md](AGENTS.md) | Implementation policy, non-negotiables, repository structure |

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run-ghost-stack.ps1`](scripts/run-ghost-stack.ps1) | One-command full stack launch, including canonical `sdfdag` `.gmdag` flows |
| [`scripts/train.ps1`](scripts/train.ps1) | Canonical training on one compiled `.gmdag` asset |
| [`scripts/train-all-night.ps1`](scripts/train-all-night.ps1) | Canonical long-duration training on one compiled `.gmdag` asset |
| [`scripts/download-habitat-data.ps1`](scripts/download-habitat-data.ps1) | Download HSSD/ReplicaCAD scenes |
| [`scripts/generate_sample_scene.py`](scripts/generate_sample_scene.py) | Generate sample PointNav episodes |
| [`scripts/bench_raycast.py`](scripts/bench_raycast.py) | Raycast engine benchmark |
