# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first, message-driven reinforcement learning stack for
autonomous agentic navigation. The architecture fundamentally decouples
geometric simulation from temporal cognition and asynchronous observability.

```text
┌─────────────────────┐    ZMQ PUB/SUB     ┌──────────────────────┐
│  Simulation Layer   │ ──────────────────▶ │    Brain Layer       │
│  (environment)  │ ◀── REQ/REP ────── │    (actor)           │
│  Pluggable backends │                     │  Sacred CNN+Mamba2   │
│  Voxel · Habitat ·  │                     │  +Memory+PPO engine  │
│  Mesh               │                     └──────────┬───────────┘
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
- **Optional:** `habitat-sim` for Habitat backend, `mamba-ssm` for Mamba2
  temporal core, `faiss-cpu` for fast episodic memory KNN

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
uv run navi-actor run --sub tcp://localhost:5559 --pub tcp://*:5557 \
    --mode step --step-endpoint tcp://localhost:5560

# Terminal 3 — Gallery Layer (optional — passive observer)
cd projects/auditor
uv sync
uv run navi-auditor record --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr
```

### One-Command Launch (Windows PowerShell)

```powershell
# Default voxel backend
./scripts/run-ghost-stack.ps1

# Habitat backend
./scripts/run-ghost-stack.ps1 -Backend habitat -HabitatScene /path/to/scene.glb

# With a pre-trained checkpoint
./scripts/run-ghost-stack.ps1 -ActorPolicyCheckpoint "checkpoints/policy_step_10000.pt"
```

### Compile Mesh Assets

Convert `.ply`, `.obj`, or `.stl` meshes into the canonical sparse Zarr format:

```bash
cd projects/environment
uv run navi-environment compile-world \
    --source ../../data/scenes/world.ply --output ../../data/scenes/world.zarr
```

## Training

### Online PPO Training (single scene)

```bash
# Start Environment in Terminal 1 (see Quick Start above), then:
cd projects/actor
uv run navi-actor train-ppo --sub tcp://localhost:5559 \
    --step-endpoint tcp://localhost:5560 --steps 10000 \
    --checkpoint-every 1000 --checkpoint-dir artifacts/checkpoints
```

### Sequential Multi-Scene Training (Habitat)

```powershell
# Downloads 40+ HSSD stages, then trains sequentially with knowledge accumulation
./scripts/download-habitat-data.ps1
./scripts/train-habitat-sequential.ps1
```

### All-Night Continuous Training (Optimized)

Run the fully-optimized continuous training engine that cycles through 48 scenes with Ghost-Matrix Persistence:

```powershell
./scripts/train-all-night.ps1
```

### Live Dashboard (standalone — decoupled via ZMQ PUB/SUB)

Monitor training progress, real-time depth views, and PPO curves:

```powershell
./scripts/run-dashboard.ps1
```

The dashboard runs independently from training. Connect it to any running
Environment or Actor to observe live spherical depth views and PPO metrics.

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
| [docs/SIMULATION.md](docs/SIMULATION.md) | Simulation layer, backends, raycasting, world generators |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Canonical wire format — models, serialization, ZMQ topics |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Theoretical performance baselines & throughput targets (roadmap) |
| [AGENTS.md](AGENTS.md) | Implementation policy, non-negotiables, repository structure |

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run-ghost-stack.ps1`](scripts/run-ghost-stack.ps1) | One-command full stack launch |
| [`scripts/train-habitat-sequential.ps1`](scripts/train-habitat-sequential.ps1) | Sequential multi-scene training |
| [`scripts/download-habitat-data.ps1`](scripts/download-habitat-data.ps1) | Download HSSD/ReplicaCAD scenes |
| [`scripts/generate_sample_scene.py`](scripts/generate_sample_scene.py) | Generate sample PointNav episodes |
| [`scripts/bench_raycast.py`](scripts/bench_raycast.py) | Raycast engine benchmark |
