# navi-auditor

Gallery Layer for the Navi ecosystem. Provides passive observability through a
live PyQtGraph dashboard, Zarr stream recording, and session replay — without
gating simulation throughput or modifying training.

**Auditor specification:** [docs/AUDITOR.md](../../docs/AUDITOR.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Install

```bash
cd projects/auditor
uv sync
```

---

## Components

| Component | Module | Description |
|-----------|--------|-------------|
| `GhostMatrixDashboard` | `dashboard/app.py` | PyQtGraph live actor perspective + status HUD |
| `StreamEngine` | `stream_engine.py` | Multi-topic ZMQ subscriber with ring-buffer state |
| `Recorder` | `recorder.py` | Records live streams to Zarr archives |
| `Rewinder` | `rewinder.py` | Replays recorded Zarr sessions via ZMQ PUB |
| `ZarrBackend` | `storage/zarr_backend.py` | Zarr v3 storage backend |

---

## Commands

### Dashboard

```bash
# Live dashboard (standalone, runs independently from training)
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559

# Dashboard with actor selector and step controls
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559 \
    --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560

# Passive attach to running trainer (actor stream only)
uv run navi-auditor dashboard --actor-sub tcp://localhost:5557 --passive

# Shortcut
uv run dashboard
```

The dashboard displays:
- Actor 0 perspective view with spherical-to-rectilinear projection (180° half-sphere)
- WAITING / OBSERVER / TRAINING / INFERENCE mode indicator
- Compact status: stall time, SPS, reward EMA, episode count, optimizer wall-time

### Manual Exploration & Demonstration Recording

```bash
# Launch explorer with automatic demonstration recording
uv run explore --record
uv run explore --record --gmdag-file <scene.gmdag>

# Custom kinematic limits for action normalisation
uv run explore --record --drone-max-speed 5.0 --drone-climb-rate 2.0 --drone-yaw-rate 3.0
```

When `--record` is active:
- Recording starts automatically when the dashboard opens
- Navigate with WASD/arrow keys, Space/Shift for vertical, A/D for yaw
- **B** toggles pause/resume recording
- Demonstrations auto-save to `artifacts/demonstrations/` on close (ESC/Q)
- Status bar shows `MANUAL ● REC (N steps)` during active recording

Saved `.npz` files contain `(observation, action)` pairs compatible
with `navi-actor bc-pretrain` for behavioral cloning pre-training.

### Multi-Scene Exploration & Training

```powershell
# Step 1: Fly through scenes continuously (no training wait between scenes)
./scripts/run-explore-scenes.ps1
./scripts/run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
./scripts/run-explore-scenes.ps1 -MaxSteps 2000

# Step 2: Train on all accumulated demos when ready
./scripts/run-bc-pretrain.ps1

# Step 3: Fine-tune with RL
./scripts/train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt
```

### Recording & Replay

```bash
# Record a session
uv run navi-auditor record \
    --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr

# Replay a session
uv run navi-auditor replay --input session.zarr --pub tcp://*:5558
```

### Diagnostics

```bash
# Passive runtime-backed dataset audit
uv run navi-auditor dataset-audit --json

# Headless attach proof for qualification
uv run navi-auditor dashboard-attach-check \
    --actor-sub tcp://localhost:5557 --json

# One-shot frame capture for raw-vs-projected inspection
uv run navi-auditor dashboard-capture-frame \
    --actor-sub tcp://localhost:5557 --json
```

### Windows Wrapper

```powershell
./scripts/run-dashboard.ps1
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 \
    --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

---

## Design Principles

- **Passive only** — never gates simulation throughput or modifies training
- **Resilient** — handles missing ZMQ streams gracefully with WAITING state
- **Observer-side rendering** — all visualization transforms (slicing, palette,
  labels) stay in the auditor domain; environment/actor contracts are unchanged
- **Dashboard displays actor 0** by default; selector available for other envs

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
