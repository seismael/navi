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
