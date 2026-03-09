# navi-auditor

Gallery Layer for Ghost-Matrix runtime.

**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Overview

- **Passive only** — never gates simulation throughput or modifies training.
- Records `distance_matrix_v2`, `action_v2`, and `telemetry_event_v2` streams.
- Replays recorded streams via ZMQ PUB.
- Provides a live PyQtGraph dashboard with a single selected actor depth view
  (actor 0 by default).
- Dashboard metrics are intentionally not rendered in UI; metrics remain
  available via logs, telemetry events, and recorder artifacts.
- Visualization types (RGB frames, camera images) are handled here in the Gallery
  Layer — they are never part of the canonical wire contracts.

## Components

| Component | Module | Description |
|-----------|--------|-------------|
| `StreamEngine` | `stream_engine.py` | Multi-topic ZMQ subscriber with ring-buffer state |
| `Recorder` | `recorder.py` | Records live streams to Zarr archives |
| `Rewinder` | `rewinder.py` | Replays recorded Zarr sessions via ZMQ PUB |
| `MatrixViewer` | `matrix_viewer.py` | OpenCV depth/semantic viewer with teleop |
| `GhostMatrixDashboard` | `dashboard/app.py` | PyQtGraph live dashboard |
| `ZarrBackend` | `storage/zarr_backend.py` | Zarr v3 storage backend |

## Usage

```bash
cd projects/auditor
uv sync

# Record a session
uv run navi-auditor record --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr

# Replay a session
uv run navi-auditor replay --input session.zarr --pub tcp://*:5558

# Live matrix dashboard (standalone — runs independently from training)
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559

# Dashboard with step controls (WASD/arrows, Tab toggle)
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560

# Shortcut command (equivalent to: navi-auditor dashboard)
uv run dashboard

# Enable dynamic actor selector when needed for diagnostics
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --enable-actor-selector
```

## Windows Wrapper Script

```powershell
# From repository root
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
./scripts/run-dashboard.ps1 --enable-actor-selector
```

## Dashboard

The `GhostMatrixDashboard` is a standalone PyQtGraph application that connects
via ZMQ PUB/SUB and can run independently from training. It displays:

- One selected live actor depth view (actor 0 by default)
- WAITING / OBSERVER / TRAINING / INFERENCE mode indicator
- Compact same-line status telemetry: stall time, SPS, reward EMA, episode
  count, latest step, optimizer wall-time, and zero-wait ratio

Optional selector mode can be enabled to switch the displayed actor when
diagnosing specific env IDs.

The dashboard displays compact live status metrics for fast observability while
full histories remain available through service logs, telemetry streams, and
recorder outputs.

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
