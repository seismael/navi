# navi-auditor

Gallery Layer for Ghost-Matrix runtime.

**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Overview

- **Passive only** — never gates simulation throughput or modifies training.
- Records `distance_matrix_v2`, `action_v2`, and `telemetry_event_v2` streams.
- Replays recorded streams via ZMQ PUB.
- Provides a live PyQtGraph dashboard with spherical depth views, overhead map,
  and 10 PPO training metric plots.
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
```

## Dashboard

The `GhostMatrixDashboard` is a standalone PyQtGraph application that connects
via ZMQ PUB/SUB and can run independently from training. It displays:

- Spherical depth panorama (azimuth × elevation)
- Overhead minimap with heading indicator
- 10 live PPO training metric plots (policy loss, value loss, entropy, KL,
  clip fraction, RND loss, intrinsic reward, loop similarity, beta, reward EMA)

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
