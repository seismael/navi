# navi-auditor

Gallery Layer for Ghost-Matrix runtime.

**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Overview

- **Passive only** — never gates simulation throughput or modifies training.
- Records `distance_matrix_v2`, `action_v2`, and `telemetry_event_v2` streams.
- Replays recorded streams via ZMQ PUB.
- Provides a live PyQtGraph dashboard with one selected actor perspective view
  and actor selector enabled by default.
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

# Passive runtime-backed dataset audit against the canonical environment CLI
uv run navi-auditor dataset-audit --json

# Headless passive attach proof for live training or replay streams
uv run navi-auditor dashboard-attach-check --actor-sub tcp://localhost:5557 --json

# One-shot live frame capture for raw-vs-projected inspection
uv run navi-auditor dashboard-capture-frame --actor-sub tcp://localhost:5557 --json

# Shortcut command (equivalent to: navi-auditor dashboard)
uv run dashboard

# Live dashboard with actor selector enabled by default
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557
```

## Windows Wrapper Script

```powershell
# From repository root
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
./scripts/run-dashboard.ps1
```

## Dashboard

The `GhostMatrixDashboard` is a standalone PyQtGraph application that connects
via ZMQ PUB/SUB and can run independently from training. It displays:

- One selected live actor perspective view with corrected spherical-to-rectilinear
  projection geometry and the selector enabled by default
- The live HUD shows `CTR ...m` for the centre ray and `RANGE ...m` for the
  observation horizon so the centre label is not mistaken for the global limit
- WAITING / OBSERVER / TRAINING / INFERENCE mode indicator
- Compact same-line status telemetry: stall time, SPS, reward EMA, episode
  count, latest step, optimizer wall-time, and zero-wait ratio

Use `dashboard-capture-frame` when you need the raw spherical `DistanceMatrix`
offline for diagnostics; that debug surface is not part of the normal live
observer layout.

The selector stays available by default so operators can switch the
displayed actor when diagnosing specific env IDs.

The dashboard displays compact live status metrics for fast observability while
full histories remain available through service logs, telemetry streams, and
recorder outputs.

## Dataset Audit

`dataset-audit` is the first runtime-backed dataset QA surface in the auditor
layer.

It remains passive and observer-side:

- runs `check-sdfdag --json` and optionally `bench-sdfdag --json` through the
  environment CLI
- avoids importing environment service packages into the auditor runtime
- emits one combined JSON summary suitable for scripted verification
- defaults to the promoted corpus when `--gmdag-file` is omitted

## Passive Attach Proof

`dashboard-attach-check` is the first headless proof that the dashboard-visible
actor stream is actually observable.

It remains passive and observer-side:

- subscribes to the actor-visible topics the dashboard consumes
- works against the live actor PUB stream or a replay PUB stream
- emits one JSON summary suitable for qualification scripts

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
