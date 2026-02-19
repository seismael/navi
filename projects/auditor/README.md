# navi-auditor

Gallery Layer for Ghost-Matrix runtime.

## Overview

- Records `distance_matrix_v2`, `action_v2`, and telemetry streams.
- Replays recorded streams via ZMQ PUB.
- Provides a lightweight live distance-matrix dashboard (passive by default).

## Usage

```bash
cd projects/auditor
uv sync

# Record a session
uv run navi-auditor record --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr

# Replay a session
uv run navi-auditor replay --input session.zarr --pub tcp://*:5558

# Live matrix dashboard
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559

# Manual stepping controls are built in (Tab toggle, WASD/arrows)
uv run navi-auditor dashboard --matrix-sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560
```
