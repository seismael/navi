← [Navi Overview](../../README.md)

# navi-auditor

Gallery Layer for the Navi ecosystem. Provides passive observability through a
live PyQtGraph dashboard, Zarr stream recording, and session replay — without
gating simulation throughput or modifying training.

**Auditor specification:** [docs/AUDITOR.md](../../docs/AUDITOR.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Table of Contents

- [Install](#install)
- [Components](#components)
- [Design Principles](#design-principles)
- [Live Dashboard](#live-dashboard)
- [Interactive Exploration](#interactive-exploration)
- [Demonstration Recording](#demonstration-recording)
- [Recording & Replay](#recording--replay)
- [CLI Reference](#cli-reference)
- [Diagnostics](#diagnostics)
- [Scripts Reference](#scripts-reference)
- [Legacy 3-Process Inference](#legacy-3-process-inference)
- [Validation](#validation)

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

## Design Principles

- **Passive only** — never gates simulation throughput or modifies training
- **Resilient** — handles missing ZMQ streams gracefully with WAITING state
- **Observer-side rendering** — all visualization transforms (slicing, palette,
  labels) stay in the auditor domain; environment/actor contracts are unchanged
- **Dashboard displays actor 0** observations; shows active actor count in the status bar
- **Split-socket architecture** — a dedicated `zmq.CONFLATE` observation socket
  ensures the displayed frame is always the latest published, while a separate
  telemetry socket delivers ordered metrics and actions

---

## Live Dashboard

The dashboard is a passive observer. It subscribes to the actor PUB telemetry
stream and renders actor 0 observations at 5–10 Hz.

```powershell
# Attach to a running training session or inference stack
.\scripts\run-dashboard.ps1

# Direct CLI with explicit port
uv run --project projects\auditor navi-auditor dashboard `
    --actor-sub tcp://localhost:5557 --passive
```

**Dashboard behavior:**

- Always displays **actor 0** observations (no selector)
- Renders a direct centered 180° half-sphere slice from the canonical `256×48`
  spherical observation (exact `128×48` forward hemisphere before viewport scaling)
- Shows active actor count in the status bar
- Shows observation freshness (`Obs=XXms`) in the status metrics line
- Auto-detects mode: **TRAINING** / **INFERENCE** / **OBSERVER**
  - TRAINING: triggered by `actor.training.*` telemetry events
  - INFERENCE: triggered by `actor.inference.*` events
  - OBSERVER: default state when no actor telemetry is present
- Handles missing streams gracefully (shows WAITING state)
- During training: passive actor-only mode — does not open environment control
  paths or depend on environment PUB availability
- Renders only when a new observation arrives; UI ticks without fresh data
  update status metrics without wasting CPU on redundant rendering

---

## Interactive Exploration

Keyboard-controlled drone flight for scene inspection and demonstration recording.

```powershell
.\scripts\run-explore.ps1 -GmdagFile .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-GmdagFile` | (required) | Scene `.gmdag` file path |
| `-LinearSpeed` | `1.5` | Movement speed (m/s) |
| `-YawRate` | `1.5` | Rotation speed (rad/s) |

**Controls:**

| Key | Action |
|-----|--------|
| `W` / `S` | Forward / Backward |
| `A` / `D` | Strafe Left / Right |
| `Space` / `Shift` | Up / Down |
| `Q` / `E` | Yaw Left / Right |
| `ESC` | Close |

```powershell
# Direct CLI (with recording):
uv run --project projects\auditor explore `
    --gmdag-file .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag `
    --record --max-steps 2000
```

---

## Demonstration Recording

### Multi-Scene Navigation

Fly through corpus scenes continuously. Each scene auto-closes after `MaxSteps`
and the next one opens immediately. Demonstrations accumulate in `DemoDir`.

```powershell
.\scripts\run-explore-scenes.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-CorpusRoot` | `artifacts\gmdag\corpus` | Scene directory to discover |
| `-Scenes` | `@()` (discover all) | Specific scene file paths |
| `-DemoDir` | `artifacts\demonstrations` | Demonstration output directory |
| `-MaxSteps` | `1000` | Steps per scene before auto-close |
| `-LinearSpeed` | `1.5` | Flight speed (m/s) |
| `-YawRate` | `1.5` | Rotation speed (rad/s) |

```powershell
# Examples:
.\scripts\run-explore-scenes.ps1                                       # Full corpus
.\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
.\scripts\run-explore-scenes.ps1 -MaxSteps 2000                        # Longer sessions
```

**Per-scene flow:**

1. Explorer opens with auto-recording (WASD / arrow keys)
2. Dashboard auto-closes after `MaxSteps` steps (or press ESC/Q to skip)
3. Demo `.npz` is saved to `DemoDir`
4. Next scene opens immediately — no training wait
5. Repeat until all scenes are done, or Ctrl+C to stop

When `--record` is active:
- Recording starts automatically when the dashboard opens
- **B** toggles pause/resume recording
- Demonstrations auto-save to `artifacts/demonstrations/` on close (ESC/Q)
- Status bar shows `MANUAL ● REC (N steps)` during active recording

Saved `.npz` files contain `(observation, action)` pairs compatible
with `navi-actor bc-pretrain` for behavioral cloning pre-training.

### Training on Demonstrations

When you have enough demonstrations, train a BC checkpoint and fine-tune with RL.
See [projects/actor/README.md](../actor/README.md#behavioral-cloning) for the
full BC → RL workflow.

---

## Recording & Replay

### Session Recording

```bash
# Record live ZMQ streams to Zarr storage
uv run navi-auditor record \
    --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr
```

### Session Playback

```bash
# Replay a recorded session via ZMQ PUB
uv run navi-auditor replay --input session.zarr --pub tcp://*:5558
```

---

## CLI Reference

**Base command:** `uv run --project projects\auditor navi-auditor <command>`  
**Shortcuts:**
- `uv run --project projects\auditor dashboard` → `navi-auditor dashboard`
- `uv run --project projects\auditor explore` → `navi-auditor explore`

### `dashboard` — Live Passive Observation Dashboard

```powershell
uv run navi-auditor dashboard [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | `tcp://localhost:5557` | Actor PUB socket to subscribe |
| `--matrix-sub` | string | `""` | Environment PUB socket (optional) |
| `--step-endpoint` | string | `""` | Environment REP endpoint (optional) |
| `--passive` | flag | | Force passive mode (no env control) |
| `--hz` | float | `30.0` | Target frame rate |
| `--linear-speed` | float | | Manual control speed |
| `--yaw-rate` | float | | Manual control yaw rate |
| `--max-distance` | float | | Override max distance for colorization |
| `--scene` | string | `""` | Scene label for title bar |

### `explore` — Interactive Keyboard-Controlled Explorer

Spawns environment backend + dashboard with WASD controls. No training process required.

```powershell
uv run navi-auditor explore [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | `.gmdag` scene to explore |
| `--pub-address` | string | `tcp://localhost:5559` | Environment PUB address |
| `--rep-address` | string | `tcp://localhost:5560` | Environment REP address |
| `--hz` | float | `30.0` | Target frame rate |
| `--record` | flag | | Record demonstration to `.npz` |
| `--drone-max-speed` | float | `5.0` | Drone max speed (m/s) |
| `--azimuth-bins` | int | `256` | Azimuth resolution |
| `--elevation-bins` | int | `48` | Elevation resolution |
| `--max-steps` | int | | Auto-close after N steps |

### `record` — Session Recording

```powershell
uv run navi-auditor record [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sub` | string | | Comma-separated ZMQ SUB addresses |
| `--out` | string | | Output file path |

### `replay` — Session Playback

```powershell
uv run navi-auditor replay [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | string | | Recorded session file |
| `--pub` | string | | PUB socket to publish on |
| `--speed` | float | `1.0` | Playback speed multiplier |

### `dataset-audit` — Dataset Quality Assurance

Runtime-backed dataset validation: preflight checks + optional benchmark.

```powershell
uv run navi-auditor dataset-audit [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | `.gmdag` file to audit |
| `--expected-resolution` | int | `512` | Expected compile resolution |
| `--benchmark` | flag | | Run throughput benchmark |
| `--actors` | int | `1` | Actor count (benchmark mode) |
| `--steps` | int | `8` | Benchmark steps |
| `--warmup-steps` | int | `1` | Warmup steps |
| `--azimuth-bins` | int | `64` | Audit azimuth resolution |
| `--elevation-bins` | int | `16` | Audit elevation resolution |
| `--json` | flag | | Machine-readable JSON output |

### `dashboard-attach-check` — Headless Attach Proof

Verify passive dashboard can connect to actor stream (used by qualification).

```powershell
uv run navi-auditor dashboard-attach-check [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | | Actor PUB socket to test |
| `--timeout-seconds` | float | `15.0` | Connection timeout |
| `--json` | flag | | Machine-readable JSON output |

### `dashboard-capture-frame` — Capture Dashboard Frame

Capture one live dashboard frame + rendered diagnostics to file.

```powershell
uv run navi-auditor dashboard-capture-frame [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | | Actor PUB socket |
| `--actor-id` | int | `0` | Actor ID to capture |
| `--timeout-seconds` | float | `15.0` | Connection timeout |
| `--output-dir` | string | | Output directory for captured frames |
| `--max-distance` | float | | Override max distance |
| `--json` | flag | | Machine-readable JSON output |

---

## Diagnostics

```powershell
# Passive runtime-backed dataset audit
uv run navi-auditor dataset-audit --json

# Headless attach proof for qualification
uv run navi-auditor dashboard-attach-check --actor-sub tcp://localhost:5557 --json

# One-shot frame capture for raw-vs-projected inspection
uv run navi-auditor dashboard-capture-frame --actor-sub tcp://localhost:5557 --json
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `run-dashboard.ps1` | Launch passive observation dashboard |
| `run-explore.ps1` | Single-scene interactive explorer |
| `run-explore-scenes.ps1` | Multi-scene navigation + demo recording |

### `run-dashboard.ps1` — Passive Observation Dashboard

```powershell
.\scripts\run-dashboard.ps1
.\scripts\run-dashboard.ps1 -- --actor-sub tcp://localhost:5557
```

### `run-explore.ps1` — Single-Scene Interactive Explorer

```powershell
.\scripts\run-explore.ps1 -GmdagFile .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag
.\scripts\run-explore.ps1 -GmdagFile .\scene.gmdag -LinearSpeed 3.0
```

---

## Legacy 3-Process Inference

The legacy multi-process stack runs Environment, Actor, and Dashboard as 3
separate processes communicating over ZMQ. Prefer in-process inference (see
[projects/actor/README.md](../actor/README.md#inference)) for new work.

**Manual 3-process launch (for debugging):**

```powershell
# Terminal 1 — Environment server
.\scripts\run-environment.ps1 -- --mode step --gmdag-file .\scene.gmdag

# Terminal 2 — Actor policy server
.\scripts\run-brain.ps1 -- --mode step --temporal-core mamba2

# Terminal 3 — Dashboard
.\scripts\run-dashboard.ps1
```

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
