# AUDITOR.md - Passive Dashboard, Recorder, And Replay Architecture

## 1. Executive Summary

The auditor layer is the Gallery and observability surface of Navi. Its job is
not to participate in training. Its job is to observe, render, record, and
replay without regulating the core runtime.

The imported dashboard document had the right architectural instinct: passive
observer first. Navi now documents that idea against the current codebase.

## 2. Current Components

| Component | Current Role |
| --- | --- |
| `StreamEngine` | multi-stream ZMQ ingestion and selected-actor state |
| `GhostMatrixDashboard` | primary PyQtGraph operator UI |
| `LiveDashboard` | legacy OpenCV live matrix dashboard surface |
| `Recorder` | persistent capture of stream data |
| `Rewinder` | replay via PUB for review and diagnostics |
| `renderers.py` | pure NumPy rendering utilities |

## 3. Current Operator UI

The primary operator UI is `GhostMatrixDashboard`.
Current implementation characteristics include:

- PyQtGraph and PyQt-based desktop UI
- selected-actor view with actor `0` as the default throughput-safe choice
- optional actor selector for diagnostics
- passive status-line telemetry
- forward-FOV extraction from the spherical observation convention
- capped ZMQ ingestion per tick for UI responsiveness

The dashboard is intentionally visual-only by default and should remain
non-blocking with respect to the trainer.

## 4. Rendering Model

The current renderer stack is more concrete than the imported web-first design.
It uses:

- pure NumPy renderers for portability and testability
- OpenCV utilities for resize and overlays
- Viridis and Turbo-style depth colormaps
- forward-centered spherical slicing for actor view panels
- orientation guides and semantic coloring for diagnostic readability

This matters because rendering logic remains testable without requiring the full
UI layer.

## 5. Mode Detection And Status

The dashboard uses actor-side telemetry and state to infer mode:

- `TRAINING` when reward and PPO-related signals are present
- `INFERENCE` when live features exist without training progression
- `OBSERVER` when streams are present but actor training/inference signals are absent
- `WAITING` when required streams are missing

This matches the repo policy that observer tooling must survive partial system
availability.

## 6. Throughput Rules

The auditor layer must obey strict throughput rules:

- it must never become a required dependency for canonical training
- it may drop frames rather than backpressure producers
- its ingestion must be capped per UI tick
- actor-stream-only passive operation must remain supported during training

Dashboard heartbeat republishing from the trainer is allowed only as a coarse,
diagnostic convenience during optimizer windows.

Current canonical default:

- actor `0` receives a passive live observation stream at a low but visibly live
  cadence of about `10 Hz`
- optimizer-window heartbeats reuse that same cadence so the dashboard does not
  appear frozen between rollout and update phases
- the stream remains droppable and must never backpressure the trainer

## 7. Recorder And Replay Surfaces

The recorder and rewinder remain architecturally important because they decouple
inspection from live training.

### 7.1 Recorder

The recorder subscribes to stream traffic and persists it for later analysis.
Its existence means live diagnostics do not need to keep every visualization
consumer attached in real time.

### 7.2 Rewinder

The rewinder republishes stored data via ZMQ so dashboards and tools can consume
historical sessions as if they were live.

This is important for architecture because it keeps postmortem analysis out of
the training hot path.

## 8. Imported Ideas Adopted As Roadmap

The imported dashboard design contained several strong ideas that are not yet the
current implementation but are worth preserving as explicit design direction.

### 8.1 Dataset Auditor Through The Real Runtime

The first runtime-backed dataset-auditor surface is now a passive CLI
orchestrator:

- `uv run navi-auditor dataset-audit --json`
- shells out at the CLI boundary to `navi-environment check-sdfdag --json`
  and `bench-sdfdag --json`
- validates promoted compiled assets through the same canonical runtime used by
  training without importing the environment service package into auditor code
- emits one merged parseable summary suitable for qualification artifacts

This is intentionally a first QA seam, not the full end-state visual dataset
auditor. Runtime-native pinhole or richer render-based QA can build on top of
this surface later without reopening geometry-export surrogates as the primary
proof path.

Current proof status:

- unit proof covers merged JSON success, preflight-failure short-circuit, and
  explicit asset override behavior
- integration smoke proof covers live `dataset-audit --json` execution against
  the promoted corpus when the compiled runtime is available

### 8.2 Passive Attach Proof

The first headless dashboard-attach proof surface is now:

- `uv run navi-auditor dashboard-attach-check --actor-sub tcp://localhost:5557 --json`
- subscribes passively to the same actor-visible topics the dashboard consumes
- proves observer-side attach readiness without opening a GUI or environment
  control socket
- works both against live training telemetry and replay PUB streams

This keeps Phase 5 qualification observer-only while still proving that the
dashboard surface can see canonical traffic when it exists.

### 8.3 Richer Spatial Diagnostics

Reasonable roadmap surfaces include:

- trajectory overlays
- information-foraging heatmaps
- kinematic actuation gauges
- critic-value or hidden-state strips
- hardware-utilization summaries

These should remain passive and optional.

## 9. Optional Web Surfaces

Browser-facing dashboards or dataset-QA pages are allowed only as observer-side
proxy layers.

Rules:

- the canonical observer ingestion surface remains ZMQ-based
- any WebSocket or HTTP streaming layer sits behind an observer-side proxy
- no browser transport may be treated as part of the training hot path
- web-facing surfaces must not widen the canonical wire contract or become a
  required dependency for production training

## 10. Non-Goals

The auditor layer should not:

- become a mandatory service in the production trainer loop
- own environment stepping policy
- define training-time synchronization barriers
- replace benchmark and verification surfaces with visual inspection alone

## 11. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/DATAFLOW.md`
- `docs/SIMULATION.md`
- `docs/CONTRACTS.md`
