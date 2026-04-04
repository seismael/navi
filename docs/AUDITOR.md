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
| `StreamEngine` | split-socket ZMQ ingestion with CONFLATE observation delivery and actor 0 state |
| `GhostMatrixDashboard` | primary PyQtGraph operator UI |
| `LiveDashboard` | legacy OpenCV live matrix dashboard surface |
| `Recorder` | persistent capture of stream data |
| `Rewinder` | replay via PUB for review and diagnostics |
| `DemonstrationRecorder` | (observation, action) pair capture during manual navigation |
| `renderers.py` | pure NumPy rendering utilities |

## 3. Current Operator UI

The primary operator UI is `GhostMatrixDashboard`.
Current implementation characteristics include:

- PyQtGraph and PyQt-based desktop UI
- hardcoded actor `0` view for maximum throughput
- actor count and active scene name display in status bar
- observation age indicator (`Obs=XXms`) in the status metrics line
- passive status-line telemetry
- forward-FOV extraction from the spherical observation convention
- conditional rendering: the actor panel redraws only when a genuinely new
  observation arrives, not on every UI tick
- split-socket ZMQ ingestion: a dedicated `CONFLATE` socket for observations
  ensures the dashboard always displays the latest published frame regardless
  of transient UI pauses

The dashboard is intentionally visual-only by default and should remain
non-blocking with respect to the trainer.

The auditor is not an authority over the observation contract. It receives the
published sphere and may reshape it locally for display, but it must never
require environment or actor changes for view convenience.

## 4. Rendering Model

The current renderer stack is more concrete than the imported web-first design.
It uses:

- pure NumPy renderers for portability and testability
- OpenCV utilities for resize and overlays
- direct log-encoded depth visualization: the environment publishes depth as
  `log1p(d)/log1p(100)` in `[0,1]`, and the renderers work on these values
  directly, preserving logarithmic near-field detail (~52% of the color range
  maps to 0-10 m)
- percentile-based dynamic contrast stretch (p2-p98) adapts per-scene so both
  tight indoor spaces and open outdoor vistas use the full palette
- forward-centered spherical slicing for actor view panels
- orientation guides and semantic coloring for diagnostic readability

Canonical primary actor-panel rule:

- the main dashboard actor view is a direct centered `180` degree half-sphere heatmap
- for the canonical `256x48` observation contract this means an exact forward `128x48` slice before viewport scaling
- the panel must not add its own `30m` range cap, `CTR`, or `HFOV` meter overlays
- LEFT and RIGHT labels belong on the horizontal midline because the center column is the actor heading
- the observer palette runs yellow-green (near) → teal → blue → gray (far),
  keeping ordinary structure readable and avoiding saturated rainbow extremes
- these are observer-side rendering rules only; they do not change the environment, actor, or contract layers

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
- the actor telemetry PUB socket uses `SNDHWM=50` so unsent frames are dropped
  rather than accumulating unbounded memory on slow or absent consumers
- actor-stream-only passive operation must remain supported during training
- any crop, roll, half-sphere extraction, or display-only reinterpretation must happen in the auditor after receipt of the published contract

Dashboard heartbeat republishing from the trainer is allowed only as a coarse,
diagnostic convenience during optimizer windows.

Current socket architecture:

- a dedicated `zmq.CONFLATE` observation socket subscribes only to
  `distance_matrix_v2` on the actor PUB address; `CONFLATE` keeps at most one
  buffered message, automatically replacing old observations with new ones so
  the dashboard always renders the latest frame even after transient UI pauses,
  GC stalls, or window operations
- a separate telemetry socket subscribes to `action_v2` and
  `telemetry_event_v2` with `RCVHWM=50`; telemetry events are small and
  ordered, so a conventional bounded queue is appropriate
- an optional Environment PUB socket (`matrix_sub`) handles service-mode
  observation delivery with `RCVHWM=200` and is not connected in passive
  training mode
- the `poll()` loop drains the `CONFLATE` observation socket first (at most one
  message per tick), then drains all available telemetry; rendering occurs only
  when a new observation was received, saving CPU for queue draining and status
  updates

Current canonical default:

- actor `0` receives a passive live observation stream at a low but visibly live
  cadence of about `10 Hz`
- widening the live observation stream to every actor is reserved for explicit diagnostic fan-out, not the throughput-safe default
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
They must not widen the wire contract or force extra training-time host work.

## 9. Optional Web Surfaces

Browser-facing dashboards or dataset-QA pages are allowed only as observer-side
proxy layers.

Rules:

- the canonical observer ingestion surface remains ZMQ-based
- any WebSocket or HTTP streaming layer sits behind an observer-side proxy
- no browser transport may be treated as part of the training hot path
- web-facing surfaces must not widen the canonical wire contract or become a
  required dependency for production training

## 10. Demonstration Recording

**Module:** `demonstration_recorder.py`

The `DemonstrationRecorder` captures `(observation, action)` pairs during manual
dashboard navigation for behavioral cloning pre-training.  It is a passive data
collection component that does not modify environment contracts or training
semantics.

### 10.1 Capture Format

Observations are stored as `(3, Az, El)` float32 arrays matching the actor
tensor contract:

| Channel | Content | Range |
|---------|---------|-------|
| 0 | depth (log-normalised) | `[0, 1]` |
| 1 | semantic class ID | `[0, N)` |
| 2 | ray validity mask | `{0.0, 1.0}` |

Actions are stored as `(4,)` float32 in normalised `[-1, 1]` policy space:
`[forward, vertical, lateral, yaw]`.  Raw m/s and rad/s velocities are divided
by the drone kinematic limits.

### 10.2 Recording Lifecycle

When the dashboard is launched with `--record`, recording starts automatically.
Each teleop step that produces movement captures one pair.  The **B** key
toggles pause/resume.  Data is auto-saved as a compressed `.npz` archive under
`artifacts/demonstrations/` when the dashboard is closed.

### 10.3 Integration with BC Training

The saved `.npz` files are consumed by `BehavioralCloningTrainer` in the actor
project (`navi-actor bc-pretrain`).  The auditor does not depend on torch or the
actor package — it only writes numpy archives.

### 10.4 Passivity Guarantee

The recorder:

- does not modify `DistanceMatrix` or `Action` contracts
- does not add ZMQ streams or synchronisation barriers
- does not affect environment stepping cadence
- captures data strictly from the existing teleop path

## 11. Non-Goals

The auditor layer should not:

- become a mandatory service in the production trainer loop
- own environment stepping policy
- define training-time synchronization barriers
- replace benchmark and verification surfaces with visual inspection alone

## 12. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/DATAFLOW.md`
- `docs/SIMULATION.md`
- `docs/CONTRACTS.md`
