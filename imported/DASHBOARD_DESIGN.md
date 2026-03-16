> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/SIMULATION.md`, `docs/TRAINING.md`, `docs/ARCHITECTURE.md`, and `AGENTS.md`.

# DASHBOARD_DESIGN.md — TopoNav Behavioral Observer Suite

## 1. Executive Summary

The TopoNav Behavioral Observer Suite (`toponav-dash`) is a strictly decoupled, asynchronous telemetry dashboard for monitoring and diagnosing high-throughput RL actors. It enforces a read-only architecture via ZeroMQ `PUB/SUB`, ensuring the $O(1)$ CUDA engine sustains >10,000 SPS without UI-induced bottlenecks.

---

## 2. Architecture Principles

| Principle | Enforcement |
|-----------|-------------|
| **Zero Coupling** | Dashboard subscribes to ZMQ PUB; never writes to the engine |
| **Flat Design** | No shadows, no gradients; solid colors, 0px border radius |
| **Canvas-First** | All high-frequency data rendered via HTML5 `<canvas>`. DOM reserved for structural framing |
| **Asymmetric Grid** | CSS Grid with dynamic area templates — stat cubes (small) + viewports (large) |

### Color Ontology

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg-base` | `#121212` | Body background |
| `--bg-surface` | `#1E1E1E` | Panel backgrounds |
| `--bg-elevated` | `#2D2D2D` | Hover/active states |
| `--border-dim` | `#333333` | Border separators |
| `--accent-blue` | `#007ACC` | Primary metrics, active nav |
| `--accent-green` | `#4CAF50` | Positive metrics |
| `--accent-red` | `#E53935` | Alerts, danger states |

---

## 3. View Hierarchy

```text
┌──────────────────────────────────────────────┐
│  App Container (100vw × 100vh, flex)         │
├──────────┬───────────────────────────────────┤
│          │  Contextual Top Bar (60px)        │
│ Sidebar  ├───────────────────────────────────┤
│ (250px)  │  Content Grid (auto-flow)         │
│          │  ┌────┬────┬────────────────┐     │
│ • Live   │  │ KPI│ KPI│   Good View    │     │
│ • Hist.  │  │Cube│Cube│ (Trajectory)   │     │
│ • Audit  │  ├────┴────┼────────────────┤     │
│ • HW     │  │Gauges   │ Foveated Vision│     │
│          │  └─────────┴────────────────┘     │
└──────────┴───────────────────────────────────┘
```

### Navigation Model
- **Left Sidebar** — Global navigation (Live Observer, Historical Data, Dataset Auditor, Hardware Telemetry)
- **Contextual Top Bar** — View-specific controls (run selector, epoch controls, dataset picker)
- **Content Grid** — Responsive CSS Grid with `auto-fit` columns (~200px base)

---

## 4. Modules

### 4.1. Live Foveated Vision (v1.0 — Implemented)

- **Data:** $256 \times 48$ `DistanceMatrix` streamed via WebSockets
- **Rendering:** HTML5 `<canvas>` with `image-rendering: pixelated`, hardware-accelerated upscaling
- **Colormap:** JavaScript HSL transformation (Google Turbo): close geometry (warm reds) → distant (cool blues) → void (dark charcoal)

### 4.2. Historical Training Comparison (v1.0 — Implemented)

- **Data:** `.jsonl` episodic summaries from `reports/`
- **Rendering:** Chart.js interactive time-series
- **Features:** Multi-select overlay (up to 10 runs), PPO loss, entropy, mean reward

---

### 4.3. Good View — Global Trajectory Overlay (v1.5)

- **Purpose:** Diagnose exploration efficiency, loop closures, pathfinding stagnation
- **Backend:** Extract a Z-axis slice of `.gmdag` → static 2D occupancy grid (PNG)
- **Frontend:** Actor $(X, Y)$ + Yaw heading streamed via WebSocket → persistent "snail trail" over the occupancy PNG

### 4.4. Kinematic Actuation Gauges (v1.5)

- **Purpose:** Monitor policy saturation and motor boundary clipping
- **Implementation:** Bidirectional horizontal bar gauges (linear vel $[-1, 1]$, angular vel $[-1, 1]$)
- **Alert State:** Bars flash accent-red if the neural network pegs output at absolute maximums for sustained durations (reward hacking indicator)

### 4.5. Cognitive Valuation Strip (v1.5)

- **Purpose:** Correlate visual stimuli with temporal neural network internal state
- **Implementation:** Scrolling sparkline beneath the foveated vision canvas
- **Metric:** Critic $V(s)$ scalar output
- **Diagnostic:** A sudden $V(s)$ drop should align with the actor encountering a dead-end or collision

---

### 4.6. Advanced Expansions (v2.0+)

| Module | Purpose | Backend | Frontend |
|--------|---------|---------|----------|
| **Information Foraging Heatmap** | Identify over/under-explored areas | Aggregate $(X,Y)$ over rolling 10K steps -> 2D density | Canvas heatmap overlay on Good View |
| **Hardware Utilization Telemetry** | Detect pipeline bottlenecks | `pynvml` GPU metrics + ZMQ latency in `TelemetryEvent` | Dashboard footer KPIs + threshold alerts |
| **Sim-to-Real Degradation Monitor** | Visualize simulation vs. noised data delta | Stream raw + noise-injected matrices | Side-by-side canvas or difference view |
| **SSM Hidden State Entropy** | Monitor Mamba-2 memory stability | Extract $L2$ norm / variance of hidden state | Secondary scrolling sparkline |

---

## 5. Dataset Auditor — Virtual Pinhole Camera

### 5.1. Problem

Web browsers render polygon meshes (WebGL), not 64-bit DAGs. Converting `.gmdag` back to `.obj` risks introducing conversion errors — defeating the QA audit.

### 5.2. Solution: Server-Side Raytracing

The dashboard renders the dataset through the **exact same C++ physics engine** the actor uses. A "Virtual Camera" on the FastAPI backend casts a dense $512 \times 512$ pinhole ray grid via `torch-sdf`, applies the Turbo colormap, and streams JPEG to the browser.

**Guarantee:** What you see on screen is mathematically identical to what CUDA holds in memory.

### 5.3. API Contract

```
GET /api/dataset/render?dataset_name=<name>&cam_x=<float>&cam_y=<float>&cam_z=<float>&pitch=<float>&yaw=<float>
→ Response: image/jpeg (512×512 rendered frame)
```

### 5.4. Observer Experience

1. Select `habitat_apartment_01.gmdag` from the dropdown
2. Backend renders via `torch-sdf` sphere tracing
3. WASD + mouse drag to fly through the mathematical `.gmdag` structure
4. Turbo colormap proves walls/floors/voids compiled at correct scale and orientation

### 5.5. Implementation Notes

- **Backend:** FastAPI `StreamingResponse` with `cv2.imencode('.jpg', ...)`
- **Frontend:** `setInterval(fetchFrame, 100)` for ~10 FPS fly-through
- **Pending:** `generate_pinhole_rays()` tensor math (intrinsic camera matrix for rectilinear ray generation)

---

## 6. Reference Implementation — CSS Grid

### Stat Cubes
Small KPI panels (`100px` height) — SPS, PPO Entropy, Mean Reward

### Large Panels
Multi-column spans (`grid-column: span 2/3`) for canvas viewports — Good View, Foveated Vision

### Layout State
- Sidebar toggle: `collapsed` CSS class → `width: 0px`
- Navigation: `data-target` attributes on nav items → contextual toolbar switching
- All HF rendering: `<canvas>` with `object-fit: contain`
