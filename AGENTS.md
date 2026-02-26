# AGENTS.md — Ghost-Matrix Implementation Blueprint

This file is the **sacred source of truth** for implementation policy and architectural standards. All agent actions must be validated against this blueprint before execution.

## 1) Scope & Sovereignty

Navi is a modular ecosystem of isolated projects. Each project is a sovereign entity with its own environment, purpose, and configuration.

### Isolated Projects:
- **`projects/contracts`**: Wire-format models and serialization (The source of truth for communication).
- **`projects/environment`**: Simulation Layer (headless stepping + sensing).
- **`projects/actor`**: Brain Layer (policy + training — sacred, immutable engine).
- **`projects/auditor`**: Gallery Layer (record/replay/visualization — passive only).

## 2) Professional Standards

### 2.1 Configuration Standard
- All projects must use `pydantic-settings` for configuration management.
- **Centralized Defaults:** Professional defaults are defined in a root `.env` file.
- **Global Discovery:** `BaseSettings` MUST be configured to search up the directory tree or use an absolute path to the root `.env`.
- **Robust Fallback:** Hard-coded defaults in `config.py` MUST match the `.env` standard (`5559`, `5560`, `5557`) to ensure functionality if `.env` is missing.
- **Network Defaults:**
  - `NAVI_ENV_PUB_ADDRESS=tcp://localhost:5559`
  - `NAVI_ENV_REP_ADDRESS=tcp://localhost:5560`
  - `NAVI_ACTOR_PUB_ADDRESS=tcp://localhost:5557`
- **Resolution Defaults:**
  - `NAVI_AZIMUTH_BINS=256`
  - `NAVI_ELEVATION_BINS=48` (High-fidelity navigation)

### 2.2 Simple Launch Commands
Each project must provide a dedicated `uv run` shortcut and a corresponding wrapper script in `scripts/`:
- **Dashboard:** `uv run dashboard` → `scripts/run-dashboard.ps1`
- **Environment:** `uv run environment` → `scripts/run-environment.ps1`
- **Brain (Actor):** `uv run brain` → `scripts/run-brain.ps1`

### 2.3 Mode Detection Standard
- The Dashboard MUST detect mode dynamically:
  - **TRAINING:** Triggered by `actor.training.*` telemetry events.
  - **INFERENCE:** Triggered by `actor.inference.*` events.
  - **OBSERVER:** Default state when no actor telemetry is present.
- Training mode MUST be detectable even when high-frequency per-step telemetry is disabled for performance.

### 2.4 Logging Architecture Standard
- **Unified Logging:** All projects MUST use the standardized `setup_logging` utility provided by `navi-contracts`.
- **Cyclic Logs:** File logging MUST be cyclic (RotatingFileHandler) with a strict cap to prevent disk explosion (e.g., max 1MB per file, max 10 backups).
- **Format:** Logs must use a professional, high-quality structure: `[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s`.
- **Enforcement:** Each project's CLI entry point (`cli.py`) MUST invoke this setup immediately upon startup.

### 2.5 Non-Negotiables
1. **Wire Contracts:** v2 only (`RobotPose`, `DistanceMatrix`, `Action`, etc.).
2. **Sacred Brain:** The cognitive pipeline (`RayViTEncoder` → `Mamba2` → `EpisodicMemory` → `ActorCriticHeads`) is **immutable**.
3. **No Stall Mandate:** High-throughput training is the primary success metric. Optimization must never stall rollout.
4. **Project Isolation:** No service imports another service package (except for recognized CLI-level integrations).
5. **Quality Gates:** `ruff`, `mypy --strict`, and `pytest` are mandatory for all changes.

## 3) Performance Mandates

### 3.1 Batched Raycasting (Mesh Backend)
- The `MeshSceneBackend` MUST use batched raycasting in `batch_step`.
- Individual actor rays must be concatenated into a single `intersects_location` call to leverage SIMD/Parallel throughput.
- **Benchmark:** Target ≥ 66 SPS for 4 actors at 128x24 on standard hardware.

### 3.2 Vision Transformer Optimization
- `RayViTEncoder` MUST cache fixed spherical positional encodings.
- Avoid redundant sin/cos recomputation on every forward pass.

### 3.3 Zero-Stall Telemetry
- High-frequency per-actor per-step telemetry is forbidden during training as it bottlenecks the CPU rollout loop.
- Use coarse-grained metrics (every 100 steps) for performance tracking.

### 3.4 Ghost-Matrix Persistence
- **No Collision Death:** Simulation backends MUST NOT trigger `done=True` on collision during training.
- **Continuous Learning:** Agents must learn to "escape" or "fly away" from geometry through continuous per-step negative rewards.
- **Context Preservation:** Temporal hidden states (Mamba) MUST NOT be reset upon grazing geometry, preserving situational awareness.
- **Hard Truncation:** A hard step limit (e.g., 2000 steps) MUST be enforced to ensure episodic diversity.

## 4) Resilient Diagnostic Standard
- Gallery Layer tools (Dashboard, Recorder) MUST be operational independent of Simulation/Brain layers.
- Tools MUST handle missing ZMQ streams gracefully, displaying a `WAITING` state instead of crashing.
- **Dynamic Discovery:** UI MUST NOT require hard-coded actor counts. It must detect, list, and switch between actors dynamically based on incoming ZMQ `env_ids`.
- **UI Throughput:** Processing (ZMQ polling, heavy rendering) MUST NOT block the UI thread for > 16ms per tick. Ingestion MUST be capped or moved to a background thread to maintain 60 FPS responsiveness.
- UI components MUST be non-blocking during stream connection attempts.

## 5) Data Isolation Boundary
External data sources connect strictly through a `DatasetAdapter` Protocol.
- **Location:** `environment/backends/`
- **Responsibility:** The ONLY place for axis transposes, depth normalisation, and semantic remapping.
- **Output:** Canonical `(1, Az, El)` DistanceMatrix format.

## 6) System Lifecycle Standard
- **Clean Reset:** Agents MUST use robust termination (taskkill /T) before restarting to ensure ZMQ ports are released.
- **Stateless Launch:** Tools MUST favor environment-based configuration over CLI parameters.
- **Visibility Verification:** Agents MUST NOT "fire and forget." Every GUI launch must be followed by a process-check and, if possible, a window-title verification to confirm the user can actually see the tool.
- **Verification:** Every restart MUST be followed by a log-check to confirm successful socket binding.

## 7) The Sacred Validation Pipeline

Every instruction MUST follow this exact sequence. Implementation without alignment is a structural failure.

1.  **Validate:** Cross-reference the request against `AGENTS.md` non-negotiables.
2.  **Document & Standardize:** Update `AGENTS.md` and `docs/*.md` to codify any new patterns, architectural shifts, or refined benchmarks.
3.  **Plan:** Present a detailed implementation strategy that adheres to the updated documentation.
4.  **Implement:** Execute code changes ONLY after the plan and documentation are verified and aligned.

## 8) Performance Benchmarks (Feb 2026)

| Metric | Target | Status |
|--------|--------|--------|
| **Rollout Throughput (4 Actors)** | ≥ 60 SPS | ACHIEVED (via Batched Raycasting) |
| **Inference Latency (CPU)** | ≤ 15ms/actor | ACHIEVED (via ViT Caching) |
| **Environment Latency (4 Actors)** | ≤ 25ms | ACHIEVED (via trimesh batching) |

---
*Status: Active canonical specification (Feb 2026)*
