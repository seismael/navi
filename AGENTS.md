# AGENTS.md — Ghost-Matrix Implementation Blueprint

This file is the **sacred source of truth** for implementation policy and architectural standards. All agent actions must be validated against this blueprint before execution.

## 1) Scope & Sovereignty

Navi is a modular ecosystem of isolated projects. Each project is a sovereign entity with its own environment, purpose, and configuration.

### Isolated Projects:
- **`projects/contracts`**: Wire-format models and serialization (The source of truth for communication).
- **`projects/environment`**: Simulation Layer (headless stepping + sensing).
- **`projects/actor`**: Brain Layer (policy + training — sacred, immutable engine).
- **`projects/auditor`**: Gallery Layer (record/replay/visualization — passive only).
- **`projects/voxel-dag`**: Offline mesh compiler producing `.gmdag` world caches.
- **`projects/torch-sdf`**: CUDA sphere-tracing runtime for batched DAG ray evaluation.

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
  - `NAVI_GMDAG_RESOLUTION=512` (Canonical corpus compile profile for the 256x48 observation contract)

### 2.2 Simple Launch Commands
Each project must provide a dedicated `uv run` shortcut and a corresponding wrapper script in `scripts/`:
- **Dashboard:** `uv run dashboard` → `scripts/run-dashboard.ps1` (Dynamic actor detection, no fleet size required).
- **Environment:** `uv run environment` → `scripts/run-environment.ps1`
- **Brain (Actor):** `uv run brain` → `scripts/run-brain.ps1`

### 2.3 Fleet Size Standard
- **Standard Fleet:** Training scripts and backends default to **4 parallel actors** for optimal hardware utilization.
- **Dashboard Throughput Default:** The Auditor Dashboard defaults to **actor 0** ingestion/rendering for maximum throughput.
- **Dynamic Discovery Mode:** Optional selector mode MAY enable dynamic actor discovery/switching for diagnostics.
- **Canonical Training Corpus Default:** When the user does not explicitly request a scene, manifest, or subset, canonical training MUST use the full discovered dataset corpus rather than a single sample scene.
- **Canonical Training Duration Default:** When the user does not explicitly request a step/time bound, canonical training MUST run continuously until stopped.

### 2.4 Mode Detection Standard
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
2. **Sacred Brain:** The cognitive pipeline (`RayViTEncoder` → `TemporalCore` → `EpisodicMemory` → `ActorCriticHeads`) is **immutable**.
3. **No Stall Mandate:** High-throughput training is the primary success metric. Optimization must never stall rollout.
4. **Project Isolation:** No service imports another service package (except for recognized CLI-level integrations).
5. **Quality Gates:** `ruff`, `mypy --strict`, and `pytest` are mandatory for all changes.

### 2.8 SDF/DAG Integration Standard
- **Internal Performance Domains:** `projects/voxel-dag` and `projects/torch-sdf` are sovereign internal domains and must be integrated as first-class projects, not copied into service packages.
- **Performance-First Path:** Canonical runtime evolution prioritizes `.gmdag` compilation plus batched `torch_sdf.cast_rays()` execution before any actor-brain redesign work.
- **Canonical Training Runtime:** All actor training entrypoints (`train`, `run-ghost-stack.ps1 -Train`, and wrapper training scripts) MUST run only on the `sdfdag` backend over compiled `.gmdag` assets.
- **Unified Training Direction:** Canonical high-throughput training MUST use one in-process unified trainer that removes the Environment<->Actor ZMQ control path from the rollout hot loop. Parallel training architectures, equivalent alternate modes, and dual canonical surfaces are forbidden.
- **Actor Preservation:** Environment and compiler upgrades MUST preserve the actor-side `DistanceMatrix` contract so `RayViTEncoder` and `Mamba2TemporalCore` remain unchanged.
- **No CPU Fallback In Canonical Path:** The canonical SDF/DAG runtime must fail fast when CUDA or the compiled backend is unavailable rather than silently degrading to a slower path.
- **Batching Rule:** New SDF/DAG execution must be designed around reusable GPU buffers and `batch_step()` throughput, not per-actor stepping or per-step allocation churn.
- **CLI-Level Integration Rule:** Any direct actor-to-environment integration for canonical training MUST occur at CLI or orchestration boundaries. Service packages remain sovereign; the actor may instantiate environment backends only in the single canonical training entrypoint, not as a permanent service-to-service dependency surface.
- **Tensor-Native Training Rule:** Canonical training internals MUST prefer CUDA tensor representations for observations, actions, rewards, and rollout storage. Materializing Python `DistanceMatrix` or `Action` objects inside the rollout hot path is forbidden except for coarse diagnostics, replay, or external service surfaces.
- **Corpus Refresh Rule:** Canonical training may auto-compile source scene meshes into `.gmdag` assets, and dataset refresh tooling MUST support overwrite-first refresh of source data and compiled outputs when explicitly requested.
- **Compile Profile Rule:** Canonical corpus refresh, environment CLI compilation, and actor training wrappers MUST default to `.gmdag` compile resolution `512` unless the user explicitly overrides it.
- **Transactional Refresh Rule:** Canonical corpus refresh MUST stage downloaded source datasets and newly compiled outputs outside the live corpus, promote the compiled corpus only after a successful rebuild, and remove transient source downloads after successful integration.
- **Resolution Mismatch Rule:** When a discovered compiled `.gmdag` asset does not match the requested canonical compile resolution, refresh tooling MUST automatically replace it rather than silently reusing it.

### 2.6 Compatibility Elimination Standard
- **Canonical-Only Runtime:** Backward-compatible runtime paths are forbidden. Keep only actively emitted, actively consumed, canonical event and API paths.
- **No Legacy Wrappers:** Do not keep compatibility wrappers, deprecated aliases, no-op compatibility methods, or old format shims once migration is complete.
- **No Deprecated Surface:** Do not introduce deprecation markers for removed legacy behavior. Remove unused legacy code directly and update callers/tests in the same change.
- **Hard Dependency Policy:** Optional fallback implementations that duplicate canonical behavior (e.g., algorithm/model fallbacks) are not permitted; fail fast when required dependencies are missing.
- **Strict Contract Evolution:** When a contract changes, update all producers, consumers, tests, and docs in one pass. Partial dual-path support is not allowed.
- **Training Surface Rule:** Scripts, examples, and CLI defaults for production training must advertise only the canonical `sdfdag` path. Alternative backend references may remain only where they are explicitly framed as diagnostic or regression tooling.
- **Default Scene Rule:** Scripts, examples, and CLI defaults for production training must advertise the full discovered canonical corpus as the default training dataset. Single-scene examples must be framed as explicit overrides.

### 2.7 Temporal Core Migration Standard (Mar 2026)
- **Cross-Platform Canonical Target:** The actor temporal core canonical runtime must be first-class on native Windows, Linux, and WSL2.
- **CUDA-Native Requirement:** Canonical actor training/inference must be supported with CUDA acceleration on native Windows and native Linux using officially supported package stacks.
- **Benchmark-Gated Selection:** Canonical temporal core selection is decided by measured bake-off results under actor interface parity `(B,T,D)->(B,T,D)` and rollout constraints.
- **Benchmark Device Rule:** Final backend selection benchmarks must run on CUDA (not CPU-only diagnostics) on both Windows and Linux.
- **Performance Floor:** Migration is accepted only if 4-actor throughput remains `>= 60 SPS`; otherwise selection re-opens.
- **Single Canonical Runtime:** Candidate comparison tooling is allowed during migration, but production runtime must keep one canonical temporal backend with no fallback branch.

## 3) Performance Mandates

### 3.1 Batched Sphere Tracing (SDF/DAG Canonical Path)
- The high-performance target runtime is `.gmdag` compilation via `projects/voxel-dag` followed by batched `torch_sdf.cast_rays()` execution from `projects/torch-sdf`.
- The SDF/DAG backend MUST keep DAG data and ray buffers GPU-resident across steps.
- Observation adaptation back to `(1, Az, El)` `DistanceMatrix` format MUST be vectorized and must not introduce avoidable CPU copies in the hot path.
- The SDF/DAG backend MUST expose one canonical rolling perf snapshot surface consumed by both runtime telemetry and direct benchmarking; duplicate timing paths are forbidden.
- `uv run navi-environment bench-sdfdag --gmdag-file ...` is the canonical environment-layer throughput command for the compiled path.
- **Benchmark Gate:** The SDF/DAG path is accepted only if it meets or exceeds current fleet rollout throughput and materially advances the repository toward the `>= 60 SPS` 4-actor floor.

### 3.2 Vision Transformer Optimization
- `RayViTEncoder` MUST cache fixed spherical positional encodings.
- Avoid redundant sin/cos recomputation on every forward pass.

### 3.3 Zero-Stall Telemetry
- High-frequency per-actor per-step telemetry is forbidden during training as it bottlenecks the CPU rollout loop.
- Use coarse-grained metrics (every 100 steps) for performance tracking.
- Default mode detection/reporting should rely on low-volume `actor.training.*` update/perf/episode telemetry.
- Environment-side compiled-path perf telemetry MUST remain coarse (`environment.sdfdag.perf`) and must not emit more frequently than the existing 100-step cadence.

### 3.3.1 Actor Hot-Path Discipline
- Once SDF/DAG acceleration makes the environment subdominant, actor-side rollout code becomes the critical path and must be treated like systems code.
- Episodic-memory eviction MUST NOT trigger full-index rebuilds on every post-capacity insert.
- Canonical rollout loops MUST minimize avoidable CUDA-to-CPU synchronization (`.item()`, `.cpu()`, `.numpy()`) inside per-step per-actor sections.
- Canonical rollout loops MUST not convert batched CUDA ray outputs into CPU `numpy`/Python observation objects and then back into CUDA tensors during training; tensor-native runtime seams are mandatory once the backend is already GPU-resident.
- Canonical PPO update code MUST avoid redundant tensor device copies, allocator churn, and repeated optimizer-side host sync when minibatch tensors already reside on CUDA.
- Canonical PPO update work SHOULD prefer optimizer/runtime improvements on the existing learner path before changing training hyperparameter defaults.
- Actor-side performance telemetry SHOULD expose enough sub-metrics to distinguish memory, transport, reward shaping, and buffer-append overhead when investigating stalls.
- Canonical throughput investigation MUST add attribution on the existing `navi-actor train` surface rather than creating alternate trainer modes or shadow benchmarking entrypoints.
- Diagnostic ablations MAY temporarily disable telemetry emission, episodic-memory query/add, or reward shaping on the canonical trainer, but only as explicitly labeled attribution controls and never as a second production default.
- Diagnostic ablations MUST remain robust in every emitted combination; disabling observation and training telemetry while leaving perf telemetry enabled must not crash the canonical trainer.
- Canonical rollout code MUST batch any unavoidable host extraction so a single tick does not perform repeated per-actor scalar synchronizations when equivalent batched transfers are available.
- Canonical CLI defaults for actor training MUST match the intended performance-safe config defaults unless a wrapper explicitly overrides them for a documented profile.
- Canonical throughput work MUST prefer GPU-resident rollout storage and direct backend stepping before spending effort on smaller Python-side micro-optimisations.
- Canonical throughput work MUST remove Python per-actor reward, memory, and transition loops from the rollout hot path once observation and action tensors are already batched on device.
- Canonical cleanup work MUST remove non-essential training modes rather than preserving multiple architecture paths with equal status.

### 3.5 Soft Stall Monitoring
- Optimization stall prevention defaults to **soft monitoring** (warning logs), not automatic intervention.
- Warn when optimizer wall-time or rollout health signals degrade beyond thresholds.

### 3.4 Ghost-Matrix Persistence
- **No Collision Death:** Simulation backends MUST NOT trigger `done=True` on collision during training.
- **Continuous Learning:** Agents must learn to "escape" or "fly away" from geometry through continuous per-step negative rewards.
- **Context Preservation:** Temporal hidden states (selected canonical temporal core) MUST NOT be reset upon grazing geometry, preserving situational awareness.
- **Hard Truncation:** A hard step limit (e.g., 2000 steps) MUST be enforced to ensure episodic diversity.

## 4) Resilient Diagnostic Standard
- Gallery Layer tools (Dashboard, Recorder) MUST be operational independent of Simulation/Brain layers.
- Tools MUST handle missing ZMQ streams gracefully, displaying a `WAITING` state instead of crashing.
- During canonical training, the Dashboard MUST run in passive actor-only mode: subscribe to the actor PUB stream only and MUST NOT open environment REP/manual-step control paths or depend on environment PUB availability.
- **Default Filter:** UI defaults to actor-0 stream filtering to preserve training throughput.
- **Dynamic Discovery (Optional):** UI may detect/list/switch actors dynamically when selector mode is explicitly enabled.
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
- **Fail-Fast Startup:** Port/thread/readiness verification failures MUST terminate startup immediately with actionable errors (no warn-and-continue defaults).
- **Visibility Verification:** Agents MUST NOT "fire and forget." Every GUI launch must be followed by a process-check and, if possible, a window-title verification to confirm the user can actually see the tool.
- **Verification:** Every restart MUST be followed by a log-check to confirm successful socket binding.
- **Refresh Semantics:** Dataset download/transform tooling MUST support explicit overwrite-first refresh semantics so stale local source meshes and stale compiled `.gmdag` outputs are not silently reused during corpus rebuilds.

## 7) The Sacred Validation Pipeline

Every instruction MUST follow this exact sequence. Implementation without alignment is a structural failure.

1.  **Validate:** Cross-reference the request against `AGENTS.md` non-negotiables.
2.  **Document & Standardize:** Update `AGENTS.md` and `docs/*.md` to codify any new patterns, architectural shifts, or refined benchmarks.
3.  **Plan:** Present a detailed implementation strategy that adheres to the updated documentation.
4.  **Implement:** Execute code changes ONLY after the plan and documentation are verified and aligned.

## 8) Performance Benchmarks (Feb 2026)

| Metric | Target | Status |
|--------|--------|--------|
| **Rollout Throughput (4 Actors)** | ≥ 60 SPS | ACHIEVED (via batched SDF/DAG execution) |
| **Inference Latency (CPU)** | ≤ 15ms/actor | ACHIEVED (via ViT Caching) |
| **Environment Latency (4 Actors)** | ≤ 25ms | ACHIEVED (via compiled SDF/DAG batching) |

---
*Status: Active canonical specification (Feb 2026)*
