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
- **Dashboard Throughput Default:** The Auditor Dashboard always displays **actor 0** observations with zero selector overhead and shows the active actor count in the status bar.
- **Actor Display Rule:** The dashboard hardcodes actor 0 display. No selector mechanism, control endpoint, or roster query is required. Live dashboard-grade observations MUST stay scoped to actor 0 on the canonical path.
- **Canonical Training Corpus Default:** When the user does not explicitly request a scene, manifest, or subset, canonical training MUST use the full discovered dataset corpus rather than a single sample scene.
- **Canonical Bootstrap Dataset Default:** When bootstrap downloads are requested without an explicit dataset override, the canonical source refresh MUST fetch both the public Habitat test scenes and the public ReplicaCAD stage set so default training is not limited to the 3-scene test bundle.
- **Canonical Observation Default:** All production-facing CLI commands, wrappers, benchmarks, and training surfaces MUST default to the full-resolution `256x48` observation contract. Retired `128x24` defaults are forbidden.
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
- **Run Identity:** Every canonical process and wrapper MUST stamp one shared `run_id` into logs, metrics, manifests, summaries, captures, and default checkpoint roots so whole-run review is correlation-safe.
- **Format:** Logs must use a professional, high-quality structure: `[%(asctime)s] [%(levelname)-8s] [run=%(run_id)s] [%(name)s:%(lineno)d] - %(message)s`.
- **Stable Plus Run-Scoped Logs:** Canonical services MUST keep the stable top-level `logs/` surface for operator tailing while also mirroring process logs into the active run root under `artifacts/runs/<run_id>/logs/` or another explicitly selected canonical run root.
- **Manifest Rule:** Every CLI or wrapper entrypoint MUST write one machine-readable manifest under the run root describing the process, command surface, key parameters, and output roots.
- **Metrics Rule:** Canonical training, qualification, nightly, and benchmark surfaces MUST emit append-only machine-readable metrics or summary artifacts under the run root `metrics/` or `reports/` directories in addition to human-readable logs.
- **Enforcement:** Each project's CLI entry point (`cli.py`) MUST invoke this setup immediately upon startup.

### 2.4.4 Artifact Governance Standard
- **Canonical Run Root:** Default generated operational outputs MUST live under one run root, normally `artifacts/runs/<run_id>/`, unless a workflow intentionally uses a more specific governed root such as `artifacts/nightly/<run_id>/`.
- **Structured Layout:** Canonical run roots MUST separate `logs/`, `metrics/`, `manifests/`, `reports/`, `captures/`, and `checkpoints/` so operators do not need to infer file meaning from ad hoc names.
- **Wrapper Ownership:** PowerShell orchestration surfaces in `scripts/` MUST create the run context before launching child processes and MUST pass the shared run environment through to descendants.
- **Retention Rule:** Wrapper-owned cleanup MAY aggressively remove stale generated captures, detached validation leftovers, benchmark scratch outputs, and other transient artifacts using explicit retention windows, but MUST NOT delete the active run root.
- **Review Rule:** Major operational flows MUST leave enough machine-readable evidence to reconstruct what ran, with which parameters, on which surfaces, and where the primary logs, metrics, and checkpoints were written.
- **Git Hygiene Rule:** Generated logs, run roots, checkpoints, captures, benchmark outputs, and other local operational artifacts MUST be ignored in `.gitignore` with explicit path coverage rather than relying only on loose wildcard patterns.

### 2.4.5 Training Measurement Standard
- **Coarse Phase Metrics Required:** Canonical training entrypoints and wrappers MUST emit append-only machine-readable phase metrics for corpus preparation, trainer construction, trainer start/stop, rollout heartbeat, PPO update, checkpoint load/save, and final summary boundaries.
- **Resource Snapshot Rule:** Canonical training metrics MUST capture low-overhead process resource snapshots at those coarse boundaries, including CPU-process memory and thread state plus CUDA allocator/free-memory state when the trainer is running on GPU.
- **Wrapper Measurement Rule:** Canonical wrapper surfaces in `scripts/` MUST record orchestration costs such as cleanup, child-process launch, readiness wait, and observed exit completion under the active run root metrics directory.
- **Control Surface Rule:** Measurement enablement and verbosity MUST be controlled from the canonical training config, environment variables, or the existing training CLI surface. Standalone monitor-only launcher scripts are not canonical control surfaces.
- **No Hot-Path Polling Rule:** High-frequency per-step resource polling, shelling out for GPU stats inside the rollout hot path, or any measurement pattern that materially perturbs throughput is forbidden on the canonical training surface.
- **Log Hygiene Rule:** Detailed measurement belongs in run-scoped metrics artifacts, not repetitive human log lines. Human-readable logs MAY summarize coarse measurements at the existing logging cadence but MUST NOT expand into per-step stat spam.

### 2.4.1 Nightly Validation Standard
- **Single Canonical Nightly Surface:** End-to-end overnight validation MUST run from one orchestration entrypoint in `scripts/` rather than a loose manual checklist.
- **Hard Gates First:** CUDA/runtime preflight, focused regression suites, and bounded canonical qualification MUST complete successfully before the overnight soak begins.
- **Shared-Model Proof:** The nightly flow MUST prove that canonical multi-actor training can checkpoint, resume, and continue emitting fresh checkpoints from one shared model state.
- **Soft Warning Policy:** Throughput drift, attach instability, and other non-fatal regressions MAY emit warnings, but hard failures must stop the nightly and produce machine-readable artifacts.
- **Artifact Rule:** Every nightly run MUST emit one governed run root under `artifacts/nightly/<run_id>/` containing raw phase outputs plus a top-level machine-readable summary for morning review.

### 2.4.3 Observation-Resolution Benchmark Standard
- **Split Benchmark Rule:** Observation-resolution changes MUST be benchmarked on both the environment-only `bench-sdfdag` surface and the end-to-end canonical trainer surface. One measurement does not stand in for the other.
- **Canonical Interpretation Rule:** Environment viability at a higher ray count does NOT prove end-to-end trainer viability. Documentation, scripts, and benchmark summaries MUST distinguish runtime scaling from actor-side scaling.
- **Active Machine Truth Rule:** On the active MX150 `sm_61` machine, the canonical `256x48` contract remains the production default. Higher profiles such as `512x96` are diagnostic comparison surfaces, and `768x144` currently exceeds full-trainer memory limits during actor-side attention even though the environment runtime itself remains benchmark-viable.
- **Future-Promotion Rule:** Better hardware or a future fused temporal core MAY move the trainer ceiling outward, but those upgrades MUST NOT be documented as solving high-resolution scaling by themselves while RayViT token attention still dominates the actor hot path.

### 2.4.2 SDF/DAG Validation Standard
- **Layered Proof Required:** `projects/voxel-dag` and `projects/torch-sdf` MUST be validated at compiler, binary-format, runtime, corpus-promotion, and qualification levels rather than by one smoke test.
- **Mathematical Oracle Rule:** Small canonical fixtures MUST be checked against independent or analytic geometry expectations, not only implementation-to-implementation self-comparison.
- **Oracle Fixture Family Rule:** Canonical correctness work MUST maintain one reusable small-fixture family with known expected observations so compiler, environment, actor-seam, and auditor tests share the same geometry truth instead of duplicating ad hoc literals.
- **Determinism Rule:** Repeated compilation of the same fixture with the same inputs MUST produce byte-identical `.gmdag` output unless the documented format contract is intentionally revised in the same change.
- **Binary Integrity Rule:** The `.gmdag` loader MUST reject malformed headers, non-finite bounds, impossible pointer layouts, out-of-range child references, cycles, and trailing or truncated payloads.
- **Runtime Validation Boundary Rule:** Full `.gmdag` pointer-layout traversal belongs to explicit integrity and promoted-corpus validation surfaces. Ordinary environment startup and benchmarking MUST trust already-qualified canonical assets after lightweight binary sanity checks rather than re-running whole-DAG validation on every load.
- **Real Corpus Rule:** Canonical promoted-corpus validation MUST prove that real compiled assets from approved datasets load, match manifest metadata, and remain benchmark-viable at the canonical compile resolution.
- **Benchmark Drift Rule:** Compiler/runtime validation MUST track both correctness and throughput so a faster but geometrically wrong transform, or a correct but throughput-regressing transform, cannot silently ship.

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
- **Actor Preservation:** Environment and compiler upgrades MUST preserve the actor-side `DistanceMatrix` contract so `RayViTEncoder` and the selected canonical temporal core remain unchanged.
- **No CPU Fallback In Canonical Path:** The canonical SDF/DAG runtime must fail fast when CUDA or the compiled backend is unavailable rather than silently degrading to a slower path.
- **Batching Rule:** New SDF/DAG execution must be designed around reusable GPU buffers and `batch_step()` throughput, not per-actor stepping or per-step allocation churn.
- **CLI-Level Integration Rule:** Any direct actor-to-environment integration for canonical training MUST occur at CLI or orchestration boundaries. Service packages remain sovereign; the actor may instantiate environment backends only in the single canonical training entrypoint, not as a permanent service-to-service dependency surface.
- **Tensor-Native Training Rule:** Canonical training internals MUST prefer CUDA tensor representations for observations, actions, rewards, and rollout storage. Materializing Python `DistanceMatrix` or `Action` objects inside the rollout hot path is forbidden except for coarse diagnostics, replay, or external service surfaces.
- **No Host-Staged Canonical Rollout Rule:** Canonical rollout storage remains GPU-resident on the active tensor-native training path. Pinned-CPU or host-first rollout slabs are forbidden on the canonical path unless the architecture standard is intentionally revised in the same change.
- **Corpus Refresh Rule:** Canonical training may auto-compile source scene meshes into `.gmdag` assets, and dataset refresh tooling MUST support overwrite-first refresh of source data and compiled outputs when explicitly requested.
- **Compile Profile Rule:** Canonical corpus refresh, environment CLI compilation, and actor training wrappers MUST default to `.gmdag` compile resolution `512` unless the user explicitly overrides it.
- **Real Dataset Scene Rule:** Canonical tests, benchmarks, scripts, and training surfaces MUST use only real dataset scenes or real compiled dataset `.gmdag` assets. Generated, synthetic, procedural, or sample-scene substitutes are forbidden on the canonical path.
- **Transactional Refresh Rule:** Canonical corpus refresh MUST stage downloaded source datasets and newly compiled outputs outside the live corpus, promote the compiled corpus only after a successful rebuild, and remove transient source downloads after successful integration.
- **Resolution Mismatch Rule:** When a discovered compiled `.gmdag` asset does not match the requested canonical compile resolution, refresh tooling MUST automatically replace it rather than silently reusing it.
- **Upgrade Refresh Rule:** After compiler/runtime package upgrades that may affect compiled asset structure or performance, the canonical corpus MUST be fully refreshed before performance conclusions are drawn.

### 2.6 Compatibility Elimination Standard
- **Canonical-Only Runtime:** Backward-compatible runtime paths are forbidden. Keep only actively emitted, actively consumed, canonical event and API paths.
- **No Legacy Wrappers:** Do not keep compatibility wrappers, deprecated aliases, no-op compatibility methods, or old format shims once migration is complete.
- **No Deprecated Surface:** Do not introduce deprecation markers for removed legacy behavior. Remove unused legacy code directly and update callers/tests in the same change.
- **Hard Dependency Policy:** Optional fallback implementations that duplicate canonical behavior (e.g., algorithm/model fallbacks) are not permitted; fail fast when required dependencies are missing.
- **Strict Contract Evolution:** When a contract changes, update all producers, consumers, tests, and docs in one pass. Partial dual-path support is not allowed.
- **Training Surface Rule:** Scripts, examples, and CLI defaults for production training must advertise only the canonical `sdfdag` path. Alternative backend references may remain only where they are explicitly framed as diagnostic or regression tooling.
- **Default Scene Rule:** Scripts, examples, and CLI defaults for production training must advertise the full discovered canonical corpus as the default training dataset. Single-scene examples must be framed as explicit overrides.

### 2.7 Temporal Core Standard (Mar 2026)
- **Canonical Runtime:** The actor temporal core canonical runtime is the pure-PyTorch Mamba-2 SSD path because a 25K-step head-to-head training comparison proved significantly better learning quality (final reward_ema -0.88 vs -1.48) over cuDNN GRU, with only a modest throughput trade-off (~72 SPS vs ~100 SPS) that is dominated by PPO optimizer cost rather than the temporal core itself.
- **No-Build Rule:** The canonical temporal runtime must work from the standard PyTorch CUDA install with no custom C++ extension build step, no Triton dependency, and no platform-specific fused wheels.
- **Controlled Selector Rule:** Production scripts, wrappers, defaults, and docs MUST expose one temporal-core selector contract on the one canonical trainer surface. The active machine supports `mamba2` as the default runtime, with `gru` and `mambapy` available as explicit comparison backends; alternate trainer architectures and parallel production surfaces are forbidden.
- **Update-Frequency Rule:** PPO cadence tuning MAY reduce how often optimizer updates run, but must happen on the one canonical trainer surface rather than by introducing alternate temporal-runtime branches.
- **Future Promotion Rule:** Hardware-fused `mamba-ssm` remains a future upgrade target. It may replace the active Mamba2 SSD path only after a supported environment exists and the real bounded trainer proves the upgrade.
- **Promotion Rule:** Benchmark-proven end-to-end training quality or throughput wins MAY replace the current temporal-core default or other canonical performance defaults, but only when config defaults, wrappers, tests, and docs are updated together in the same change so the repository has one coherent source of truth.

## 3) Performance Mandates

### 3.1 Batched Sphere Tracing (SDF/DAG Canonical Path)
- The high-performance target runtime is `.gmdag` compilation via `projects/voxel-dag` followed by batched `torch_sdf.cast_rays()` execution from `projects/torch-sdf`.
- The SDF/DAG backend MUST keep DAG data and ray buffers GPU-resident across steps.
- Canonical `SdfDagBackend` tensor-only post-cast, reward, and kinematics helper graphs SHOULD use `torch.compile` when they stay on the PyTorch side of the `torch_sdf.cast_rays()` boundary so eager micro-kernel dispatch does not become the next environment bottleneck. When the active GPU/compiler stack cannot support that path, the runtime MUST warn and remain on the eager tensor path rather than fail startup.
- Observation adaptation back to `(1, Az, El)` `DistanceMatrix` format MUST be vectorized and must not introduce avoidable CPU copies in the hot path.
- The SDF/DAG backend MUST expose one canonical rolling perf snapshot surface consumed by both runtime telemetry and direct benchmarking; duplicate timing paths are forbidden.
- `uv run navi-environment bench-sdfdag --gmdag-file ...` is the canonical environment-layer throughput command for the compiled path.
- Any TSDF-derived runtime changes MUST be benchmark-gated. The first accepted canonical step is aligning the CUDA sphere-tracing horizon with the configured environment horizon before any `.gmdag` format or storage-layout redesign work.
- **Benchmark Gate:** The SDF/DAG path is accepted only if it meets or exceeds current fleet rollout throughput and materially advances the repository toward the `1K SPS` theoretical limit on current hardware, pacing toward the ultimate `10K SPS` distributed advanced-hardware target.

### 3.1.1 Low-Level Tensor Contract Standard
- Python-to-CUDA boundaries MUST validate device, dtype, rank, shape, and contiguity before launching raw kernels.
- Canonical `torch-sdf` ray inputs use contiguous CUDA `float32` tensors shaped `[batch, rays, 3]`; canonical outputs use preallocated CUDA tensors shaped `[batch, rays]`.
- Long-running CUDA extension calls MUST release the Python GIL while the kernel is executing.
- Canonical environment and actor runtimes MUST keep tensor-native observation and action seams available so Python `DistanceMatrix` and `Action` materialization is optional for diagnostics, telemetry, and passive viewers only.
- Canonical hot-path `cast_rays()` calls MUST use `skip_direction_validation=True` when ray directions are mathematically guaranteed normalized (e.g. yaw-rotated unit vectors) to eliminate GPU→CPU synchronization barriers; probe and inspection calls MUST retain full validation.
- Low-level `torch-sdf` cache removals or redesigns MUST be justified by reproduced current-branch evidence and benchmarked against the canonical trainer or `bench-sdfdag` surface before promotion.
- Low-level DAG layout, leaf payload packing, cache assumptions, and related storage changes MAY be documented, but they do not become stable architectural guarantees until end-to-end benchmark wins are proven on the canonical trainer.

### 3.1.2 GPU Compute Utilization Standard (Mar 2026)
- On the active MX150 (`sm_61`), GPU compute utilization is structurally limited by eager PyTorch dispatcher overhead; each tensor operation dispatches a separate CUDA kernel with ~10-100μs Python-side idle gap between launches.
- `torch.compile` (requires `sm_70+`) is the highest-ROI path to GPU utilization improvement; it fuses multiple PyTorch operations into single kernels, eliminating dispatcher gaps.
- CUDA graph capture of the full environment step path is NOT FEASIBLE due to data-dependent control flow (`nonzero()`, `.item()`, dynamic allocation, tensor-dependent loop bounds, advanced indexing).
- The Mamba2 SSD temporal core dispatches ~55-60 CUDA kernels per forward pass versus GRU's 2-4 (cuDNN fused); this is the largest single contributor to Mamba2's throughput disadvantage and cannot be fixed without hardware-fused `mamba-ssm` Triton kernels (not available on Windows).
- PPO/rollout double-buffer overlap (running environment steps during PPO backward passes) would eliminate ~1000ms of GPU idle per PPO window but requires complex architecture work not yet implemented.
- `rollout_overlap_groups` (`ActorConfig`) enables multi-group pipelined rollout for larger GPUs; default `1` is optimal for MX150's 3 SMs. 2-group overlap causes ~47% throughput regression on MX150 due to SM starvation.

### 3.2 Vision Transformer Optimization
- `RayViTEncoder` MUST cache fixed spherical positional encodings.
- Avoid redundant sin/cos recomputation on every forward pass.
- Observation-resolution documentation and performance conclusions MUST account for the real RayViT token path: with the canonical `patch_size=8`, token count scales as `(Az / 8) * (El / 8)` and full self-attention cost grows roughly with the square of that count.

### 3.3 Zero-Stall Telemetry
- **Decoupled Two-Tier Telemetry Structure:** Performance and observability must be strictly separated.
  - **Tier 1 (Heavy Math):** Episode summaries, collision counts, max/min reward shapes, and complex stats MUST be deferred to the PPO Update block (Epoch Boundary). Aggregations MUST happen via bulk device-side tensor mapping to avoid step-by-step CPU disruption.
  - **Tier 2 (Visual Heartbeat):** Live dashboard frame delivery MUST remain independent of step math, pushing solitary `(1, 256, 48)` distance tensors to an async stream exactly 5-10 times a second using a clock, avoiding synchronization with the specific mathematical rollouts.
- High-frequency per-actor per-step tensor-to-scalar extraction logic in the hot loop is functionally forbidden as it triggers mandatory CUDA/CPU blocking queues.
- Environment-side compiled-path perf telemetry MUST remain coarse (`environment.sdfdag.perf`) and must not emit more frequently than the existing cadence.

### 3.3.1 Actor Hot-Path Discipline
- Once SDF/DAG acceleration makes the environment subdominant, actor-side rollout code becomes the critical path and must be treated like systems code.
- Episodic-memory eviction MUST NOT trigger full-index rebuilds on every post-capacity insert.
- Canonical rollout loops MUST minimize avoidable CUDA-to-CPU synchronization (`.item()`, `.cpu()`, `.numpy()`) inside per-step per-actor sections.
- Canonical rollout loops MUST not convert batched CUDA ray outputs into CPU `numpy`/Python observation objects and then back into CUDA tensors during training; tensor-native runtime seams are mandatory once the backend is already GPU-resident.
- Canonical tensor-step training surfaces MUST keep per-step reward, truncation, env-id, and episode-id bookkeeping in tensors; rebuilding Python `StepResult` objects during every rollout tick is forbidden unless an explicit diagnostic or public-result caller requests materialization.
- Canonical observation shape, axis ordering, normalization, and stepping semantics MUST remain viewer-invariant; dashboard, recorder, replay, or other observer requirements MUST NOT change environment/core math or contract behavior.
- Any observation publication or telemetry materialization required for live viewing MUST remain a passive seam outside the core rollout math and MUST NOT become a prerequisite for canonical training or inference.
- Tensor-only actor helper graphs such as reward shaping SHOULD use `torch.compile` on supported GPU/compiler stacks when they remain on the PyTorch side of the rollout hot path; unsupported stacks MUST fall back cleanly to eager execution with explicit attribution visibility.
- Canonical PPO update code MUST avoid redundant tensor device copies, allocator churn, and repeated optimizer-side host sync when minibatch tensors already reside on CUDA.
- Canonical PPO update code MUST keep the temporal-core sequence path inside efficient backend operators such as Mamba2 SSD chunked scans or cuDNN GRU on the active path; Python-level sequence unrolling, dispatcher-heavy scan decomposition, and large autograd-graph construction are forbidden on the production hot path.
- Canonical throughput analysis MUST not blame high-resolution trainer regressions on the SDF/DAG runtime alone when RayViT encoder attention or PPO update memory clearly dominates the measured wall time.
- Canonical PPO update work SHOULD prefer optimizer/runtime improvements on the existing learner path before changing training hyperparameter defaults.
- Actor-side performance telemetry SHOULD expose enough sub-metrics to distinguish memory, transport, reward shaping, and buffer-append overhead when investigating stalls.
- Canonical throughput investigation MUST add attribution on the existing `navi-actor train` surface rather than creating alternate trainer modes or shadow benchmarking entrypoints.
- Diagnostic ablations MAY temporarily disable telemetry emission, episodic-memory query/add, or reward shaping on the canonical trainer, but only as explicitly labeled attribution controls and never as a second production default.
- Diagnostic ablations MUST remain robust in every emitted combination; disabling observation and training telemetry while leaving perf telemetry enabled must not crash the canonical trainer.
- Canonical rollout code MUST batch any unavoidable host extraction so a single tick does not perform repeated per-actor scalar synchronizations when equivalent batched transfers are available.
- Canonical CLI defaults for actor training MUST match the intended performance-safe config defaults unless a wrapper explicitly overrides them for a documented profile.
- Canonical throughput work MUST prefer GPU-resident rollout storage and direct backend stepping before spending effort on smaller Python-side micro-optimisations.
- Canonical throughput work MUST remove Python per-actor reward, memory, and transition loops from the rollout hot path once observation and action tensors are already batched on device.
- Grouped rollout-overlap rewrites MUST preserve actor-local `observation -> action -> next observation` ordering. No group may consume stale observations from another phase just to increase overlap.
- Canonical cleanup work MUST remove non-essential training modes rather than preserving multiple architecture paths with equal status.

### 3.5 Soft Stall Monitoring
- Optimization stall prevention defaults to **soft monitoring** (warning logs), not automatic intervention.
- Warn when optimizer wall-time or rollout health signals degrade beyond thresholds.

### 3.4 Ghost-Matrix Persistence
- **No Collision Death:** Simulation backends MUST NOT trigger `done=True` on collision during training.
- **Continuous Learning:** Agents must learn to "escape" or "fly away" from geometry through continuous per-step negative rewards.
- **Escape Incentive:** Canonical environment reward MUST provide a positive shaping signal when an actor increases obstacle clearance while near geometry so recovery behavior is learned in-scene instead of via reset churn.
- **Proximity-Discounted Progress:** Progress reward MUST be discounted by proximity ratio so approaching walls yields diminishing forward credit instead of rewarding unsafe approaches.
- **Velocity-Scaled Collision:** Collision penalty MUST scale with movement speed so fast crashes are punished more severely than gentle grazing, creating a natural speed-awareness incentive.
- **Clearance-Gated Exploration:** Exploration rewards MUST be gated by current clearance so pushing into tight geometry yields diminishing exploration credit instead of rewarding wall-hugging.
- **Forward Velocity Bias Elimination:** Actor-side forward velocity reward weight MUST default to `0.0` so reward shaping does not create an inherent approach bias toward obstacles.
- **Conservative Speed Default:** Drone max speed MUST default to `5.0 m/s` or less so the proximity speed limiter has adequate reaction time before geometry contact.
- **Information Foraging Incentive:** Canonical environment reward MUST penalize overly blind horizon-saturated views and near-field wall-hugging using starvation/proximity ratios derived from the already-produced spherical observation.
- **Structure Seeking Incentive:** Canonical environment reward MUST positively value stable mid-range structure visibility and reorientation toward informative geometry so agents prefer navigable interiors over empty void-facing views.
- **Inspection Incentive:** Canonical reward MAY positively score controlled look-around only when the spherical observation gains structure information; blind spinning in empty space MUST NOT become rewarding.
- **Context Preservation:** Temporal hidden states (selected canonical temporal core) MUST NOT be reset upon grazing geometry, preserving situational awareness.
- **Hard Truncation:** A hard step limit (e.g., 2000 steps) MUST be enforced to ensure episodic diversity.
- **Scene Residency:** Canonical scene-pool training MUST keep actors on each compiled scene for multiple completed episodes before rotation; switching after only one episode per actor is too eager for both throughput and local scene mastery.

## 4) Resilient Diagnostic Standard
- Gallery Layer tools (Dashboard, Recorder) MUST be operational independent of Simulation/Brain layers.
- Tools MUST handle missing ZMQ streams gracefully, displaying a `WAITING` state instead of crashing.
- During canonical training, the Dashboard MUST run in passive actor-only mode: subscribe to the actor PUB stream only and MUST NOT open environment REP/manual-step control paths or depend on environment PUB availability.
- Dashboard heartbeats during optimizer windows MUST reuse the same passive observation cadence policy as live observation publication instead of falling back to a slower hidden rate.
- Observer-side slicing, half-sphere extraction, palette choice, labels, or other presentation transforms MUST remain inside the Auditor domain and MUST NOT be implemented by changing environment or actor contracts.
- **Actor 0 Display:** The dashboard always displays actor 0 observations and shows the active actor count in the status bar.
- **Actor View Geometry:** The primary actor panel MUST render a direct centered `180` degree half-sphere slice from the canonical `256x48` spherical observation, which is an exact `128x48` forward hemisphere before viewport scaling.
- **No Dashboard Range Cap:** The primary actor panel MUST colorize the published distance matrix directly and MUST NOT overlay dashboard-specific range caps such as fixed `RANGE`, `CTR`, or `HFOV` meter labels.
- **Heading Guides:** The actor heading remains the center column; LEFT and RIGHT labels MUST sit on the horizontal midline of the actor panel rather than the bottom edge.
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
| **Rollout Throughput Limit (Current Hardware)** | ~1,000 SPS | IN PROGRESS (Bottleneck resolution ongoing) |
| **Rollout Throughput Target (Advanced Hardware)** | 10,000 SPS | PLANNED (Architectural target) |
| **Inference Latency (CPU)** | ≤ 15ms/actor | ACHIEVED (via ViT Caching) |
| **Environment Latency (4 Actors)** | ≤ 25ms | ACHIEVED (via compiled SDF/DAG batching) |

---
*Status: Active canonical specification (Feb 2026)*
