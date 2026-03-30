# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first reinforcement-learning system for autonomous drone
navigation. Source scene meshes are compiled into compressed `.gmdag` signed
distance fields, traced at batch scale on the GPU by a CUDA sphere-tracing
kernel, and consumed by a sacred cognitive actor engine (RayViTEncoder →
Mamba-2 SSD → EpisodicMemory → PPO) that trains directly on the compiled
corpus with no intermediate graphics pipeline.

The canonical runtime path:

1. **Compile** — offline mesh → `.gmdag` via `voxel-dag` (SDF + DAG deduplication)
2. **Trace** — batched CUDA sphere tracing via `torch-sdf`, results written
   directly into PyTorch CUDA tensors
3. **Train** — unified in-process PPO trainer with GPU-resident rollout storage,
   selectable temporal core (`mamba2` default, `gru` and `mambapy` comparison)
4. **Observe** — passive auditor dashboard subscribes to the actor telemetry
   stream without gating throughput

---

## Quick Start

```powershell
# 1. Install all project dependencies
make sync-all

# 2. Refresh the public scene corpus (download + compile to .gmdag)
./scripts/refresh-scene-corpus.ps1

# 3. Start canonical continuous training on the full corpus
./scripts/train.ps1

# 4. (Optional) Attach a live dashboard to the running trainer
uv run --project .\projects\auditor navi-auditor dashboard --actor-sub tcp://localhost:5557 --passive
```

---

## Project Layout

| Project | Layer | Purpose |
|---------|-------|---------|
| [`projects/contracts`](projects/contracts) | Shared | Wire-format models (`DistanceMatrix`, `Action`, `StepResult`), serialization, ZMQ topics |
| [`projects/environment`](projects/environment) | Simulation | Headless `sdfdag` stepping, corpus preparation, `.gmdag` compiler orchestration |
| [`projects/actor`](projects/actor) | Brain | Sacred cognitive engine, PPO trainer, policy checkpointing |
| [`projects/auditor`](projects/auditor) | Gallery | Live PyQtGraph dashboard, Zarr recording, session replay |
| [`projects/voxel-dag`](projects/voxel-dag) | Compiler | Offline mesh-to-`.gmdag` compiler (FSM SDF + DAG deduplication) |
| [`projects/torch-sdf`](projects/torch-sdf) | Runtime | CUDA sphere-tracing kernel with zero-copy PyTorch tensor I/O |

Each project is a sovereign package with its own `pyproject.toml`, virtual
environment, and test suite. Cross-project imports only occur at CLI
orchestration boundaries.

---

## Training

```powershell
# Canonical continuous training (full corpus, mamba2 default)
./scripts/train.ps1

# Long-duration overnight wrapper
./scripts/train-all-night.ps1

# Full ghost stack with passive dashboard
./scripts/run-ghost-stack.ps1 -Train -WithDashboard

# Explicit fleet size
./scripts/run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard

# Alternate temporal core on the same trainer surface
./scripts/train.ps1 -TemporalCore gru

# Dataset filtering
./scripts/run-ghost-stack.ps1 -Train -Datasets replicacad
./scripts/run-ghost-stack.ps1 -Train -ExcludeDatasets hssd

# Override to a single scene
./scripts/train.ps1 -Scene .\data\scenes\hssd\102343992.glb -AutoCompileGmDag -TotalSteps 500000

# Alternate telemetry port when 5557 is occupied
./scripts/train.ps1 -ActorTelemetryPort 5565

# Direct trainer CLI
uv run --project .\projects\actor navi-actor train --actors 4
```

When no scene, manifest, or step limit is supplied, training uses the full
discovered dataset corpus and runs continuously until stopped.

---

## Manual Training (Behavioral Cloning)

Human demonstration capture and supervised pre-training before RL fine-tuning.

```powershell
# 1. Explore a scene and record demonstrations (auto-starts recording)
uv run --project .\projects\auditor explore --record --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# 2. Train from recorded demonstrations
uv run --project .\projects\actor brain bc-pretrain

# 3. Resume training with existing checkpoint (incremental multi-scene)
uv run --project .\projects\actor brain bc-pretrain --checkpoint artifacts\checkpoints\bc_base_model.pt

# 4. Automated multi-scene training loop
./scripts/run-manual-training.ps1
```

Demonstrations are saved as `.npz` files under `artifacts/demonstrations/`.
The BC checkpoint is a standard v2 file loadable by `navi-actor train --checkpoint`
for RL fine-tuning.

See [docs/TRAINING.md](docs/TRAINING.md) for algorithm details and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) § 10.4 for the full pipeline design.

---

## Corpus Tooling

```powershell
# Full staged overwrite-first corpus refresh (download + compile)
./scripts/refresh-scene-corpus.ps1

# Download source scenes only (raw assets retained locally)
./scripts/download-habitat-data.ps1

# Expand the ReplicaCAD corpus with baked-lighting scenes from HuggingFace
./scripts/expand-replicacad-corpus.ps1

# Explicit environment-layer corpus prep
uv run --project .\projects\environment navi-environment prepare-corpus --force-recompile

# Compile a single asset
uv run --project .\projects\environment navi-environment compile-gmdag --source .\data\scenes\hssd\102343992.glb --output .\artifacts\gmdag\corpus\hssd\102343992.gmdag --resolution 512

# Validate and benchmark a compiled asset
uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag --actors 4 --steps 200
```

---

## Services (Manual Launch)

Each service can be started independently for debugging or inference workflows.

```powershell
# Environment
uv run --project .\projects\environment environment

# Actor (brain)
uv run --project .\projects\actor brain

# Dashboard
uv run --project .\projects\auditor dashboard
```

### Windows Wrapper Scripts

```powershell
# Environment service
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# Actor service
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Full inference stack
./scripts/run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\apartment_1.gmdag

# Full training stack
./scripts/run-ghost-stack.ps1 -Train

# Dashboard
./scripts/run-dashboard.ps1
```

---

## Benchmarking & Comparison

```powershell
# Bounded temporal-core comparison (mamba2 vs gru)
./scripts/run-temporal-compare.ps1

# Temporal-core microbenchmark (writes JSON artifact)
./scripts/run-temporal-bakeoff.ps1

# Observation-resolution sweep on the canonical trainer
./scripts/run-resolution-compare.ps1
./scripts/run-resolution-compare.ps1 -Profiles @('256x48','384x72','512x96') -Repeats 2 -TotalSteps 512

# Actor-scaling test
./scripts/run-actor-scaling-test.ps1

# Nightly end-to-end validation
./scripts/run-nightly-validation.ps1

# Qualification suite
./scripts/qualify-canonical-stack.ps1
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `train.ps1` | Canonical continuous training wrapper |
| `train-all-night.ps1` | Durable overnight training |
| `run-ghost-stack.ps1` | Orchestrated multi-service launcher (train / inference) |
| `refresh-scene-corpus.ps1` | Staged overwrite-first corpus refresh |
| `download-habitat-data.ps1` | Source scene download only |
| `expand-replicacad-corpus.ps1` | Incremental ReplicaCAD baked-lighting expansion |
| `run-environment.ps1` | Environment service wrapper |
| `run-brain.ps1` | Actor service wrapper |
| `run-dashboard.ps1` | Dashboard wrapper |
| `run-temporal-compare.ps1` | Bounded end-to-end temporal-core comparison |
| `run-temporal-bakeoff.ps1` | Temporal-core microbenchmark |
| `run-resolution-compare.ps1` | Observation-resolution trainer sweep |
| `run-actor-scaling-test.ps1` | Fleet-size scaling benchmark |
| `run-nightly-validation.ps1` | End-to-end nightly validation suite |
| `qualify-canonical-stack.ps1` | Canonical stack qualification |
| `run-attribution-matrix.ps1` | Throughput attribution diagnostics |
| `run-manual-training.ps1` | Multi-scene BC demonstration + training loop |
| `setup-actor-cuda.ps1` | CUDA wheel installation for actor env |
| `check_gpu.py` | GPU availability check |
| `summarize-bounded-train-log.ps1` | Post-run log summarizer |

---

## Wire Protocol

All inter-service communication uses ZMQ with MessagePack serialization.

| Topic | Direction | Transport |
|-------|-----------|-----------|
| `distance_matrix_v2` | Environment → Brain, Gallery | PUB/SUB |
| `action_v2` | Brain → Environment, Gallery | PUB/SUB |
| `step_request_v2` | Brain → Environment | REQ/REP |
| `step_result_v2` | Environment → Brain | REQ/REP |
| `telemetry_event_v2` | Any → Gallery | PUB/SUB |

### Default Ports

| Port | Service | Role |
|------|---------|------|
| `5559` | Environment | PUB (`distance_matrix_v2`) |
| `5560` | Environment | REP (`step_request_v2` / `step_result_v2`) |
| `5557` | Actor | PUB (`action_v2`, `telemetry_event_v2`) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and runtime boundaries |
| [docs/GMDAG.md](docs/GMDAG.md) | `.gmdag` binary format specification |
| [docs/ACTOR.md](docs/ACTOR.md) | Sacred actor engine specification |
| [docs/TRAINING.md](docs/TRAINING.md) | Corpus refresh, training, resume, and recovery |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Environment runtime and corpus preparation |
| [docs/SDFDAG_RUNTIME.md](docs/SDFDAG_RUNTIME.md) | SDF/DAG backend runtime details |
| [docs/COMPILER.md](docs/COMPILER.md) | Voxel-DAG compiler internals |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Wire-format contract specification |
| [docs/DATAFLOW.md](docs/DATAFLOW.md) | End-to-end data flow |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Throughput targets and analysis |
| [docs/RESOLUTION_BENCHMARKS.md](docs/RESOLUTION_BENCHMARKS.md) | Observation-resolution sweep results |
| [docs/AUDITOR.md](docs/AUDITOR.md) | Auditor layer specification (incl. demonstration recording) |
| [docs/NIGHTLY_VALIDATION.md](docs/NIGHTLY_VALIDATION.md) | Nightly validation pipeline |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | SDF/DAG validation standard |
| [docs/PARALLEL.md](docs/PARALLEL.md) | Parallel architecture notes |
| [docs/COMPARISON.md](docs/COMPARISON.md) | Temporal-core comparison results |
| [docs/TSDF.md](docs/TSDF.md) | Legacy TSDF reference |
| [AGENTS.md](AGENTS.md) | Implementation policy and non-negotiables |

---

## Performance

| Metric | Target | Status |
|--------|--------|--------|
| Rollout throughput (current hardware) | ~1,000 SPS | In progress |
| Rollout throughput (advanced hardware) | 10,000 SPS | Planned |
| Inference latency (CPU) | ≤ 15 ms/actor | Achieved |
| Environment latency (4 actors) | ≤ 25 ms | Achieved |

The canonical `256×48` observation contract remains the production default.
Higher profiles (`384×72`, `512×96`) are viable but increasingly limited by
RayViT self-attention and PPO update cost rather than the CUDA ray marcher.
See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/RESOLUTION_BENCHMARKS.md](docs/RESOLUTION_BENCHMARKS.md) for the full
analysis.

---

## Repository Commands

```bash
make help            # Show all available targets
make sync-all        # Install dependencies in all sub-projects
make test-all        # Run pytest in all sub-projects
make lint-all        # Run ruff check + format check
make format-all      # Run ruff format
make typecheck-all   # Run mypy --strict
make check-all       # lint + typecheck + tests (CI gate)
make clean-all       # Remove .venv, caches, build artifacts
make bench-temporal  # Run temporal-core bake-off
```

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use,
modify, and distribute this software, provided the original copyright notice and
license text are included in all copies or substantial portions of the software.
