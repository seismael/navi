# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first reinforcement learning stack built around one canonical
runtime path:

- transient source-scene staging during corpus refresh
- staged overwrite-first corpus refresh into compiled `.gmdag` assets
- batched CUDA `sdfdag` environment execution
- one sacred actor training engine
- passive auditor observability

## Canonical Workflow

```powershell
# 1. Refresh source scenes and compiled corpus
./scripts/refresh-scene-corpus.ps1

# 2. Start canonical continuous training
./scripts/train.ps1

# 3. Or launch the full training stack
./scripts/run-ghost-stack.ps1 -Train
```

When no scene, manifest, or step limit is supplied, Navi now:

- uses the full discovered dataset corpus by default
- recompiles corpus assets when explicitly requested
- removes transient source downloads after a successful staged refresh
- trains continuously until stopped

## Project Layout

| Project | Layer | Description |
|---------|-------|-------------|
| [`projects/contracts`](projects/contracts) | Shared | Wire-format models, serialization, ZMQ topics |
| [`projects/environment`](projects/environment) | Environment | Canonical `sdfdag` runtime, corpus prep, `.gmdag` compiler orchestration |
| [`projects/actor`](projects/actor) | Brain | Sacred cognitive engine, PPO training, policies |
| [`projects/auditor`](projects/auditor) | Gallery | Recording, replay, live dashboard |
| [`projects/voxel-dag`](projects/voxel-dag) | Compiler | Offline mesh-to-`.gmdag` compiler |
| [`projects/torch-sdf`](projects/torch-sdf) | Runtime | CUDA sphere-tracing execution engine |

## Manual Services

```bash
# Environment
cd projects/environment
uv sync
uv run environment

# Actor
cd ../actor
uv sync
uv run brain

# Auditor dashboard
cd ../auditor
uv sync
uv run dashboard
```

## Windows Launchers

```powershell
# Canonical environment service
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag

# Canonical actor service
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Canonical full inference stack
./scripts/run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag

# Canonical full training stack
./scripts/run-ghost-stack.ps1 -Train
```

## Corpus Tooling

```powershell
# Download source scenes directly when you explicitly want raw assets retained
./scripts/download-habitat-data.ps1

# Full staged overwrite-first source + compiled corpus refresh
./scripts/refresh-scene-corpus.ps1

# Explicit environment-layer corpus prep
uv run --project .\projects\environment navi-environment prepare-corpus --force-recompile

# Validate and benchmark the compiled runtime
uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag --actors 4 --steps 200
```

## Training

```powershell
# Canonical continuous training on the full corpus
./scripts/train.ps1

# Canonical long-duration training wrapper
./scripts/train-all-night.ps1

# Explicit narrowing override
./scripts/train.ps1 -Scene .\data\scenes\replicacad\frl_apartment_stage.glb -AutoCompileGmDag -TotalSteps 500000
```

## Wire Topics

| Topic | Direction |
|-------|-----------|
| `distance_matrix_v2` | Environment → Brain, Gallery |
| `action_v2` | Brain → Environment, Gallery |
| `step_request_v2` | Brain → Environment |
| `step_result_v2` | Environment → Brain |
| `telemetry_event_v2` | Any → Gallery |

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Canonical system architecture and runtime boundaries |
| [docs/ACTOR.md](docs/ACTOR.md) | Sacred actor engine specification |
| [docs/TRAINING.md](docs/TRAINING.md) | Canonical corpus refresh, training, resume, and recovery flow |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Canonical environment runtime and corpus preparation |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Canonical wire contracts |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Throughput targets and performance notes |
| [AGENTS.md](AGENTS.md) | Implementation policy and non-negotiables |

## Repository Commands

```bash
make sync-all
make test-all
make lint-all
make typecheck-all
make check-all
make clean-all
```
