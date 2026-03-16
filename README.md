# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first reinforcement learning stack built around one canonical
runtime path:

- transient source-scene staging during corpus refresh
- staged overwrite-first corpus refresh into compiled `.gmdag` assets
- batched CUDA `sdfdag` environment execution
- one sacred actor training engine with a selectable temporal core
- passive auditor observability

Canonical training and inference now assume a precompiled actor temporal-core
path that works on the active Windows hardware with no extra extension build.
The environment hot path is already sufficiently optimized that unfused Python
temporal implementations simply move the bottleneck into host-side PPO sequence
execution. For the active Windows machine, the native cuDNN GRU path is the
default canonical production path, with `mambapy` available only as an
explicit comparison backend on the same trainer and serve surfaces.

Current benchmark interpretation is now split clearly by layer:

- the environment CUDA path scales materially better with observation
	resolution than the full trainer
- end-to-end trainer scaling is currently limited by RayViT patch-token
	self-attention and PPO update cost, not by the CUDA ray marcher alone
- on the active MX150 `sm_61` machine, `512x96` remains runnable but is much
	slower than the canonical contract, while `768x144` remains viable on the
	environment-only benchmark surface and fails on the full trainer during
	actor-side attention allocation

## Canonical Workflow

```powershell
# 1. Refresh source scenes and compiled corpus
./scripts/refresh-scene-corpus.ps1

# 2. Start canonical continuous training
./scripts/train.ps1

# 2b. Compare the same run with Mambapy on the same trainer surface
./scripts/train.ps1 -TemporalCore mambapy

# 2c. Run a bounded side-by-side comparison for mambapy and GRU
./scripts/run-temporal-compare.ps1

# 2d. Run a bounded observation-resolution sweep on the canonical trainer
./scripts/run-resolution-compare.ps1

# 3. Or launch the full training stack
./scripts/run-ghost-stack.ps1 -Train

# 4. If actor telemetry 5557 is occupied later, move it explicitly
./scripts/train.ps1 -ActorTelemetryPort 5565
```

When no scene, manifest, or step limit is supplied, Navi now:

- uses the full discovered dataset corpus by default
- recompiles corpus assets when explicitly requested
- removes transient source downloads after a successful staged refresh
- trains continuously until stopped
- defaults the actor temporal core to `gru`, with `mambapy` available through an explicit selector on the same canonical surface

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
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# Canonical actor service
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Canonical actor service with Mambapy selected explicitly
./scripts/run-brain.ps1 -TemporalCore mambapy --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Canonical full inference stack
./scripts/run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\apartment_1.gmdag

# Canonical full training stack
./scripts/run-ghost-stack.ps1 -Train

# Canonical full training stack with Mambapy selected explicitly
./scripts/run-ghost-stack.ps1 -Train -TemporalCore mambapy
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
uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag --actors 4 --steps 200
```

## Training

```powershell
# Canonical continuous training on the full corpus
./scripts/train.ps1

# Canonical long-duration training wrapper
./scripts/train-all-night.ps1

# Canonical continuous training with an alternate actor telemetry port
./scripts/train.ps1 -ActorTelemetryPort 5565

# Canonical full-corpus training with 8 actors and passive dashboard attach
./scripts/run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard

# Durable direct trainer surface with explicit fleet size
uv run --project .\projects\actor navi-actor train --actors 4
uv run --project .\projects\actor navi-actor train --actors 8 --total-steps 0

# Separate passive dashboard attach for the direct trainer surface
uv run --project .\projects\auditor navi-auditor dashboard --actor-sub tcp://localhost:5557 --actor-control-endpoint tcp://localhost:5561 --passive --actor-id 0

# Bounded observation-resolution sweep on the canonical trainer surface
./scripts/run-resolution-compare.ps1
./scripts/run-resolution-compare.ps1 -Profiles @('256x48','384x72','512x96') -Repeats 2 -TotalSteps 512

# Refresh the default public bootstrap corpus (Habitat test scenes + ReplicaCAD stages)
./scripts/refresh-scene-corpus.ps1

# Explicit narrowing override
./scripts/train.ps1 -Scene .\data\scenes\hssd\102343992.glb -AutoCompileGmDag -TotalSteps 500000
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
| [docs/RESOLUTION_BENCHMARKS.md](docs/RESOLUTION_BENCHMARKS.md) | One-page appendix for the March 2026 observation-resolution sweep |
| [AGENTS.md](AGENTS.md) | Implementation policy and non-negotiables |

## Resolution Scaling Notes

Current repository evidence says the canonical `256x48` contract is still the
production default for good reason.

- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-003916/`
	shows the 4-actor GRU trainer at about `49.6 SPS` on `256x48`, about
	`49.3 SPS` on `384x72`, and about `44.0 SPS` on `512x96`, while
	`ppo_update_ms` rises from about `1.0 s` to about `17.7 s`
- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-004714/`
	plus `artifacts/benchmarks/resolution-compare/gpu-sample-512x96-20260317.csv`
	show `512x96` saturating the active MX150 near `2 GB` VRAM and pushing GPU
	utilization to `100%` during the PPO update window
- `artifacts/benchmarks/resolution-compare/resolution-compare-20260317-002948/768x144/repeat-01/train.log`
	shows the current trainer failing at `768x144` in
	`torch.nn.functional.multi_head_attention_forward`, which confirms that the
	active limit is actor-side attention memory rather than a `torch-sdf`
	runtime failure

This means stronger hardware can move the ceiling outward, but true
high-resolution end-to-end scaling will also require actor-side perception work
in addition to any future temporal-core promotion.

## Repository Commands

```bash
make sync-all
make test-all
make lint-all
make typecheck-all
make check-all
make clean-all
```
