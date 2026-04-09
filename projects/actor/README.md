← [Navi Overview](../../README.md)

# navi-actor

Brain Layer for the Navi ecosystem. Houses the sacred cognitive engine, PPO
trainer, and policy checkpoint management.

**Full specification:** [docs/ACTOR.md](../../docs/ACTOR.md)  
**Training guide:** [docs/TRAINING.md](../../docs/TRAINING.md)  
**Inference guide:** [docs/INFERENCE.md](../../docs/INFERENCE.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Table of Contents

- [Install](#install)
- [Engine Isolation Principle](#engine-isolation-principle)
- [Cognitive Architecture](#cognitive-architecture)
- [CLI Reference](#cli-reference)
- [Training](#training)
- [Inference](#inference)
- [Behavioral Cloning](#behavioral-cloning)
- [Model Management](#model-management)
- [Temporal Core](#temporal-core)
- [Benchmarking](#benchmarking)
- [Validation & Qualification](#validation--qualification)
- [Checkpoint Format (v3)](#checkpoint-format-v3)
- [Resolution Scaling](#resolution-scaling)
- [Default Port](#default-port)
- [Validation](#validation)

---

## Install

```bash
cd projects/actor
uv sync
```

CUDA wheel setup (pins `torch==2.5.1+cu121` for `sm_61+`):

```powershell
.\scripts\setup-actor-cuda.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-CudaTag` | `cu121` | CUDA version tag |
| `-TorchVersion` | `2.5.1` | PyTorch version |
| `-SkipActorSync` | (switch) | Skip `uv sync` after install |
| `-InstallFusedTemporal` | (switch) | Install mamba-ssm fused wheels |

**Linux/WSL2 variant:** `./scripts/setup-actor-cuda.sh`

---

## Engine Isolation Principle

**The training engine is sacred.** The cognitive pipeline is never modified to
accommodate a new data source. External data always connects through a
`DatasetAdapter` in `environment/backends/` that transforms raw observations
into the canonical `(1, Az, El)` `DistanceMatrix` format.

---

## Cognitive Architecture

`CognitiveMambaPolicy` implements a 5-stage pipeline:

1. **RayViTEncoder** — ViT `(B, 3, Az, El)` → `(B, 128)` spatial embedding
2. **RND Curiosity** — intrinsic exploration reward from embedding novelty
3. **EpisodicMemory** — tensor-native cosine-similarity loop-avoidance
4. **TemporalCore** — sequence engine (`mamba2` default, `gru` and `mambapy` comparison)
5. **ActorCriticHeads** — 4-DOF action distribution + value estimation

Input contract: only `depth` and `semantic` from `DistanceMatrix` are consumed.

---

## CLI Reference

**Base command:** `uv run --project projects\actor navi-actor <command>`  
**Shortcut:** `uv run --project projects\actor brain <subcommand>` → unified entry point

### `train` — Unified In-Process PPO Training

The canonical training entrypoint. Instantiates the sdfdag environment backend
directly with GPU-resident rollout storage and no ZMQ in the hot loop.

```powershell
uv run --project projects\actor navi-actor train [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actors` | int | `4` | Parallel actor count |
| `--temporal-core` | string | `mamba2` | Temporal backend (`mamba2` / `gru` / `mambapy`) |
| `--total-steps` | int | `0` | Steps before auto-stop (0 = continuous) |
| `--rollout-length` | int | `256` | Steps per rollout window |
| `--minibatch-size` | int | `64` | PPO minibatch size |
| `--ppo-epochs` | int | `2` | PPO optimizer epochs per update |
| `--learning-rate` | float | `3e-4` | Learning rate |
| `--bptt-len` | int | `8` | Truncated BPTT sequence length |
| `--checkpoint-dir` | string | `""` | Checkpoint output directory |
| `--checkpoint` | string | `""` | Resume from checkpoint |
| `--scene` | string | `""` | Single scene GLB override |
| `--manifest` | string | `""` | Corpus manifest path |
| `--corpus-root` | string | `""` | Corpus root directory |
| `--gmdag-root` | string | `""` | `.gmdag` root directory |
| `--gmdag-file` | string | `""` | Single `.gmdag` file override |
| `--compile-resolution` | int | `512` | Auto-compile resolution |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |
| `--enable-episodic-memory` | flag | | Enable episodic memory module |
| `--enable-reward-shaping` | flag | | Enable reward shaping |
| `--emit-observation-stream` | flag | | Enable observation PUB stream |
| `--emit-training-telemetry` | flag | | Enable training telemetry events |
| `--emit-perf-telemetry` | flag | | Enable performance telemetry |

### `serve` — Actor Policy Server

ZMQ-based actor for the 3-process inference stack.

```powershell
uv run --project projects\actor navi-actor serve [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sub` | string | `tcp://localhost:5559` | SUB socket (environment observations) |
| `--pub` | string | `tcp://localhost:5557` | PUB socket (actions + telemetry) |
| `--mode` | string | `async` | Service mode (`async` / `step`) |
| `--policy-checkpoint` | string | `""` | Trained checkpoint to load |
| `--temporal-core` | string | `mamba2` | Temporal backend |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |

### `infer` — In-Process Policy Evaluation

Evaluate a trained checkpoint with direct in-process CUDA stepping.
Same backend as training (SdfDagBackend, tensor-native), but without PPO,
rollout buffers, or episodic memory.

```powershell
uv run --project projects\actor navi-actor infer --checkpoint .\model.pt [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint` | string | (required) | Trained model checkpoint path |
| `--actors` | int | `4` | Parallel actor count |
| `--deterministic` | flag | | Use action mean instead of sampling |
| `--total-steps` | int | `0` | Step limit (0 = unlimited) |
| `--total-episodes` | int | `0` | Episode limit (0 = unlimited) |
| `--temporal-core` | string | `mamba2` | Temporal backend |
| `--gmdag-file` | string | | Single `.gmdag` file path |
| `--gmdag-root` | string | | Custom `.gmdag` root |
| `--corpus-root` | string | | Corpus root directory |
| `--manifest` | string | | Corpus manifest path |
| `--datasets` | string | | Include only these datasets |
| `--exclude-datasets` | string | | Exclude datasets by name |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |

### `bc-pretrain` — Behavioral Cloning Pre-Training

Supervised learning from recorded `.npz` demonstrations.

```powershell
uv run --project projects\actor brain bc-pretrain [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--demonstrations` | string | `artifacts/demonstrations` | Demonstration directory |
| `--output` | string | | Output checkpoint path |
| `--checkpoint` | string | `""` | Resume from BC checkpoint |
| `--temporal-core` | string | `""` | Temporal backend override |
| `--embedding-dim` | int | `128` | Embedding dimension |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |
| `--epochs` | int | `50` | Training epochs |
| `--learning-rate` | float | `1e-3` | Learning rate |
| `--bptt-len` | int | `8` | BPTT sequence length |

### `profile` — Throughput Profiling

Run fixed-length rollout with CUDA profiling active.

```powershell
uv run --project projects\actor navi-actor profile [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scene` | string | | Scene to profile |
| `--steps` | int | `512` | Rollout steps |
| `--actors` | int | `4` | Actor count |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |

### `brain` — Unified Entry Point

Delegates to `train`, `serve`, `infer`, or `profile` based on the subcommand:

```powershell
uv run --project projects\actor brain train [options]
uv run --project projects\actor brain serve [options]
uv run --project projects\actor brain infer [options]
uv run --project projects\actor brain bc-pretrain [options]
uv run --project projects\actor brain profile [options]
```

---

## Training

Unified in-process training: the actor instantiates the `sdfdag` environment
backend directly, with GPU-resident rollout storage and no ZMQ in the hot loop.
When no scene or step limit is supplied, training uses the **full discovered
corpus** and runs **continuously until stopped**.

### Standard PPO Training

```powershell
.\scripts\train.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-NumActors` | `4` | Parallel actor count |
| `-TemporalCore` | `mamba2` | Temporal backend (`mamba2` / `gru` / `mambapy`) |
| `-TotalSteps` | `0` (continuous) | Steps before auto-stop (0 = forever) |
| `-RolloutLength` | `512` | Steps per rollout before PPO update |
| `-MinibatchSize` | `64` | PPO minibatch size |
| `-PpoEpochs` | `1` | PPO optimizer epochs per update |
| `-LearningRate` | `5e-4` | Learning rate |
| `-EntropyCoeff` | `0.02` | Entropy regularization weight |
| `-ExistentialTax` | `-0.02` | Per-step existence penalty |
| `-BpttLen` | `8` | Truncated BPTT sequence length |
| `-CheckpointEvery` | `25000` | Steps between checkpoint saves |
| `-CheckpointDir` | `""` | Custom checkpoint output directory |
| `-ResumeCheckpoint` | `""` | Resume from existing checkpoint |
| `-Scene` | `""` | Override to single scene file |
| `-CorpusRoot` | `""` | Custom corpus root directory |
| `-AzimuthBins` | `256` | Observation azimuth resolution |
| `-ElevationBins` | `48` | Observation elevation resolution |
| `-ActorTelemetryPort` | `5557` | Actor PUB port for dashboard |

```powershell
# Examples:
.\scripts\train.ps1                                                    # Full corpus, continuous
.\scripts\train.ps1 -TemporalCore gru                                 # GRU temporal core
.\scripts\train.ps1 -TotalSteps 50000                                  # Bounded 50K steps
.\scripts\train.ps1 -NumActors 8                                       # 8 parallel actors
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\latest.pt  # Resume training

# Direct CLI (bypasses wrapper):
uv run --project projects\actor navi-actor train --actors 4 --temporal-core mamba2
```

### Ghost Stack Training (Orchestrated)

Full-stack launcher with optional passive dashboard.

```powershell
.\scripts\run-ghost-stack.ps1 -Train
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Train` | (switch) | Unified training mode |
| `-Actors` | `4` | Parallel actor count |
| `-WithDashboard` | (switch) | Auto-attach passive dashboard |
| `-NoDashboard` | (switch) | Suppress dashboard |
| `-Datasets` | `""` | Dataset filter (e.g. `"quake3-arenas"`) |
| `-ExcludeDatasets` | `""` | Exclude datasets by name |
| `-TotalSteps` | `0` | Step limit |
| `-CheckpointEvery` | `25000` | Checkpoint interval |
| `-TemporalCore` | `mamba2` | Temporal backend |
| `-NoPreKill` | (switch) | Skip killing stale processes |

```powershell
# Examples:
.\scripts\run-ghost-stack.ps1 -Train -WithDashboard                    # Train with live view
.\scripts\run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard          # 8 actors + dashboard
.\scripts\run-ghost-stack.ps1 -Train -Datasets "quake3-arenas"         # Q3 maps only
.\scripts\run-ghost-stack.ps1 -Train -TotalSteps 10000                 # Bounded training
```

### Overnight Training

Durable unattended training with checkpoint monitoring.

```powershell
.\scripts\train-all-night.ps1
```

Same parameters as `train.ps1`. Includes CUDA environment setup and process
cleanup. Runs until interrupted (Ctrl+C) or system shutdown.

### Training Artifacts

```
artifacts/
├── runs/<run_id>/
│   ├── logs/               Per-run log files
│   ├── metrics/            Append-only machine-readable metrics
│   ├── manifests/          Process manifest (params, surfaces, roots)
│   ├── checkpoints/        Model checkpoints (periodic)
│   └── reports/            Human-readable summaries
├── checkpoints/            Stable checkpoint surface
└── logs/                   Stable log surface (rotating, max 1MB × 10)
```

---

## Inference

Evaluate a trained policy checkpoint with direct in-process CUDA stepping.
Same architecture as training (SdfDagBackend, tensor-native, ZMQ telemetry)
but without PPO, rollout buffers, reward shaping, or episodic memory.
See also: [docs/INFERENCE.md](../../docs/INFERENCE.md)

### Ghost Stack Inference

```powershell
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Infer` | (switch) | In-process inference mode |
| `-Checkpoint` | (required) | Trained model checkpoint path |
| `-Actors` | `4` | Parallel actor count |
| `-Deterministic` | (switch) | Use action mean instead of sampling |
| `-TotalSteps` | `0` | Step limit (0 = unlimited) |
| `-TotalEpisodes` | `0` | Episode limit (0 = unlimited) |
| `-Datasets` | `""` | Dataset filter |
| `-ExcludeDatasets` | `""` | Exclude datasets |
| `-NoDashboard` | (switch) | Skip dashboard (enabled by default) |
| `-TemporalCore` | `mamba2` | Temporal backend |

```powershell
# Examples:
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -Deterministic
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -TotalSteps 10000
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -NoDashboard
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -Datasets "quake3-arenas"
```

### Standalone Inference Wrapper

```powershell
.\scripts\run-inference.ps1 -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Checkpoint` | (required) | Trained model checkpoint path |
| `-Actors` | `4` | Parallel actor count |
| `-Deterministic` | (switch) | Use action mean instead of sampling |
| `-TotalSteps` | `0` | Step limit (0 = unlimited) |
| `-TotalEpisodes` | `0` | Episode limit (0 = unlimited) |
| `-TemporalCore` | `mamba2` | Temporal backend |
| `-Datasets` | `""` | Include only these datasets |
| `-ExcludeDatasets` | `""` | Exclude datasets by name |

### Direct CLI

```powershell
uv run --project projects\actor navi-actor infer `
    --checkpoint .\artifacts\checkpoints\bc_base_model.pt `
    --actors 4 --deterministic --total-steps 10000
```

---

## Behavioral Cloning

Human demonstration capture and supervised pre-training. Use this to bootstrap
a base policy before RL fine-tuning.

### Phase 1 — Collect Demonstrations

Demonstrations are recorded in the auditor project. Navigate scenes continuously;
each scene auto-closes after `MaxSteps` and the next one opens immediately.

```powershell
.\scripts\run-explore-scenes.ps1
.\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
.\scripts\run-explore-scenes.ps1 -MaxSteps 2000
```

See [projects/auditor/README.md](../auditor/README.md#demonstration-recording)
for the full exploration and recording workflow.

### Phase 2 — Train on Demonstrations

```powershell
.\scripts\run-bc-pretrain.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Demonstrations` | `artifacts/demonstrations` | Path to `.npz` demo files |
| `-Output` | `artifacts/checkpoints/bc_base_model.pt` | Output checkpoint |
| `-Checkpoint` | `""` | Resume from existing checkpoint |
| `-Epochs` | `50` | Training epochs |
| `-LearningRate` | `1e-3` | Learning rate |
| `-BpttLen` | `8` | BPTT sequence length |
| `-MinibatchSize` | `32` | Minibatch size |
| `-TemporalCore` | `""` | Temporal backend override |

```powershell
# Examples:
.\scripts\run-bc-pretrain.ps1                                          # Fresh start
.\scripts\run-bc-pretrain.ps1 -Epochs 100 -LearningRate 5e-4          # Custom hyperparams
.\scripts\run-bc-pretrain.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt  # Resume
```

### Phase 3 — Fine-Tune with RL

Promote the BC checkpoint and start RL training:

```powershell
# Promote to model registry (makes it the default starting point)
uv run --project projects\actor brain promote artifacts\checkpoints\bc_base_model.pt --notes "BC baseline"

# RL training auto-continues from latest promoted model
.\scripts\run-ghost-stack.ps1 -Train -Actors 4
```

---

## Model Management

```bash
# Promote a checkpoint to the model registry
uv run brain promote ./checkpoints/policy_step_0050000.pt --notes "50K RL run" --tags rl,mamba2

# List all promoted models
uv run brain models

# Evaluate a checkpoint (bounded inference with quality metrics)
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000

# Compare two checkpoints side-by-side
uv run brain compare ./artifacts/models/v001.pt ./artifacts/models/v002.pt --steps 2000

# Evaluate with JSON output for scripting
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000 --json
```

Training auto-continues from `artifacts/models/latest.pt` when no `--checkpoint` is specified.
After training, the final checkpoint is auto-promoted if its `reward_ema` exceeds the current latest.

**Registry location:** `artifacts/models/` with `registry.json` (version catalog),
`latest.pt` (best model pointer), and versioned `vNNN.pt` copies.

---

## Temporal Core

The canonical default is **pure-PyTorch Mamba-2 SSD**, proven by a 25K-step
training comparison to deliver significantly better learning quality
(reward_ema −0.88 vs GRU's −1.48) with a modest throughput trade-off
(~72 SPS vs ~100 SPS).

Available backends on the same trainer surface:

| Backend | Status |
|---------|--------|
| `mamba2` | **Default** — pure-PyTorch SSD, no build step |
| `gru` | Comparison — cuDNN fused, highest throughput |
| `mambapy` | Comparison — reference Mamba implementation |

Future hardware-fused `mamba-ssm` remains deferred until a supported
environment exists and end-to-end training proves the upgrade.

---

## Benchmarking

### Temporal Core Comparison (End-to-End)

Compare `mamba2`, `gru`, and `mambapy` on identical training runs.

```powershell
.\scripts\run-temporal-compare.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-TemporalCores` | `@("mamba2")` | Cores to compare |
| `-TotalSteps` | `256` | Steps per run |
| `-Repeats` | `3` | Independent repetitions |
| `-ProfileCudaEvents` | (switch) | Enable CUDA event tracing |
| `-OutputRoot` | `artifacts/benchmarks/temporal-compare` | Results directory |

```powershell
# Full comparison:
.\scripts\run-temporal-compare.ps1 -TemporalCores @("mamba2","gru") -TotalSteps 1024 -Repeats 3
```

### Temporal Kernel Microbenchmark

Isolated forward/backward pass timing (no training loop overhead).

```powershell
.\scripts\run-temporal-bakeoff.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Candidates` | `@("mamba2")` | Backends to benchmark |
| `-Batch` / `-SeqLen` / `-DModel` | `16` / `128` / `128` | Tensor dimensions |
| `-Repeats` / `-Warmup` | `40` / `10` | Iteration counts |
| `-Device` | `cuda` | Device (`cuda` / `cpu`) |
| `-OutputDir` | `artifacts/benchmarks/temporal` | JSON output directory |

### Resolution Scaling Benchmark

Compare training throughput across observation resolutions.

```powershell
.\scripts\run-resolution-compare.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Profiles` | `@("256x48","512x96","768x144")` | Resolution profiles |
| `-Repeats` | `3` | Repetitions per profile |
| `-TotalSteps` | `512` | Steps per run |
| `-TemporalCore` | `mamba2` | Temporal backend |
| `-OutputRoot` | `artifacts/benchmarks/resolution-compare` | Results |

### Actor Scaling Test

Find optimal parallel actor count for current hardware.

```powershell
.\scripts\run-actor-scaling-test.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-ActorCountsStr` | `"4,8,12,16,20,24,28,32"` | Actor counts to test |
| `-StepsPerRun` | `2000` | Training steps per count |
| `-TemporalCore` | `mamba2` | Temporal backend |

**Output:** Summary table showing peak SPS per actor count → `artifacts/benchmarks/actor-scaling/`

### Attribution Matrix

Systematic ablation: disable individual actor-side components to isolate
throughput bottlenecks.

```powershell
.\scripts\run-attribution-matrix.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-GmDagFile` | (scene file) | Test scene |
| `-Actors` | `4` | Actor count |
| `-TotalSteps` | `2048` | Steps per configuration |
| `-OutputRoot` | `artifacts/benchmarks/attribution-matrix` | Results |

Tests combinations of: episodic memory, reward shaping, observation stream,
training telemetry, perf telemetry — all enabled/disabled systematically.

---

## Validation & Qualification

### Canonical Stack Qualification

One-pass end-to-end proof: dataset audit → bounded training → checkpoint
resume → replay.

```powershell
.\scripts\qualify-canonical-stack.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-TotalSteps` | `512` | Qualification training steps |
| `-CheckpointEvery` | `256` | Checkpoint interval |
| `-SkipDatasetAudit` | (switch) | Skip dataset validation |
| `-SkipResumeQualification` | (switch) | Skip resume proof |
| `-RunRoot` | `artifacts\qualification\canonical_stack` | Output root |

### Nightly Validation (Overnight)

Comprehensive automated validation pipeline (typically 8+ hours).

```powershell
.\scripts\run-nightly-validation.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-SoakHours` | `8` | Overnight soak duration |
| `-QualificationSteps` | `512` | Initial qualification steps |
| `-MonitorIntervalSeconds` | `300` | Checkpoint monitoring interval |
| `-AttachTimeoutSeconds` | `20` | Dashboard attach timeout |
| `-RunRoot` | `artifacts/nightly` | Nightly run root |

**Nightly pipeline stages:**

1. Runtime preflight (CUDA, compiler, dataset audit)
2. Focused regression suites
3. Bounded qualification (512 steps)
4. Checkpoint + resume proof (256 + 256 steps)
5. Environment drift benchmarks
6. Long-duration soak with checkpoint monitoring
7. Summary artifact emission

**Artifacts:** Governed run root → `artifacts/nightly/<run_id>/`

See also: [docs/NIGHTLY_VALIDATION.md](../../docs/NIGHTLY_VALIDATION.md)

---

## Checkpoint Format (v3)

| Key | Description |
|-----|-------------|
| `version` | Format version (3) |
| `step_id` | Total training steps completed |
| `episode_count` | Total episodes completed |
| `reward_ema` | Exponential moving average of episode reward |
| `wall_time_hours` | Cumulative training wall time |
| `parent_checkpoint` | Path to the checkpoint this was resumed from |
| `training_source` | `"rl"`, `"bc"`, or `"inference"` |
| `temporal_core` | Active temporal core (`mamba2`, `gru`, `mambapy`) |
| `corpus_summary` | Description of training data |
| `created_at` | ISO timestamp |
| `policy_state_dict` | All policy network weights |
| `rnd_state_dict` | RND predictor + running statistics |
| `optimizer_state_dict` | Adam optimizer state (if available) |
| `rnd_optimizer_state_dict` | RND optimizer state (if available) |
| `reward_shaper_step` | Reward shaper annealing step counter |

Only v3 checkpoints are accepted. The `--checkpoint` flag resumes training
from any previous checkpoint, preserving all learned knowledge across sessions.

---

## Resolution Scaling

Token count scales as `(Az / 8) × (El / 8)` with `patch_size=8`:

| Profile | Tokens | Notes |
|---------|--------|-------|
| `256×48` | 192 | Canonical production default |
| `384×72` | 432 | Viable, moderate PPO cost increase |
| `512×96` | 768 | Viable, significant PPO cost |
| `768×144` | 1728 | OOM during self-attention on MX150 |

See [docs/RESOLUTION_BENCHMARKS.md](../../docs/RESOLUTION_BENCHMARKS.md).

---

## Default Port

| Port | Topics |
|------|--------|
| `5557` | `action_v2`, `telemetry_event_v2` (PUB) |

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
