# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first reinforcement-learning system for autonomous drone
navigation. Source scene meshes are compiled into compressed `.gmdag` signed
distance fields, traced at batch scale on the GPU by a CUDA sphere-tracing
kernel, and consumed by a sacred cognitive actor engine (RayViTEncoder →
Mamba-2 SSD → EpisodicMemory → PPO) that trains directly on the compiled
corpus with no intermediate graphics pipeline.

**Canonical runtime path:**

```
 Source Meshes (.glb/.obj/.bsp)
         │
    ┌────▼─────┐
    │ voxel-dag │  Offline: mesh → SDF → DAG deduplication → .gmdag
    └────┬─────┘
         │
    ┌────▼──────┐
    │ torch-sdf  │  Runtime: batched CUDA sphere tracing → PyTorch CUDA tensors
    └────┬──────┘
         │
    ┌────▼───────────────────────────────────────────────┐
    │ Unified In-Process PPO Trainer                      │
    │  RayViTEncoder → Mamba-2 SSD → EpisodicMemory → PPO│
    │  GPU-resident rollout storage, 4 parallel actors    │
    └────┬───────────────────────────────────────────────┘
         │
    ┌────▼──────┐
    │  Auditor   │  Passive dashboard subscribes to actor telemetry stream
    └───────────┘
```

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Configuration](#configuration)
- [Operational Flows](#operational-flows)
  - [Flow 1 — First-Time Setup](#flow-1--first-time-setup)
  - [Flow 2 — Corpus Preparation](#flow-2--corpus-preparation)
  - [Flow 3 — PPO Training (Canonical)](#flow-3--ppo-training-canonical)
  - [Flow 4 — Manual Training (Behavioral Cloning)](#flow-4--manual-training-behavioral-cloning)
  - [Flow 5 — Interactive Exploration](#flow-5--interactive-exploration)
  - [Flow 6 — Inference (3-Process Stack)](#flow-6--inference-3-process-stack)
  - [Flow 7 — Live Dashboard](#flow-7--live-dashboard)
  - [Flow 8 — Benchmarking & Comparison](#flow-8--benchmarking--comparison)
  - [Flow 9 — Validation & Qualification](#flow-9--validation--qualification)
  - [Flow 10 — Diagnostics & Analysis](#flow-10--diagnostics--analysis)
- [Complete Scripts Reference](#complete-scripts-reference)
- [CLI Entry Points](#cli-entry-points)
- [Wire Protocol](#wire-protocol)
- [Artifact Layout](#artifact-layout)
- [Repository Commands](#repository-commands)
- [Documentation Index](#documentation-index)
- [Performance Targets](#performance-targets)
- [License](#license)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Managed via `uv` |
| [uv](https://docs.astral.sh/uv/) | Latest | Package manager for all sub-projects |
| CUDA Toolkit | 12.1+ | Required for GPU training and sphere tracing |
| PyTorch | 2.5.1+ | Installed with CUDA support via `setup-actor-cuda.ps1` |
| C++ Compiler | MSVC 2022 / GCC 12+ | Required for `voxel-dag` and `torch-sdf` native builds |
| PowerShell | 5.1+ | All orchestration scripts are PowerShell |
| GNU Make | Any | Optional — cross-project convenience targets |

---

## Quick Start

```powershell
# 1. Install all project dependencies (contracts, environment, actor, auditor)
make sync-all

# 2. Install GPU-accelerated PyTorch into the actor environment
.\scripts\setup-actor-cuda.ps1

# 3. Download public scenes and compile the training corpus
.\scripts\refresh-scene-corpus.ps1

# 4. (Optional) Download 10 Quake 3 arena maps for diverse training
.\scripts\download-quake3-maps.ps1

# 5. Start canonical continuous training on the full corpus
.\scripts\train.ps1

# 6. (Optional) Attach a live dashboard to the running trainer
.\scripts\run-dashboard.ps1
```

---

## Project Layout

```
navi/
├── projects/
│   ├── contracts/      Wire-format models, serialization, ZMQ topics
│   ├── environment/    Headless sdfdag stepping, corpus preparation, .gmdag compiler
│   ├── actor/          Sacred cognitive engine, PPO trainer, policy checkpointing
│   ├── auditor/        Live dashboard, Zarr recording, session replay, explorer
│   ├── voxel-dag/      Offline mesh → .gmdag compiler (C++ / CUDA)
│   └── torch-sdf/      CUDA sphere-tracing kernel with PyTorch tensor I/O (C++ / CUDA)
├── scripts/            PowerShell orchestration & Python diagnostics
├── data/               Source scene assets and map manifests
├── artifacts/          All generated outputs (checkpoints, logs, corpus, benchmarks)
├── docs/               Architectural and operational documentation
├── logs/               Stable top-level log surface for operator tailing
├── tests/              Cross-project integration tests
└── tools/              Operational utilities (cleanup, observability)
```

Each project is a **sovereign package** with its own `pyproject.toml`, virtual
environment, and test suite. Cross-project imports only occur at CLI
orchestration boundaries — no service imports another service package.

---

## Configuration

All projects use `pydantic-settings` with a shared root `.env` file.

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NAVI_ENV_PUB_ADDRESS` | `tcp://localhost:5559` | Environment PUB socket |
| `NAVI_ENV_REP_ADDRESS` | `tcp://localhost:5560` | Environment REP socket |
| `NAVI_ACTOR_PUB_ADDRESS` | `tcp://localhost:5557` | Actor PUB socket (telemetry) |
| `NAVI_AZIMUTH_BINS` | `256` | Observation azimuth resolution |
| `NAVI_ELEVATION_BINS` | `48` | Observation elevation resolution |
| `NAVI_GMDAG_RESOLUTION` | `512` | Canonical `.gmdag` compile resolution |
| `NAVI_MAX_STEPS_PER_EPISODE` | `2000` | Hard truncation step limit |
| `NAVI_SCENE_EPISODES_PER_SCENE` | `16` | Episodes before scene rotation |
| `NAVI_MAX_DISTANCE` | `100.0` | Observation horizon (meters) |

### Observation Contract

The canonical observation is a `256×48` spherical distance matrix (azimuth × elevation bins).

| Profile | Tokens (patch=8) | Status |
|---------|-------------------|--------|
| `256×48` | 192 | **Production default** |
| `384×72` | 432 | Benchmark comparison |
| `512×96` | 768 | Benchmark comparison |
| `768×144` | Exceeds trainer VRAM | Environment-only viable |

### Temporal Core Selection

| Core | Description | Default? |
|------|-------------|----------|
| `mamba2` | Pure-PyTorch Mamba-2 SSD | **Yes** — best learning quality |
| `gru` | cuDNN GRU | Comparison — higher SPS, lower quality |
| `mambapy` | Pure-Python Mamba | Debugging only |

---

## Operational Flows

### Flow 1 — First-Time Setup

Complete environment preparation from a fresh clone.

```
Step 1: Install dependencies    →  make sync-all
Step 2: Install CUDA PyTorch    →  .\scripts\setup-actor-cuda.ps1
Step 3: Verify GPU              →  python scripts\check_gpu.py
Step 4: Build native extensions →  (auto-built by uv sync in voxel-dag and torch-sdf)
Step 5: Download scene corpus   →  .\scripts\refresh-scene-corpus.ps1
```

**Detailed instructions:**

```powershell
# Step 1 — Install all Python dependencies across all 4 sub-projects
make sync-all
# Equivalent manual: cd projects/contracts; uv sync; cd ../environment; uv sync; ...

# Step 2 — Install GPU-accelerated PyTorch (CUDA 12.1 wheels)
.\scripts\setup-actor-cuda.ps1
# Options:
#   -CudaTag cu124             # Use CUDA 12.4 wheels instead
#   -TorchVersion 2.6.0        # Specific PyTorch version
#   -SkipActorSync             # Skip uv sync after wheel install
#   -InstallFusedTemporal      # Install mamba-ssm fused kernels (experimental)

# Step 3 — Verify CUDA is working
python scripts\check_gpu.py
# Output: GPU name, compute capability, CUDA version, kernel execution test
# Exit 0 = OK, Exit 1 = no CUDA, Exit 2 = kernels fail (wrong torch build)

# Step 4 — Verify Embree ray tracing (optional, for mesh compilation speed)
python scripts\check_embree.py

# Step 5 — Download and compile the public scene corpus
.\scripts\refresh-scene-corpus.ps1
# Downloads from HuggingFace: Habitat test scenes + ReplicaCAD stages
# Compiles all to .gmdag at resolution 512
# Produces: artifacts/gmdag/corpus/<dataset>/*.gmdag + manifest
```

**Linux/WSL2 variant:**
```bash
# Step 2 alternative for Linux/WSL2
./scripts/setup-actor-cuda.sh
```

---

### Flow 2 — Corpus Preparation

The training corpus is a collection of compiled `.gmdag` scene files. Multiple
data sources are supported. All compilation uses resolution 512 by default.

#### 2A — Full Corpus Refresh (HuggingFace datasets)

Transactional pipeline: download → compile → validate → promote to live corpus.

```powershell
.\scripts\refresh-scene-corpus.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Datasets` | `"hssd/hssd-hab,ai-habitat/ReplicaCAD_dataset,..."` | Comma-separated HuggingFace dataset IDs |
| `-ScenesPerDataset` | `10` | Max scenes to download per dataset |
| `-Resolution` | `512` | `.gmdag` compile resolution |
| `-ForceRecompile` | (switch) | Recompile even if `.gmdag` exists |
| `-KeepScratch` | (switch) | Keep intermediate GLB downloads |
| `-IncludeQuake3` | (switch) | Include Quake 3 maps in the refresh |

```powershell
# Examples:
.\scripts\refresh-scene-corpus.ps1                              # Full default refresh
.\scripts\refresh-scene-corpus.ps1 -ScenesPerDataset 5          # Smaller subset
.\scripts\refresh-scene-corpus.ps1 -ForceRecompile              # Force recompile all
.\scripts\refresh-scene-corpus.ps1 -IncludeQuake3               # Include Q3 maps
```

**Outputs:** `artifacts/gmdag/corpus/<dataset>/*.gmdag` + `artifacts/gmdag/corpus/gmdag_manifest.json`

#### 2B — Habitat Test Scenes Only (Bootstrap)

Quick download of 3 test scenes + ReplicaCAD stages for initial setup.

```powershell
.\scripts\download-habitat-data.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-DataDir` | `data/scenes` | Download target directory |
| `-Datasets` | `"test_scenes,replicacad"` | Which datasets to fetch |
| `-PreserveExisting` | (switch) | Skip if files already exist |

#### 2C — ReplicaCAD Expansion (Incremental)

Add new ReplicaCAD baked-lighting scenes without disturbing the existing corpus.

```powershell
.\scripts\expand-replicacad-corpus.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-ScenesLimit` | `0` (all) | Max new scenes to download |
| `-Resolution` | `512` | Compile resolution |

#### 2D — Quake 3 Arena Maps

Download community Quake 3 maps from [lvlworld.com](https://lvlworld.com),
extract BSP geometry, and compile to `.gmdag`.

```powershell
.\scripts\download-quake3-maps.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-MapFilter` | `""` (all) | Comma-separated map names to download |
| `-Sources` | `"lvlworld"` | Source filter (`lvlworld`, `openarena`) |
| `-Resolution` | `512` | Compile resolution |
| `-Tessellation` | `4` | BSP bezier patch tessellation level |
| `-ForceRecompile` | (switch) | Overwrite existing `.gmdag` files |
| `-KeepIntermediate` | (switch) | Keep downloaded PK3 and intermediate OBJ |

```powershell
# Examples:
.\scripts\download-quake3-maps.ps1                                             # All maps from manifest
.\scripts\download-quake3-maps.ps1 -MapFilter "padshop,aerowalk,simpsons_map"  # Specific maps
.\scripts\download-quake3-maps.ps1 -ForceRecompile                             # Rebuild all
```

Map manifest: `data/quake3/quake3_map_manifest.json` (18 lvlworld maps defined).

**Current compiled Q3 maps (10):**

| Map | Author | Style | GMDAG Size |
|-----|--------|-------|------------|
| padshop | ENTE | Giant-scale furniture | 6.1 MB |
| japanese_castles | g1zm0 | Multi-layered architecture | 7.2 MB |
| substation11 | g1zm0 | Complex technical indoor | 3.9 MB |
| edge_of_forever | Sock | Dense interconnected (highest-rated) | 1.0 MB |
| rustgrad | Hipshot | Industrial machinery | 5.1 MB |
| simpsons_map | Maggu | Residential rooms & furniture | 5.1 MB |
| unholy_sanctuary | Martinus | Gothic corridors, vertical | 26.4 MB |
| chronophagia | Obsessed | Artistic sealed corridors | 5.3 MB |
| padkitchen | ENTE | Giant-scale kitchen | 7.1 MB |
| aerowalk | the Hubster | Tight tournament | 3.9 MB |

#### 2E — Single Asset Compilation

```powershell
# Compile one mesh file to .gmdag
uv run --project projects\environment navi-environment compile-gmdag `
    --source .\data\scenes\hssd\102343992.glb `
    --output .\artifacts\gmdag\corpus\hssd\102343992.gmdag `
    --resolution 512

# Validate a compiled asset
uv run --project projects\environment navi-environment check-sdfdag `
    --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# Benchmark a compiled asset
uv run --project projects\environment navi-environment bench-sdfdag `
    --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag --actors 4 --steps 200
```

---

### Flow 3 — PPO Training (Canonical)

Unified in-process training: the actor instantiates the `sdfdag` environment
backend directly, with GPU-resident rollout storage and no ZMQ in the hot loop.

#### 3A — Standard Training

```powershell
.\scripts\train.ps1
```

This is the primary training surface. When no scene or step limit is supplied,
training uses the **full discovered corpus** and runs **continuously until stopped**.

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
.\scripts\train.ps1                                                 # Full corpus, continuous
.\scripts\train.ps1 -TemporalCore gru                              # GRU temporal core
.\scripts\train.ps1 -TotalSteps 50000                               # Bounded 50K steps
.\scripts\train.ps1 -NumActors 8                                    # 8 parallel actors
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\latest.pt  # Resume training
.\scripts\train.ps1 -Scene .\data\scenes\hssd\102343992.glb -AutoCompileGmDag  # Single scene

# Direct CLI (bypasses wrapper):
uv run --project projects\actor navi-actor train --actors 4 --temporal-core mamba2
```

#### 3B — Ghost Stack Training (Orchestrated)

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
.\scripts\run-ghost-stack.ps1 -Train -WithDashboard                 # Train with live view
.\scripts\run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard       # 8 actors + dashboard
.\scripts\run-ghost-stack.ps1 -Train -Datasets "quake3-arenas"      # Q3 maps only
.\scripts\run-ghost-stack.ps1 -Train -TotalSteps 10000              # Bounded training
```

#### 3C — Overnight Training

Durable unattended training with checkpoint monitoring.

```powershell
.\scripts\train-all-night.ps1
```

Same parameters as `train.ps1`. Includes CUDA environment setup and process cleanup.
Runs until interrupted (Ctrl+C) or system shutdown.

#### 3D — Training Artifacts

Training produces the following artifacts:

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

### Flow 4 — Manual Training (Behavioral Cloning)

Human demonstration capture and supervised pre-training. Use this to bootstrap
a base policy before RL fine-tuning.

#### 4A — Automated Multi-Scene Loop

The recommended workflow: iterates through scenes, records your flight,
trains the model incrementally after each scene.

```powershell
.\scripts\run-manual-training.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-CorpusRoot` | `artifacts\gmdag\corpus` | Scene directory |
| `-Scenes` | `@()` (discover all) | Specific scene file paths |
| `-Checkpoint` | `""` | Starting checkpoint (empty = fresh) |
| `-CheckpointOutput` | `artifacts\checkpoints\bc_base_model.pt` | Output checkpoint |
| `-DemoDir` | `artifacts\demonstrations` | Demonstration storage |
| `-Epochs` | `30` | BC training epochs per scene |
| `-MaxSteps` | `1000` | Steps per scene (auto-close) |
| `-LearningRate` | `1e-3` | BC learning rate |
| `-LinearSpeed` | `1.5` | Flight speed (m/s) |
| `-YawRate` | `1.5` | Rotation speed (rad/s) |
| `-TemporalCore` | `""` | Temporal backend |

**Workflow per scene:**
1. Explorer opens with auto-recording (WASD flight controls)
2. Dashboard auto-closes after `MaxSteps` steps
3. BC training runs on all accumulated demonstrations
4. Updated checkpoint is saved and loaded for the next scene
5. Repeat until all scenes are completed

```powershell
# Examples:
.\scripts\run-manual-training.ps1                                   # Full corpus, fresh start
.\scripts\run-manual-training.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas  # Q3 only
.\scripts\run-manual-training.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt  # Resume
.\scripts\run-manual-training.ps1 -Scenes "artifacts\gmdag\corpus\quake3-arenas\aerowalk.gmdag"
```

#### 4B — Manual Steps (Individual Commands)

```powershell
# Step 1 — Record a demonstration
uv run --project projects\auditor explore `
    --record --gmdag-file .\artifacts\gmdag\corpus\quake3-arenas\aerowalk.gmdag

# Step 2 — Train from demonstrations
.\scripts\run-bc-pretrain.ps1
# Or directly:
uv run --project projects\actor brain bc-pretrain `
    --demonstrations artifacts\demonstrations `
    --output artifacts\checkpoints\bc_base_model.pt `
    --epochs 50

# Step 3 — Fine-tune the BC model with RL
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt
```

**BC Pre-Training Parameters** (`run-bc-pretrain.ps1`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Demonstrations` | `artifacts/demonstrations` | Path to `.npz` demo files |
| `-Output` | `artifacts/checkpoints/bc_base_model.pt` | Output checkpoint |
| `-Epochs` | `50` | Training epochs |
| `-LearningRate` | `1e-3` | Learning rate |
| `-BpttLen` | `8` | BPTT sequence length |
| `-MinibatchSize` | `32` | Minibatch size |

---

### Flow 5 — Interactive Exploration

Keyboard-controlled drone flight for scene inspection and demonstration recording.

```powershell
.\scripts\run-explore.ps1 -GmdagFile .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-GmdagFile` | (required) | Scene `.gmdag` file path |
| `-LinearSpeed` | `1.5` | Movement speed (m/s) |
| `-YawRate` | `1.5` | Rotation speed (rad/s) |

**Controls:**
- `W` / `S` — Forward / Backward
- `A` / `D` — Strafe Left / Right
- `Space` / `Shift` — Up / Down
- `Q` / `E` — Yaw Left / Right
- `ESC` — Close

```powershell
# Direct CLI (with recording):
uv run --project projects\auditor explore `
    --gmdag-file .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag `
    --record --max-steps 2000

# Without wrapper:
uv run --project projects\auditor navi-auditor explore `
    --gmdag-file .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag
```

---

### Flow 6 — Inference (3-Process Stack)

Run a trained policy in real-time with separate Environment, Actor, and
Dashboard processes communicating over ZMQ.

```powershell
.\scripts\run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\apartment_1.gmdag
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-GmDagFile` | (required for inference) | Scene to run |
| `-ActorPolicyCheckpoint` | `""` | Trained checkpoint to load |
| `-Foreground` | (switch) | Block instead of backgrounding |
| `-NoDashboard` | (switch) | Skip dashboard launch |

**Manual 3-process launch (for debugging):**

```powershell
# Terminal 1 — Environment server
.\scripts\run-environment.ps1 --mode step --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# Terminal 2 — Actor server
.\scripts\run-brain.ps1 --mode step --temporal-core mamba2

# Terminal 3 — Dashboard
.\scripts\run-dashboard.ps1
```

**Service launcher parameters:**

| Script | Key Parameters | Default |
|--------|----------------|---------|
| `run-environment.ps1` | `-Mode` (step/async), `-AzimuthBins`, `-ElevationBins` | step, 256, 48 |
| `run-brain.ps1` | `-Mode` (step/async), `-TemporalCore` | step, mamba2 |
| `run-dashboard.ps1` | (forwards all args) | passive mode |

---

### Flow 7 — Live Dashboard

The auditor dashboard is a passive observer. It subscribes to the actor PUB
telemetry stream and renders actor 0 observations at 5–10 Hz.

```powershell
# Attach to a running training session or inference stack
.\scripts\run-dashboard.ps1

# Direct CLI with explicit port
uv run --project projects\auditor navi-auditor dashboard `
    --actor-sub tcp://localhost:5557 --passive
```

**Dashboard behavior:**
- Always displays **actor 0** observations (no selector)
- Shows active actor count in the status bar
- Auto-detects mode: TRAINING / INFERENCE / OBSERVER
- Handles missing streams gracefully (shows WAITING state)
- During training: passive actor-only mode — does not open environment control paths

---

### Flow 8 — Benchmarking & Comparison

#### 8A — Temporal Core Comparison (End-to-End)

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

#### 8B — Temporal Kernel Microbenchmark

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

#### 8C — Resolution Scaling Benchmark

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

#### 8D — Actor Scaling Test

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

#### 8E — Attribution Matrix

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

### Flow 9 — Validation & Qualification

#### 9A — Canonical Stack Qualification

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

#### 9B — Nightly Validation (Overnight)

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

---

### Flow 10 — Diagnostics & Analysis

#### 10A — GPU Preflight

```powershell
python scripts\check_gpu.py      # CUDA availability + kernel test
python scripts\check_embree.py   # Embree ray tracing backend check
```

#### 10B — Corpus Diagnostics

```powershell
# Shallow header probe (fast — reads .gmdag headers only)
python scripts\diagnose_gmdag_corpus.py

# Deep DAG structural analysis (walks up to 2M nodes per scene)
python scripts\diagnose_gmdag_deep.py
```

Flags: non-finite values, extreme scales, degenerate scenes (all-void, all-surface),
low compression, coarse/fine voxels, far centers.

#### 10C — Asset Validation

```powershell
# Validate specific asset
uv run --project projects\environment navi-environment check-sdfdag `
    --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag

# Validate entire corpus root
uv run --project projects\environment navi-environment check-sdfdag `
    --gmdag-root .\artifacts\gmdag\corpus

# JSON output for automation
uv run --project projects\environment navi-environment check-sdfdag --json `
    --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
```

#### 10D — Training Log Summarization

```powershell
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1 -OutputJson .\artifacts\analysis\summary.json
```

Extracts per-step metrics (`sps`, `fwd_ms`, `env_ms`, `opt_ms`) and computes
mean/min/max statistics.

---

## Complete Scripts Reference

### Training Scripts

| Script | Purpose | Key Variations |
|--------|---------|----------------|
| `train.ps1` | Canonical PPO training wrapper | `-TemporalCore`, `-TotalSteps`, `-NumActors`, `-ResumeCheckpoint` |
| `train-all-night.ps1` | Durable overnight training | Same as `train.ps1`, runs until Ctrl+C |
| `run-ghost-stack.ps1` | Orchestrated stack launcher | `-Train` for training, omit for inference; `-WithDashboard` |
| `run-manual-training.ps1` | Multi-scene BC loop | `-CorpusRoot`, `-MaxSteps`, `-Checkpoint` |
| `run-bc-pretrain.ps1` | BC supervised pre-training | `-Demonstrations`, `-Epochs`, `-Output` |

### Corpus Scripts

| Script | Purpose | Key Variations |
|--------|---------|----------------|
| `refresh-scene-corpus.ps1` | Full transactional corpus refresh | `-Datasets`, `-ForceRecompile`, `-IncludeQuake3` |
| `download-habitat-data.ps1` | Bootstrap public Habitat scenes | `-Datasets`, `-PreserveExisting` |
| `expand-replicacad-corpus.ps1` | Incremental ReplicaCAD growth | `-ScenesLimit` |
| `download-quake3-maps.ps1` | Q3 arena download + compile | `-MapFilter`, `-ForceRecompile` |

### Service Scripts

| Script | Purpose | Key Variations |
|--------|---------|----------------|
| `run-environment.ps1` | Environment server | `-Mode` (step/async) |
| `run-brain.ps1` | Actor server | `-Mode`, `-TemporalCore` |
| `run-dashboard.ps1` | Passive dashboard | Forwards all args |
| `run-explore.ps1` | Interactive explorer | `-GmdagFile`, `-LinearSpeed`, `-YawRate` |

### Benchmark Scripts

| Script | Purpose | Key Variations |
|--------|---------|----------------|
| `run-temporal-compare.ps1` | End-to-end temporal comparison | `-TemporalCores`, `-Repeats`, `-TotalSteps` |
| `run-temporal-bakeoff.ps1` | Kernel microbenchmark | `-Candidates`, `-Batch`, `-Device` |
| `run-resolution-compare.ps1` | Resolution scaling sweep | `-Profiles`, `-Repeats` |
| `run-actor-scaling-test.ps1` | Fleet scaling benchmark | `-ActorCountsStr`, `-StepsPerRun` |
| `run-attribution-matrix.ps1` | Throughput attribution ablation | `-TotalSteps`, `-Actors` |

### Validation Scripts

| Script | Purpose | Key Variations |
|--------|---------|----------------|
| `qualify-canonical-stack.ps1` | Stack qualification | `-TotalSteps`, `-SkipDatasetAudit` |
| `run-nightly-validation.ps1` | Overnight validation | `-SoakHours`, `-QualificationSteps` |

### Setup & Diagnostics

| Script | Purpose |
|--------|---------|
| `setup-actor-cuda.ps1` | Install CUDA PyTorch wheels (Windows) |
| `setup-actor-cuda.sh` | Install CUDA PyTorch wheels (Linux/WSL2) |
| `check_gpu.py` | CUDA availability + kernel execution test |
| `check_embree.py` | Embree ray tracing backend detection |
| `diagnose_gmdag_corpus.py` | Shallow `.gmdag` header diagnostics |
| `diagnose_gmdag_deep.py` | Deep DAG structural analysis |
| `summarize-bounded-train-log.ps1` | Training log metric extraction |
| `run-json-command.py` | Machine-readable subprocess runner |
| `run-structured-surface.py` | Diagnostic surface orchestrator |
| `benchmark_canonical_stack.py` | End-to-end throughput proof (CI gate) |

---

## CLI Entry Points

Each project exposes CLI commands through `uv run`:

### Environment (`projects/environment`)

```powershell
uv run --project projects\environment navi-environment <command>

Commands:
  serve              Start the environment server
  compile-gmdag      Compile a mesh to .gmdag
  prepare-corpus     Full corpus preparation
  check-sdfdag       Validate runtime + assets
  bench-sdfdag       Benchmark compiled assets
  dataset-audit      Audit available datasets
```

**Shortcut:** `uv run --project projects\environment environment` → `serve`

### Actor (`projects/actor`)

```powershell
uv run --project projects\actor navi-actor <command>

Commands:
  serve              Start the actor policy server
  train              Unified in-process PPO training
  evaluate           Run inference evaluation
  profile            Throughput profiling

uv run --project projects\actor brain <subcommand>

Subcommands:
  bc-pretrain        Behavioral cloning pre-training
```

**Shortcut:** `uv run --project projects\actor brain` → actor CLI

### Auditor (`projects/auditor`)

```powershell
uv run --project projects\auditor navi-auditor <command>

Commands:
  dashboard          Live passive observation dashboard
  explore            Interactive keyboard-controlled explorer
  record             Session recording
  replay             Session playback
  dataset-audit      Dataset audit surface
```

**Shortcuts:**
- `uv run --project projects\auditor dashboard` → `navi-auditor dashboard`
- `uv run --project projects\auditor explore` → `navi-auditor explore`

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

### Default Network Ports

| Port | Service | Role |
|------|---------|------|
| `5559` | Environment | PUB (observation broadcast) |
| `5560` | Environment | REP (step request/response) |
| `5557` | Actor | PUB (action + telemetry broadcast) |

Ports are configurable via `.env` or CLI parameters. The unified trainer
only uses port `5557` for the passive dashboard telemetry stream.

---

## Artifact Layout

```
artifacts/
├── gmdag/
│   └── corpus/                     Compiled .gmdag scene files
│       ├── gmdag_manifest.json     Master corpus manifest
│       ├── ai-habitat_habitat_test_scenes/
│       ├── ai-habitat_ReplicaCAD_baked_lighting/
│       └── quake3-arenas/          10 compiled Q3 maps
├── checkpoints/                    Stable model checkpoints
│   └── bc_base_model.pt           BC pre-trained base model
├── demonstrations/                 Human flight recordings (.npz)
├── runs/<run_id>/                  Per-run governed outputs
│   ├── logs/                       Process logs
│   ├── metrics/                    Machine-readable metrics
│   ├── manifests/                  Process manifests
│   ├── checkpoints/                Run-scoped checkpoints
│   └── reports/                    Summaries
├── benchmarks/                     Benchmark results
│   ├── temporal-compare/
│   ├── temporal/
│   ├── resolution-compare/
│   ├── actor-scaling/
│   └── attribution-matrix/
├── qualification/                  Qualification results
│   └── canonical_stack/<run_id>/
├── nightly/<run_id>/               Nightly validation runs
├── validation/                     Validation artifacts
├── analysis/                       Post-hoc analysis outputs
└── logs/                           Overflow/archive logs

logs/                               Stable top-level log surface
├── navi_actor_train.log.*          Actor training (rotating, 1MB × 10)
└── navi_auditor_dashboard.log.*    Dashboard (rotating, 1MB × 10)
```

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

## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, runtime boundaries, layer responsibilities |
| [docs/TRAINING.md](docs/TRAINING.md) | Corpus refresh, training algorithms, resume, recovery |
| [docs/ACTOR.md](docs/ACTOR.md) | Sacred cognitive engine: RayViT → Mamba-2 → Episodic Memory → PPO |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Environment runtime, kinematics, reward shaping |
| [docs/SDFDAG_RUNTIME.md](docs/SDFDAG_RUNTIME.md) | SDF/DAG backend: batched sphere tracing, tensor contracts |
| [docs/GMDAG.md](docs/GMDAG.md) | `.gmdag` binary format specification |
| [docs/COMPILER.md](docs/COMPILER.md) | Voxel-DAG compiler internals |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | Wire-format contract specification (v2) |
| [docs/DATAFLOW.md](docs/DATAFLOW.md) | End-to-end data flow diagrams |
| [docs/AUDITOR.md](docs/AUDITOR.md) | Dashboard, recording, replay, demonstration capture |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Throughput targets, profiling, bottleneck analysis |
| [docs/RESOLUTION_BENCHMARKS.md](docs/RESOLUTION_BENCHMARKS.md) | Observation-resolution sweep results |
| [docs/COMPARISON.md](docs/COMPARISON.md) | Temporal-core comparison results (mamba2 vs gru) |
| [docs/NIGHTLY_VALIDATION.md](docs/NIGHTLY_VALIDATION.md) | Nightly validation pipeline specification |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | SDF/DAG validation standard |
| [docs/PARALLEL.md](docs/PARALLEL.md) | Parallel architecture notes |
| [docs/TSDF.md](docs/TSDF.md) | Legacy TSDF reference |
| [AGENTS.md](AGENTS.md) | Implementation policy, non-negotiables, architectural standards |

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Rollout throughput (current HW) | ~1,000 SPS | In progress |
| Rollout throughput (advanced HW) | 10,000 SPS | Planned |
| Inference latency (CPU) | ≤ 15 ms/actor | **Achieved** |
| Environment latency (4 actors) | ≤ 25 ms | **Achieved** |

The canonical `256×48` observation contract is the production default. Higher
profiles are benchmark-viable but limited by RayViT self-attention and PPO update
cost. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full analysis.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use,
modify, and distribute this software, provided the original copyright notice and
license text are included in all copies or substantial portions of the software.
