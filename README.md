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
  - [Flow 6 — In-Process Inference (Canonical)](#flow-6--in-process-inference-canonical)
  - [Flow 7 — Legacy Inference (3-Process Stack)](#flow-7--legacy-inference-3-process-stack)
  - [Flow 8 — Live Dashboard](#flow-8--live-dashboard)
  - [Flow 9 — Benchmarking & Comparison](#flow-9--benchmarking--comparison)
  - [Flow 10 — Validation & Qualification](#flow-10--validation--qualification)
  - [Flow 11 — Diagnostics & Analysis](#flow-11--diagnostics--analysis)
- [Complete Scripts Reference](#complete-scripts-reference)
  - [Training Scripts](#training-scripts)
  - [Corpus Scripts](#corpus-scripts)
  - [Service Scripts](#service-scripts)
  - [Benchmark Scripts](#benchmark-scripts)
  - [Validation Scripts](#validation-scripts)
  - [Setup & Diagnostics](#setup--diagnostics)
- [CLI Entry Points](#cli-entry-points)
  - [Environment](#environment-projectsenvironment)
  - [Actor](#actor-projectsactor)
  - [Auditor](#auditor-projectsauditor)
  - [Voxel-DAG Compiler](#voxel-dag-compiler-projectsvoxel-dag)
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
See also: [`setup-actor-cuda.ps1`](#setup-actor-cudaps1----install-cuda-pytorch-windows) | [`refresh-scene-corpus.ps1`](#refresh-scene-corpusps1----full-transactional-corpus-refresh)

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
See also: [Corpus Scripts](#corpus-scripts) | [`compile-gmdag`](#compile-gmdag----compile-mesh-to-gmdag)

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
See also: [Training Scripts](#training-scripts) | [`navi-actor train`](#train----unified-in-process-ppo-training)

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
See also: [`run-explore-scenes.ps1`](#run-explore-scenesps1----multi-scene-navigation--demo-recording) | [`run-bc-pretrain.ps1`](#run-bc-pretrainps1----behavioral-cloning-pre-training) | [`bc-pretrain`](#bc-pretrain----behavioral-cloning-pre-training)

The workflow is split into two independent phases so you can fly through as many
scenes as you want without waiting for training between each one.

#### 4A — Phase 1: Navigate Scenes (Collect Demonstrations)

Fly through scenes continuously. Each scene auto-closes after `MaxSteps`
and the next one opens immediately. Demonstrations accumulate in `DemoDir`.

```powershell
.\scripts\run-explore-scenes.ps1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-CorpusRoot` | `artifacts\gmdag\corpus` | Scene directory |
| `-Scenes` | `@()` (discover all) | Specific scene file paths |
| `-DemoDir` | `artifacts\demonstrations` | Demonstration storage |
| `-MaxSteps` | `1000` | Steps per scene (auto-close) |
| `-LinearSpeed` | `1.5` | Flight speed (m/s) |
| `-YawRate` | `1.5` | Rotation speed (rad/s) |

```powershell
# Examples:
.\scripts\run-explore-scenes.ps1                                    # Full corpus
.\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
.\scripts\run-explore-scenes.ps1 -MaxSteps 2000                     # Longer sessions
.\scripts\run-explore-scenes.ps1 -Scenes "aerowalk.gmdag","padshop.gmdag"
```

**Per-scene flow:**
1. Explorer opens with auto-recording (WASD / arrow keys)
2. Dashboard auto-closes after `MaxSteps` steps (or press ESC/Q to skip)
3. Demo `.npz` is saved to `DemoDir`
4. Next scene opens immediately — no training wait
5. Repeat until all scenes are done, or Ctrl+C to stop

#### 4B — Phase 2: Train on Demonstrations

When you have enough demonstrations, train a BC checkpoint:

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
.\scripts\run-bc-pretrain.ps1                                        # Fresh start
.\scripts\run-bc-pretrain.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt  # Resume
.\scripts\run-bc-pretrain.ps1 -Epochs 100 -LearningRate 5e-4        # Custom hyperparams
```

#### 4C — Fine-Tune the BC Model with RL

Once you have a BC checkpoint, promote it to RL training:

```powershell
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt
```

The BC checkpoint is a standard v2 file directly compatible with the canonical
PPO trainer.

#### 4D — Raw CLI Commands

```powershell
# Record a single scene
uv run --project projects\auditor explore `
    --record --gmdag-file .\artifacts\gmdag\corpus\quake3-arenas\aerowalk.gmdag

# Train from all accumulated demos
uv run --project projects\actor brain bc-pretrain `
    --demonstrations artifacts\demonstrations `
    --output artifacts\checkpoints\bc_base_model.pt `
    --epochs 50

# Fine-tune with RL
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt
```

---

### Flow 5 — Interactive Exploration

Keyboard-controlled drone flight for scene inspection and demonstration recording.
See also: [`run-explore.ps1`](#run-exploreps1----single-scene-interactive-explorer) | [`explore`](#explore----interactive-keyboard-controlled-explorer)

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

### Flow 6 — In-Process Inference (Canonical)

Evaluate a trained policy checkpoint with direct in-process CUDA stepping.
Same architecture as training (SdfDagBackend, tensor-native, ZMQ telemetry)
but without PPO, rollout buffers, reward shaping, or episodic memory.
See also: [`docs/INFERENCE.md`](docs/INFERENCE.md) | [`run-inference.ps1`](#run-inferenceps1----in-process-inference) | [`navi-actor infer`](#infer----in-process-policy-evaluation)

#### 6A — Ghost Stack Inference

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
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -Deterministic  # Deterministic
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -TotalSteps 10000  # Bounded
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -NoDashboard  # Headless
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -Datasets "quake3-arenas"  # Subset
```

#### 6B — Standalone Inference Wrapper

```powershell
.\scripts\run-inference.ps1 -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt"
```

Same parameters as ghost stack inference. Runs inference directly without
orchestration overhead.

#### 6C — Direct CLI

```powershell
uv run --project projects\actor navi-actor infer `
    --checkpoint .\artifacts\checkpoints\bc_base_model.pt `
    --actors 4 --deterministic --total-steps 10000
```

---

### Flow 7 — Legacy Inference (3-Process Stack)

Run a trained policy in real-time with separate Environment, Actor, and
Dashboard processes communicating over ZMQ. This is the legacy multi-process
mode; prefer [Flow 6](#flow-6--in-process-inference-canonical) for new work.
See also: [Service Scripts](#service-scripts) | [`run-ghost-stack.ps1`](#run-ghost-stackps1----orchestrated-stack-launcher)

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

### Flow 8 — Live Dashboard

The auditor dashboard is a passive observer. It subscribes to the actor PUB
telemetry stream and renders actor 0 observations at 5-10 Hz.
The dashboard uses a split-socket architecture: a dedicated `zmq.CONFLATE`
socket ensures the displayed observation is always the latest published frame,
while a separate telemetry socket delivers ordered metrics and actions.
See also: [`run-dashboard.ps1`](#run-dashboardps1----passive-observation-dashboard) | [`dashboard`](#dashboard----live-passive-observation-dashboard)

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
- Shows observation freshness (`Obs=XXms`) in the status metrics line
- Auto-detects mode: TRAINING / INFERENCE / OBSERVER
- Handles missing streams gracefully (shows WAITING state)
- During training: passive actor-only mode — does not open environment control paths
- Renders only when a new observation arrives; UI ticks without fresh data
  update status metrics without wasting CPU on redundant rendering

---

### Flow 9 — Benchmarking & Comparison

See also: [Benchmark Scripts](#benchmark-scripts)

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

### Flow 10 — Validation & Qualification

See also: [Validation Scripts](#validation-scripts)

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

### Flow 11 — Diagnostics & Analysis

See also: [Setup & Diagnostics](#setup--diagnostics) | [`check-sdfdag`](#check-sdfdag----validate-runtime--assets)

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

All scripts live in `scripts/` and are invoked from the repository root.
Each entry shows the full parameter surface with types, defaults, and the
underlying CLI command the wrapper calls.

> **Convention:** Parameters without a default are required.
> `(switch)` parameters are boolean flags (supply to enable, omit to disable).

---

### Training Scripts

#### `train.ps1` -- Standard PPO Training

Canonical RL training wrapper. Full corpus, continuous by default. [(Flow 3A)](#3a--standard-training)

**Wraps:** `uv run --project projects/actor brain train`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-NumActors` | int | `4` | Parallel actor count |
| `-TemporalCore` | string | `mamba2` | Temporal backend (`mamba2` / `gru` / `mambapy`) |
| `-TotalSteps` | int | `0` | Steps before auto-stop (0 = continuous) |
| `-RolloutLength` | int | `512` | Steps per rollout before PPO update |
| `-MinibatchSize` | int | `64` | PPO minibatch size |
| `-PpoEpochs` | int | `1` | PPO optimizer epochs per update |
| `-LearningRate` | float | `5e-4` | Learning rate |
| `-EntropyCoeff` | float | `0.02` | Entropy regularization weight |
| `-ExistentialTax` | float | `-0.02` | Per-step existence penalty |
| `-BpttLen` | int | `8` | Truncated BPTT sequence length |
| `-CheckpointEvery` | int | `25000` | Steps between checkpoint saves |
| `-CheckpointDir` | string | `""` | Custom checkpoint directory |
| `-ResumeCheckpoint` | string | `""` | Resume from existing checkpoint path |
| `-Scene` | string | `""` | Override single scene GLB file |
| `-Manifest` | string | `""` | Corpus manifest path |
| `-CorpusRoot` | string | `""` | Custom corpus root directory |
| `-GmDagRoot` | string | `""` | Custom .gmdag root directory |
| `-GmDagFile` | string | `""` | Single .gmdag file override |
| `-AutoCompileGmDag` | switch | | Auto-compile scene to .gmdag |
| `-GmDagResolution` | int | `512` | Compile resolution for auto-compile |
| `-AzimuthBins` | int | `256` | Observation azimuth resolution |
| `-ElevationBins` | int | `48` | Observation elevation resolution |
| `-ActorTelemetryPort` | int | `5557` | Actor PUB port for dashboard |
| `-LogDir` | string | `""` | Custom log directory |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\train.ps1                                                    # Full corpus, continuous
.\scripts\train.ps1 -TotalSteps 50000                                  # Bounded run
.\scripts\train.ps1 -TemporalCore gru -NumActors 8                    # GRU with 8 actors
.\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\latest.pt  # Resume training
```

---

#### `train-all-night.ps1` -- Overnight Continuous Training

Durable overnight training with process management. Same parameters as `train.ps1`.
Includes CUDA env setup and graceful cleanup on Ctrl+C. [(Flow 3C)](#3c--overnight-training)

**Wraps:** `uv run --project projects/actor brain train`

All parameters from `train.ps1` above apply identically.

```powershell
.\scripts\train-all-night.ps1                                          # Continuous overnight
.\scripts\train-all-night.ps1 -ResumeCheckpoint artifacts\checkpoints\latest.pt
```

---

#### `run-ghost-stack.ps1` -- Orchestrated Stack Launcher

Full-stack launcher: training (`-Train`), in-process inference (`-Infer`),
or legacy 3-process inference (default).
Handles process lifecycle, pre-kill, and optional dashboard.
[(Flow 3B)](#3b--ghost-stack-training-orchestrated)
[(Flow 6)](#flow-6--in-process-inference-canonical)
[(Flow 7)](#flow-7--legacy-inference-3-process-stack)

**Wraps (training):** `uv run --project projects/actor brain train`
**Wraps (inference):** `uv run --project projects/actor brain infer`
**Wraps (legacy):** 3-process stack (environment + actor + dashboard)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Train` | switch | | Enable unified PPO training mode |
| `-Infer` | switch | | Enable in-process inference mode |
| `-Deterministic` | switch | | Use action mean (inference only) |
| `-TotalEpisodes` | int | `0` | Episode limit (0 = unlimited, inference) |
| `-Actors` | int | `4` | Parallel actor count (training) or env actors (inference) |
| `-TotalSteps` | int | `0` | Training step limit (0 = continuous) |
| `-CheckpointEvery` | int | `25000` | Checkpoint interval (training) |
| `-CheckpointDir` | string | `""` | Checkpoint directory |
| `-Checkpoint` | string | `""` | Resume from checkpoint path |
| `-TemporalCore` | string | `mamba2` | Temporal backend |
| `-Scene` | string | `""` | Specific scene GLB file |
| `-Manifest` | string | `""` | Corpus manifest path |
| `-CorpusRoot` | string | `""` | Corpus root directory |
| `-GmDagRoot` | string | `""` | Custom .gmdag root |
| `-GmDagFile` | string | `""` | Single .gmdag file path |
| `-Datasets` | string | `""` | Include only these datasets |
| `-ExcludeDatasets` | string | `""` | Exclude datasets by name |
| `-AutoCompileGmDag` | switch | | Auto-compile scene to .gmdag |
| `-GmDagResolution` | int | `512` | Compile resolution |
| `-AzimuthBins` | int | `256` | Observation azimuth resolution |
| `-ElevationBins` | int | `48` | Observation elevation resolution |
| `-WithDashboard` | switch | | Auto-attach passive dashboard |
| `-NoDashboard` | switch | | Suppress dashboard launch |
| `-NoPreKill` | switch | | Skip stale process cleanup |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
# Training:
.\scripts\run-ghost-stack.ps1 -Train                                   # Train on full corpus
.\scripts\run-ghost-stack.ps1 -Train -WithDashboard                    # Train with live view
.\scripts\run-ghost-stack.ps1 -Train -Datasets "quake3-arenas"         # Q3 maps only
# In-process inference:
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint .\model.pt            # Infer on full corpus
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint .\model.pt -Deterministic  # Deterministic
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint .\model.pt -NoDashboard    # Headless
# Legacy inference:
.\scripts\run-ghost-stack.ps1 -GmDagFile .\scene.gmdag                 # 3-process inference
```

---

#### `run-inference.ps1` -- In-Process Inference

Standalone wrapper for in-process policy evaluation.
Same parameters as ghost stack `-Infer` mode. [(Flow 6B)](#6b--standalone-inference-wrapper)

**Wraps:** `uv run --project projects/actor brain infer`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Checkpoint` | string | (required) | Trained model checkpoint path |
| `-Actors` | int | `4` | Parallel actor count |
| `-Deterministic` | switch | | Use action mean instead of sampling |
| `-TotalSteps` | int | `0` | Step limit (0 = unlimited) |
| `-TotalEpisodes` | int | `0` | Episode limit (0 = unlimited) |
| `-TemporalCore` | string | `mamba2` | Temporal backend |
| `-Datasets` | string | `""` | Include only these datasets |
| `-ExcludeDatasets` | string | `""` | Exclude datasets by name |

```powershell
.\scripts\run-inference.ps1 -Checkpoint .\artifacts\checkpoints\bc_base_model.pt
.\scripts\run-inference.ps1 -Checkpoint .\model.pt -Deterministic -TotalSteps 10000
```

---

#### `run-explore-scenes.ps1` -- Multi-Scene Navigation + Demo Recording

Navigate all corpus scenes continuously. Each scene auto-closes after MaxSteps;
next scene opens immediately. Demos accumulate in DemoDir. [(Flow 4A)](#4a--phase-1-navigate-scenes-collect-demonstrations)

**Wraps:** `uv run --project projects/auditor explore --record`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-CorpusRoot` | string | `artifacts\gmdag\corpus` | Scene directory to discover |
| `-Scenes` | string[] | `@()` (discover all) | Specific scene file paths |
| `-DemoDir` | string | `artifacts\demonstrations` | Demonstration output directory |
| `-MaxSteps` | int | `1000` | Steps per scene before auto-close |
| `-LinearSpeed` | float | `1.5` | Flight speed (m/s) |
| `-YawRate` | float | `1.5` | Rotation speed (rad/s) |

```powershell
.\scripts\run-explore-scenes.ps1                                       # Full corpus
.\scripts\run-explore-scenes.ps1 -MaxSteps 2000                        # Longer sessions
.\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
```

---

#### `run-bc-pretrain.ps1` -- Behavioral Cloning Pre-Training

Supervised learning from recorded `.npz` demos. Trains on ALL demos found in
the demonstrations folder. [(Flow 4B)](#4b--phase-2-train-on-demonstrations)

**Wraps:** `uv run --project projects/actor brain bc-pretrain`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Demonstrations` | string | `artifacts/demonstrations` | Path to `.npz` demo files |
| `-Output` | string | `artifacts/checkpoints/bc_base_model.pt` | Output checkpoint path |
| `-Checkpoint` | string | `""` | Resume from existing BC checkpoint |
| `-Epochs` | int | `50` | Training epochs |
| `-LearningRate` | float | `1e-3` | Learning rate |
| `-BpttLen` | int | `8` | BPTT sequence length |
| `-MinibatchSize` | int | `32` | Minibatch size |
| `-TemporalCore` | string | `""` | Temporal backend override |

```powershell
.\scripts\run-bc-pretrain.ps1                                          # Fresh start
.\scripts\run-bc-pretrain.ps1 -Epochs 100 -LearningRate 5e-4          # Custom hyperparams
.\scripts\run-bc-pretrain.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt  # Resume
```

---

### Corpus Scripts

#### `refresh-scene-corpus.ps1` -- Full Transactional Corpus Refresh

Download from HuggingFace → compile → validate → promote to live corpus.
Staged transaction: stale data is only replaced after successful rebuild.
[(Flow 2A)](#2a--full-corpus-refresh-huggingface-datasets)

**Wraps:** HuggingFace API + `uv run navi-environment compile-gmdag`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-DataDir` | string | `data/scenes` | Source data directory |
| `-Datasets` | string | `"hssd/hssd-hab,ai-habitat/ReplicaCAD_dataset,..."` | Comma-separated HuggingFace dataset IDs |
| `-ScenesPerDataset` | int | `10` | Max scenes to download per dataset |
| `-CorpusRoot` | string | `artifacts/gmdag/corpus` | Live corpus root |
| `-GmDagRoot` | string | `artifacts/gmdag` | Parent .gmdag directory |
| `-ScratchRoot` | string | `artifacts/tmp/corpus-refresh` | Staging directory |
| `-Resolution` | int | `512` | .gmdag compile resolution |
| `-MinSceneBytes` | int | `100000` | Minimum scene file size filter |
| `-ForceRecompile` | switch | | Recompile all even if .gmdag exists |
| `-KeepScratch` | switch | | Keep intermediate GLB downloads |
| `-IncludeQuake3` | switch | | Include Quake 3 maps in the refresh |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\refresh-scene-corpus.ps1                                     # Full default refresh
.\scripts\refresh-scene-corpus.ps1 -ScenesPerDataset 5                 # Smaller subset
.\scripts\refresh-scene-corpus.ps1 -ForceRecompile -IncludeQuake3      # Full rebuild with Q3
```

---

#### `download-habitat-data.ps1` -- Bootstrap Habitat Scenes

Quick download of public Habitat test scenes + ReplicaCAD stages.
[(Flow 2B)](#2b--habitat-test-scenes-only-bootstrap)

**Wraps:** HuggingFace API downloads

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-DataDir` | string | `data/scenes` | Download target directory |
| `-Datasets` | string | `"test_scenes,replicacad"` | Which datasets to fetch |
| `-PreserveExisting` | switch | | Skip if files already exist |

```powershell
.\scripts\download-habitat-data.ps1                                    # Default bootstrap
.\scripts\download-habitat-data.ps1 -Datasets "test_scenes"           # Test scenes only
```

---

#### `expand-replicacad-corpus.ps1` -- Incremental ReplicaCAD Growth

Add new ReplicaCAD baked-lighting scenes without disturbing existing corpus.
[(Flow 2C)](#2c--replicacad-expansion-incremental)

**Wraps:** HuggingFace API + `uv run navi-environment compile-gmdag`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-ScenesLimit` | int | `0` (all) | Maximum new scenes to download |
| `-Resolution` | int | `512` | Compile resolution |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\expand-replicacad-corpus.ps1                                 # All available scenes
.\scripts\expand-replicacad-corpus.ps1 -ScenesLimit 20                 # First 20 only
```

---

#### `download-quake3-maps.ps1` -- Quake 3 Arena Download + Compile

Download Q3 community maps, extract BSP geometry, compile to .gmdag.
[(Flow 2D)](#2d--quake-3-arena-maps)

**Wraps:** `uv run voxel-dag bsp-to-obj` + `uv run navi-environment compile-gmdag`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-OutputRoot` | string | `artifacts/gmdag/corpus/quake3-arenas` | Output directory |
| `-Resolution` | int | `512` | Voxel-DAG compile resolution |
| `-MapFilter` | string | `""` (all) | Comma-separated map names |
| `-Sources` | string | `"lvlworld"` | Data sources (`lvlworld`, `openarena`) |
| `-Tessellation` | int | `4` | BSP bezier patch tessellation level |
| `-ForceRecompile` | switch | | Overwrite existing .gmdag files |
| `-KeepIntermediate` | switch | | Keep downloaded PK3 and intermediate OBJ |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\download-quake3-maps.ps1                                     # All maps from manifest
.\scripts\download-quake3-maps.ps1 -MapFilter "padshop,aerowalk"       # Specific maps
.\scripts\download-quake3-maps.ps1 -ForceRecompile                     # Rebuild all
```

---

### Service Scripts

These launch individual ZMQ services for the 3-process inference stack.
See [(Flow 7)](#flow-7--legacy-inference-3-process-stack) for the full startup sequence.

#### `run-environment.ps1` -- Environment Server

**Wraps:** `uv run --project projects/environment navi-environment serve`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Mode` | string | `step` | Service mode (`step` / `async`) |
| `-AzimuthBins` | int | `256` | Observation azimuth resolution |
| `-ElevationBins` | int | `48` | Observation elevation resolution |
| `-PythonVersion` | string | `3.12` | Python version for uv |
| `$ForwardArgs` | remaining | | Additional args passed to CLI |

```powershell
.\scripts\run-environment.ps1
.\scripts\run-environment.ps1 -Mode async
.\scripts\run-environment.ps1 -- --gmdag-file .\scene.gmdag
```

---

#### `run-brain.ps1` -- Actor Policy Server

**Wraps:** `uv run --project projects/actor navi-actor serve`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Mode` | string | `step` | Service mode (`step` / `async`) |
| `-TemporalCore` | string | `mamba2` | Temporal backend (`mamba2` / `gru` / `mambapy`) |
| `-AzimuthBins` | int | `256` | Observation azimuth resolution |
| `-ElevationBins` | int | `48` | Observation elevation resolution |
| `$ForwardArgs` | remaining | | Additional args passed to CLI |

```powershell
.\scripts\run-brain.ps1
.\scripts\run-brain.ps1 -TemporalCore gru
.\scripts\run-brain.ps1 -- --policy-checkpoint .\model.pt
```

---

#### `run-dashboard.ps1` -- Passive Observation Dashboard

Subscribes to actor PUB stream. Displays actor 0 observations in real time.
Mode auto-detected (TRAINING / INFERENCE / OBSERVER). [(Flow 8)](#flow-8--live-dashboard)

**Wraps:** `uv run --project projects/auditor navi-auditor dashboard`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-PythonVersion` | string | `3.12` | Python version for uv |
| `$ForwardArgs` | remaining | | Additional args passed to CLI |

```powershell
.\scripts\run-dashboard.ps1                                            # Attach to default port
.\scripts\run-dashboard.ps1 -- --actor-sub tcp://localhost:5557        # Explicit port
```

---

#### `run-explore.ps1` -- Single-Scene Interactive Explorer

Standalone manual exploration: environment backend + dashboard with keyboard navigation.
No training process required. [(Flow 5)](#flow-5--interactive-exploration)

**Wraps:** `uv run --project projects/auditor navi-auditor explore`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-GmdagFile` | string | | Scene .gmdag file to explore |
| `-LinearSpeed` | float | `1.5` | Movement speed (m/s) |
| `-YawRate` | float | `1.5` | Rotation speed (rad/s) |
| `-PythonVersion` | string | `3.12` | Python version for uv |
| `$ForwardArgs` | remaining | | Additional args passed to CLI |

```powershell
.\scripts\run-explore.ps1 -GmdagFile .\artifacts\gmdag\corpus\quake3-arenas\padshop.gmdag
.\scripts\run-explore.ps1 -GmdagFile .\scene.gmdag -LinearSpeed 3.0
```

---

### Benchmark Scripts

#### `run-temporal-compare.ps1` -- End-to-End Temporal Comparison

Compare temporal cores in real training runs. [(Flow 8A)](#8a--temporal-core-comparison-end-to-end)

**Wraps:** `uv run --project projects/actor brain train` (one run per core)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-TemporalCores` | string[] | `@("mamba2")` | Cores to compare |
| `-TotalSteps` | int | `256` | Steps per run |
| `-Repeats` | int | `3` | Independent repetitions |
| `-Actors` | int | `4` | Actor count |
| `-MinibatchSize` | int | `64` | PPO minibatch size |
| `-RolloutLength` | int | `256` | Rollout length |
| `-ProfileCudaEvents` | switch | | Enable CUDA event tracing |
| `-OutputRoot` | string | `artifacts/benchmarks/temporal-compare` | Results directory |
| `-Scene` / `-GmDagFile` | string | `""` | Scene overrides |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\run-temporal-compare.ps1 -TemporalCores @("mamba2","gru") -TotalSteps 1024
```

---

#### `run-temporal-bakeoff.ps1` -- Temporal Kernel Microbenchmark

Isolated forward/backward pass timing (no training overhead). [(Flow 8B)](#8b--temporal-kernel-microbenchmark)

**Wraps:** `projects/actor/scripts/bench_temporal_backends.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Candidates` | string[] | `@("mamba2")` | Backends to benchmark |
| `-Batch` | int | `16` | Batch size |
| `-SeqLen` | int | `128` | Sequence length |
| `-DModel` | int | `128` | Model dimension |
| `-Repeats` | int | `40` | Benchmark iterations |
| `-Warmup` | int | `10` | Warmup iterations |
| `-Device` | string | `cuda` | Device (`cuda` / `cpu`) |
| `-AllowCpuDiagnostic` | switch | | Allow CPU-side profiling |
| `-OutputDir` | string | `artifacts/benchmarks/temporal` | JSON output directory |

```powershell
.\scripts\run-temporal-bakeoff.ps1 -Candidates @("mamba2","gru","mambapy")
.\scripts\run-temporal-bakeoff.ps1 -Device cpu -AllowCpuDiagnostic
```

---

#### `run-resolution-compare.ps1` -- Resolution Scaling Benchmark

Compare training throughput across observation resolutions. [(Flow 8C)](#8c--resolution-scaling-benchmark)

**Wraps:** `uv run --project projects/actor brain train` (one run per profile)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Profiles` | string[] | `@("256x48","512x96","768x144")` | Resolution profiles |
| `-TotalSteps` | int | `512` | Steps per run |
| `-Repeats` | int | `3` | Repetitions per profile |
| `-Actors` | int | `4` | Actor count |
| `-TemporalCore` | string | `mamba2` | Temporal backend |
| `-MinibatchSize` | int | `64` | PPO minibatch size |
| `-ProfileCudaEvents` | switch | | Enable CUDA event tracing |
| `-OutputRoot` | string | `artifacts/benchmarks/resolution-compare` | Results |
| `-Scene` / `-GmDagFile` | string | `""` | Scene overrides |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\run-resolution-compare.ps1 -Profiles @("256x48","512x96") -TotalSteps 1024
```

---

#### `run-actor-scaling-test.ps1` -- Fleet Scaling Benchmark

Find optimal parallel actor count for current hardware. [(Flow 8D)](#8d--actor-scaling-test)

**Wraps:** `uv run --project projects/actor brain train` (one run per count)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-ActorCountsStr` | string | `"4,8,12,16,20,24,28,32"` | Comma-separated actor counts |
| `-StepsPerRun` | int | `2000` | Training steps per count |
| `-RolloutLength` | int | `256` | Rollout length |
| `-TemporalCore` | string | `mamba2` | Temporal backend |
| `-LogEvery` | int | `100` | Log frequency |

```powershell
.\scripts\run-actor-scaling-test.ps1
.\scripts\run-actor-scaling-test.ps1 -ActorCountsStr "4,8,16,32" -StepsPerRun 5000
```

---

#### `run-attribution-matrix.ps1` -- Throughput Attribution Ablation

Systematically disable actor-side components (episodic memory, reward shaping,
telemetry) to isolate throughput bottlenecks. [(Flow 8E)](#8e--attribution-matrix)

**Wraps:** `uv run --project projects/actor brain train` (one run per ablation)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-GmDagFile` | string | | Test scene |
| `-Actors` | int | `4` | Actor count |
| `-TotalSteps` | int | `2048` | Steps per configuration |
| `-LogEvery` | int | `256` | Log frequency |
| `-AzimuthBins` | int | `256` | Azimuth resolution |
| `-ElevationBins` | int | `48` | Elevation resolution |
| `-BasePort` | int | `5700` | Base telemetry port |
| `-OutputRoot` | string | `artifacts/benchmarks/attribution-matrix` | Results |

```powershell
.\scripts\run-attribution-matrix.ps1 -TotalSteps 4096
```

---

### Validation Scripts

#### `qualify-canonical-stack.ps1` -- Stack Qualification

One-pass end-to-end proof: dataset audit -> bounded training -> checkpoint resume -> replay.
[(Flow 9A)](#9a--canonical-stack-qualification)

**Wraps:** Orchestrates `uv run navi-auditor dataset-audit` + `uv run brain train` + dashboard attach

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-TotalSteps` | int | `512` | Qualification training steps |
| `-CheckpointEvery` | int | `256` | Checkpoint interval |
| `-StartupTimeoutSeconds` | int | `90` | Process startup timeout |
| `-AttachTimeoutSeconds` | int | `20` | Dashboard attach timeout |
| `-ResumeAdditionalSteps` | int | `0` | Extra steps after resume |
| `-SkipDatasetAudit` | switch | | Skip dataset validation phase |
| `-SkipResumeQualification` | switch | | Skip resume proof phase |
| `-EnableCorpusRefreshQualification` | switch | | Run corpus refresh validation |
| `-RunRoot` | string | `artifacts\qualification\canonical_stack` | Output root |
| `-NoPreKill` | switch | | Skip stale process cleanup |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\qualify-canonical-stack.ps1                                  # Default qualification
.\scripts\qualify-canonical-stack.ps1 -TotalSteps 1024 -SkipDatasetAudit
```

---

#### `run-nightly-validation.ps1` -- Overnight Validation Pipeline

Comprehensive automated validation: preflight -> regressions -> qualification ->
checkpoint/resume -> drift benchmark -> soak. [(Flow 9B)](#9b--nightly-validation-overnight)

**Wraps:** Orchestrates `qualify-canonical-stack.ps1` + `run-ghost-stack.ps1 -Train` + monitoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-SoakHours` | int | `8` | Overnight soak duration |
| `-QualificationSteps` | int | `512` | Initial qualification steps |
| `-MonitorIntervalSeconds` | int | `300` | Checkpoint monitoring interval |
| `-CheckpointEvery` | int | `25000` | Checkpoint interval (soak phase) |
| `-CheckpointStallMinutes` | int | `60` | Stall timeout (minutes) |
| `-BoundedSteps` | int | | Steps for bounded training |
| `-BoundedResumeSteps` | int | | Steps for resume phase |
| `-AttachTimeoutSeconds` | int | `20` | Dashboard attach timeout |
| `-RunRoot` | string | `artifacts/nightly` | Nightly run root |
| `-NoPreKill` | switch | | Preserve prior process state |
| `-PythonVersion` | string | `3.12` | Python version for uv |

```powershell
.\scripts\run-nightly-validation.ps1                                   # Full overnight run
.\scripts\run-nightly-validation.ps1 -SoakHours 2                      # Shorter soak
```

---

### Setup & Diagnostics

#### `setup-actor-cuda.ps1` -- Install CUDA PyTorch (Windows)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-CudaTag` | string | `cu121` | CUDA version tag |
| `-TorchVersion` | string | `2.5.1` | PyTorch version |
| `-SkipActorSync` | switch | | Skip `uv sync` after install |
| `-InstallFusedTemporal` | switch | | Install mamba-ssm fused wheels |
| `-FusedWheelPath` | string | `""` | Path to fused temporal wheels |

```powershell
.\scripts\setup-actor-cuda.ps1                                        # Default CUDA 12.1
.\scripts\setup-actor-cuda.ps1 -CudaTag cu124 -TorchVersion 2.6.0    # CUDA 12.4
```

**Linux/WSL2 variant:** `./scripts/setup-actor-cuda.sh`

---

#### `summarize-bounded-train-log.ps1` -- Training Log Extraction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-LogPath` | string | (required) | Path to training log file |
| `-RepoLogPath` | string | `logs/navi_actor_train.log` | Fallback log path |
| `-OutputJson` | string | `""` | Output JSON file (optional) |

```powershell
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1 -OutputJson summary.json
```

---

#### Python Diagnostic Utilities

| Script | Invocation | Purpose |
|--------|-----------|---------|
| `check_gpu.py` | `python scripts\check_gpu.py` | CUDA availability + kernel execution test |
| `check_embree.py` | `python scripts\check_embree.py` | Embree ray tracing backend detection |
| `diagnose_gmdag_corpus.py` | `python scripts\diagnose_gmdag_corpus.py` | Shallow .gmdag header diagnostics |
| `diagnose_gmdag_deep.py` | `python scripts\diagnose_gmdag_deep.py` | Deep DAG structural analysis (up to 2M nodes) |
| `run-json-command.py` | `python scripts\run-json-command.py` | Machine-readable subprocess wrapper |
| `run-structured-surface.py` | `python scripts\run-structured-surface.py` | Diagnostic surface orchestrator |
| `benchmark_canonical_stack.py` | `python scripts\benchmark_canonical_stack.py` | End-to-end throughput proof (CI gate) |

---

## CLI Entry Points

Every project exposes CLI commands via `uv run`. These are the raw commands
that the wrapper scripts call under the hood. Use them directly when you need
precise control beyond what the wrappers provide.

> **Pattern:** `uv run --project projects\<project> <command> [options]`
>
> **Shortcuts:** Some projects register shortcut names so you can skip the
> full `navi-*` prefix. Shortcuts are noted in each section.

---

### Environment (`projects/environment`)

**Base command:** `uv run --project projects\environment navi-environment <command>`
**Shortcut:** `uv run --project projects\environment environment` -> `serve`

#### `serve` -- Start Environment Server

ZMQ-based environment server for the 3-process inference stack.

```powershell
uv run --project projects\environment navi-environment serve [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pub` | string | `tcp://localhost:5559` | PUB socket address (observation broadcast) |
| `--rep` | string | `tcp://localhost:5560` | REP socket address (step request/response) |
| `--action-sub` | string | `tcp://localhost:5557` | SUB socket for actor actions |
| `--mode` | string | `step` | Service mode (`step` / `async`) |
| `--gmdag-file` | string | | .gmdag scene file to load |
| `--actors` | int | `1` | Number of parallel actors |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |
| `--max-distance` | float | `30.0` | Maximum ray distance (meters) |
| `--sdf-max-steps` | int | `256` | Sphere-tracing max iterations |

#### `compile-gmdag` -- Compile Mesh to .gmdag

```powershell
uv run --project projects\environment navi-environment compile-gmdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--source` | string | (required) | Input mesh file (.glb/.obj/.ply/.stl) |
| `--output` | string | (required) | Output .gmdag file path |
| `--resolution` | int | `512` | Target cubic voxel resolution |

#### `prepare-corpus` -- Full Corpus Preparation

Discover source scenes, compile missing .gmdag assets, emit manifest.

```powershell
uv run --project projects\environment navi-environment prepare-corpus [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scene` | string | `""` | Single scene name filter |
| `--manifest` | string | `""` | Existing manifest to update |
| `--corpus-root` | string | `artifacts/gmdag/corpus` | Live corpus root |
| `--gmdag-root` | string | `artifacts/gmdag` | Parent .gmdag directory |
| `--resolution` | int | `512` | Compile resolution |
| `--min-scene-bytes` | int | `1000` | Minimum scene file size |
| `--force-recompile` | flag | | Recompile all assets |
| `--json` | flag | | Machine-readable JSON output |

#### `check-sdfdag` -- Validate Runtime + Assets

```powershell
uv run --project projects\environment navi-environment check-sdfdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | Validate single .gmdag file |
| `--gmdag-root` | string | | Validate all files in directory |
| `--manifest` | string | | Validate against manifest |
| `--expected-resolution` | int | `512` | Expected compile resolution |
| `--json` | flag | | Machine-readable JSON output |

#### `bench-sdfdag` -- Benchmark Compiled Assets

Canonical environment-layer throughput benchmark for the SDF/DAG path.

```powershell
uv run --project projects\environment navi-environment bench-sdfdag [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | (required) | .gmdag file to benchmark |
| `--actors` | int | `4` | Parallel actor count |
| `--steps` | int | `200` | Benchmark steps |
| `--warmup-steps` | int | `25` | Warmup steps (excluded from timing) |
| `--repeats` | int | `1` | Benchmark repetitions |
| `--azimuth-bins` | int | `256` | Azimuth resolution |
| `--elevation-bins` | int | `48` | Elevation resolution |
| `--max-distance` | float | `30.0` | Max ray distance |
| `--sdf-max-steps` | int | `256` | Sphere-tracing iterations |
| `--torch-compile` | flag | | Enable torch.compile (sm_70+) |
| `--json` | flag | | Machine-readable JSON output |

---

### Actor (`projects/actor`)

**Base command:** `uv run --project projects\actor navi-actor <command>`
**Shortcut:** `uv run --project projects\actor brain <subcommand>` -> unified entry point

#### `train` -- Unified In-Process PPO Training

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
| `--gmdag-root` | string | `""` | .gmdag root directory |
| `--gmdag-file` | string | `""` | Single .gmdag file override |
| `--compile-resolution` | int | `512` | Auto-compile resolution |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |
| `--enable-episodic-memory` | flag | | Enable episodic memory module |
| `--enable-reward-shaping` | flag | | Enable reward shaping |
| `--emit-observation-stream` | flag | | Enable observation PUB stream |
| `--emit-training-telemetry` | flag | | Enable training telemetry events |
| `--emit-perf-telemetry` | flag | | Enable performance telemetry |

#### `serve` -- Actor Policy Server

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

#### `bc-pretrain` -- Behavioral Cloning Pre-Training

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

#### `profile` -- Throughput Profiling

Run fixed-length rollout with CUDA profiling active.

```powershell
uv run --project projects\actor navi-actor profile [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--scene` | string | | Scene to profile |
| `--steps` | int | `512` | Roll out steps |
| `--actors` | int | `4` | Actor count |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |

#### `infer` -- In-Process Policy Evaluation

Evaluate a trained checkpoint with direct in-process CUDA stepping.
Same backend as training (SdfDagBackend, tensor-native), but without PPO,
rollout buffers, or episodic memory. [(Flow 6)](#flow-6--in-process-inference-canonical)

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
| `--gmdag-file` | string | | Single .gmdag file path |
| `--gmdag-root` | string | | Custom .gmdag root |
| `--corpus-root` | string | | Corpus root directory |
| `--manifest` | string | | Corpus manifest path |
| `--datasets` | string | | Include only these datasets |
| `--exclude-datasets` | string | | Exclude datasets by name |
| `--azimuth-bins` | int | `256` | Observation azimuth resolution |
| `--elevation-bins` | int | `48` | Observation elevation resolution |

#### `brain` -- Unified Entry Point

Delegates to `train`, `serve`, `infer`, or `profile` based on the mode argument:

```powershell
uv run --project projects\actor brain train [options]      # -> navi-actor train
uv run --project projects\actor brain serve [options]      # -> navi-actor serve
uv run --project projects\actor brain infer [options]      # -> navi-actor infer
uv run --project projects\actor brain bc-pretrain [options] # -> bc-pretrain
uv run --project projects\actor brain profile [options]    # -> navi-actor profile
```

---

### Auditor (`projects/auditor`)

**Base command:** `uv run --project projects\auditor navi-auditor <command>`
**Shortcuts:**
- `uv run --project projects\auditor dashboard` -> `navi-auditor dashboard`
- `uv run --project projects\auditor explore` -> `navi-auditor explore`

#### `dashboard` -- Live Passive Observation Dashboard

Subscribes to actor PUB stream via a split-socket architecture: a `CONFLATE`
observation socket always delivers the latest frame, while a separate telemetry
socket preserves ordered metrics. Displays actor 0 observations. Auto-detects mode.

```powershell
uv run --project projects\auditor navi-auditor dashboard [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | `tcp://localhost:5557` | Actor PUB socket to subscribe |
| `--matrix-sub` | string | `""` | Environment PUB socket (optional) |
| `--step-endpoint` | string | `""` | Environment REP endpoint (optional) |
| `--passive` | flag | | Force passive mode (no env control) |
| `--hz` | float | `30.0` | Target frame rate |
| `--linear-speed` | float | | Manual control speed |
| `--yaw-rate` | float | | Manual control yaw rate |
| `--max-distance` | float | | Override max distance for colorization |
| `--scene` | string | `""` | Scene label for title bar |

#### `explore` -- Interactive Keyboard-Controlled Explorer

Standalone manual exploration: spawns environment backend + dashboard with WASD controls.

```powershell
uv run --project projects\auditor navi-auditor explore [options]
# Shortcut:
uv run --project projects\auditor explore [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | .gmdag scene to explore |
| `--pub-address` | string | `tcp://localhost:5559` | Environment PUB address |
| `--rep-address` | string | `tcp://localhost:5560` | Environment REP address |
| `--hz` | float | `30.0` | Target frame rate |
| `--record` | flag | | Record demonstration to .npz |
| `--drone-max-speed` | float | `5.0` | Drone max speed (m/s) |
| `--azimuth-bins` | int | `256` | Azimuth resolution |
| `--elevation-bins` | int | `48` | Elevation resolution |
| `--max-steps` | int | | Auto-close after N steps |

#### `record` -- Session Recording

Record live ZMQ streams to Zarr storage for later replay.

```powershell
uv run --project projects\auditor navi-auditor record [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sub` | string | | Comma-separated ZMQ SUB addresses |
| `--out` | string | | Output file path |

#### `replay` -- Session Playback

Play back a recorded session via ZMQ PUB.

```powershell
uv run --project projects\auditor navi-auditor replay [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | string | | Recorded session file |
| `--pub` | string | | PUB socket to publish on |
| `--speed` | float | `1.0` | Playback speed multiplier |

#### `dataset-audit` -- Dataset Quality Assurance

Runtime-backed dataset validation: preflight checks + optional benchmark.

```powershell
uv run --project projects\auditor navi-auditor dataset-audit [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gmdag-file` | string | | .gmdag file to audit |
| `--expected-resolution` | int | `512` | Expected compile resolution |
| `--benchmark` | flag | | Run throughput benchmark |
| `--actors` | int | `1` | Actor count (benchmark mode) |
| `--steps` | int | `8` | Benchmark steps |
| `--warmup-steps` | int | `1` | Warmup steps |
| `--azimuth-bins` | int | `64` | Audit azimuth resolution |
| `--elevation-bins` | int | `16` | Audit elevation resolution |
| `--json` | flag | | Machine-readable JSON output |

#### `dashboard-attach-check` -- Headless Attach Proof

Verify passive dashboard can connect to actor stream (used by qualification).

```powershell
uv run --project projects\auditor navi-auditor dashboard-attach-check [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | | Actor PUB socket to test |
| `--timeout-seconds` | float | `15.0` | Connection timeout |
| `--json` | flag | | Machine-readable JSON output |

#### `dashboard-capture-frame` -- Capture Dashboard Frame

Capture one live dashboard frame + rendered diagnostics to file.

```powershell
uv run --project projects\auditor navi-auditor dashboard-capture-frame [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--actor-sub` | string | | Actor PUB socket |
| `--actor-id` | int | `0` | Actor ID to capture |
| `--timeout-seconds` | float | `15.0` | Connection timeout |
| `--output-dir` | string | | Output directory for captured frames |
| `--max-distance` | float | | Override max distance |
| `--json` | flag | | Machine-readable JSON output |

---

### Voxel-DAG Compiler (`projects/voxel-dag`)

**Base command:** `uv run --project projects\voxel-dag <command>`

#### `voxel-dag-compiler` -- Compile OBJ to .gmdag

Low-level compiler entry point (the environment `compile-gmdag` command wraps this).

```powershell
uv run --project projects\voxel-dag voxel-dag-compiler [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | string | (required) | Input OBJ mesh path |
| `--output` | string | (required) | Output .gmdag file path |
| `--resolution` | int | `512` | Target cubic voxel resolution |
| `--padding` | float | `0.1` | Relative cubic padding |

#### `bsp-to-obj` -- Convert Quake 3 BSP/PK3 to OBJ

Extract geometry from Quake 3 maps for .gmdag compilation.

```powershell
uv run --project projects\voxel-dag bsp-to-obj [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` / `-i` | string | (required) | Input .bsp or .pk3 file |
| `--output` / `-o` | string | (required) | Output .obj (BSP) or directory (PK3) |
| `--tessellation` / `-t` | int | `4` | Bezier patch tessellation level |
| `--no-metric-conversion` | flag | | Keep Q3 native units (inches) |
| `--export-spawns` | flag | | Export spawn points as .spawns.json |

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
| [docs/INFERENCE.md](docs/INFERENCE.md) | In-process inference: architecture, telemetry, CLI, scripts |
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
