# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first reinforcement-learning system that teaches autonomous
drones to navigate complex 3D environments through direct mathematical
observation of compiled geometry. Rather than rendering pixels or depth images
from a graphics pipeline, Navi compiles arbitrary 3D scene meshes into
compressed signed distance fields, traces thousands of rays through them on the
GPU in parallel, and feeds the resulting spherical distance observations to a
sacred cognitive engine trained via Proximal Policy Optimization.

The name "Ghost-Matrix" reflects the core architectural insight: the agent never
sees the world as rendered graphics. Instead, it perceives a mathematical ghost
of the geometry — a dense spherical distance matrix that encodes obstacle
proximity in every direction simultaneously. This observation is the agent's
entire reality, and the cognitive engine learns to navigate, avoid collisions,
explore, and survive purely from this mathematical signal.

**The system is designed around a single imperative: high-throughput training
without stalls.** Every architectural decision — from GPU-resident rollout
storage to in-process environment stepping to zero-copy CUDA tensor contracts —
exists to maximize the number of training steps per second on a single machine.
The agent must experience as many episodes as possible, in as many diverse
scenes as possible, to learn robust navigation behavior that generalizes across
arbitrary geometry.

### Why This Architecture?

Traditional robot-learning systems couple the agent to a graphical simulator —
rendering frames, extracting depth maps, shipping them over inter-process
boundaries. Each stage adds latency, memory copies, and synchronization stalls
that limit how fast the agent can learn.

Navi eliminates this entire pipeline:

1. **Compile once, trace forever.** Source meshes (`.glb`, `.obj`, `.bsp`) are
   compiled offline into `.gmdag` signed distance field assets. These binary
   assets load in milliseconds and support thousands of ray queries per
   microsecond on the GPU.

2. **In-process training.** The environment backend, CUDA ray tracer, and PPO
   optimizer all run inside a single Python process on the same GPU. There is no
   inter-process serialization, no ZMQ message passing, and no host↔device
   transfer in the training hot loop.

3. **Spherical observation.** Each agent perceives a complete `(azimuth ×
   elevation)` spherical distance matrix — a 360° awareness of geometry in all
   directions. This is not a camera frustum; it is an omnidirectional proximity
   field that gives the agent full situational awareness.

4. **GPU-resident rollout storage.** Observations, actions, rewards, values, and
   hidden states never leave the GPU during training. The PPO optimizer reads
   directly from device-resident rollout buffers, eliminating the CPU↔GPU
   round-trip that dominates many RL implementations.

5. **Sacred cognitive engine.** The actor brain is architecturally immutable —
   data adapts to it, never the other way around. This forces all environment
   backends, datasets, and integration layers to produce the canonical
   observation format, guaranteeing that the cognitive architecture is never
   compromised by data convenience.

---

## Table of Contents

- [Canonical Runtime Path](#canonical-runtime-path)
- [Architecture Overview](#architecture-overview)
- [The Cognitive Engine](#the-cognitive-engine)
- [Ghost-Matrix Persistence](#ghost-matrix-persistence)
- [Reward Architecture](#reward-architecture)
- [SDF/DAG Compiled Asset Pipeline](#sdfdag-compiled-asset-pipeline)
- [Observation Contract](#observation-contract)
- [Temporal Core](#temporal-core)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Corpus Preparation](#corpus-preparation)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Model Management](#model-management)
- [Live Dashboard & Observability](#live-dashboard--observability)
- [Behavioral Cloning](#behavioral-cloning)
- [Benchmarking](#benchmarking)
- [Validation & Qualification](#validation--qualification)
- [Diagnostics](#diagnostics)
- [Repository Commands](#repository-commands)
- [Artifact Layout](#artifact-layout)
- [Documentation Index](#documentation-index)
- [Performance Targets](#performance-targets)
- [License](#license)

---

## Canonical Runtime Path

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

The pipeline has two distinct phases:

**Offline compilation** — The `voxel-dag` compiler transforms arbitrary polygon
meshes into a binary `.gmdag` asset containing a compressed Signed Distance
Field stored as a deduplicated Directed Acyclic Graph. This is a one-time cost
per scene. The compiler uses the Fast Sweeping Method (FSM) to generate the SDF
from mesh geometry and MurmurHash3 deduplication to compress repeated spatial
patterns. Scenes with local-space geometry (e.g., ReplicaCAD Baked Lighting with
scene-graph scale transforms) are optionally mesh-repaired to world space before
compilation.

**Runtime training** — The `torch-sdf` CUDA kernel loads the `.gmdag` asset onto
the GPU and executes bounded batched sphere tracing for all actors in parallel.
Results are written directly into preallocated PyTorch CUDA tensors — no CPU
staging, no memory copies. The unified PPO trainer reads these tensors, runs the
cognitive pipeline forward pass, produces actions, and steps the environment
again, all on the same device in the same process.

The `navi-auditor` dashboard attaches passively to the actor's ZMQ PUB telemetry
stream to visualize training progress without perturbing the training loop.

---

## Architecture Overview

Navi is composed of six sovereign projects, each with its own virtual
environment, package configuration, and test suite. No service imports another
service package — cross-project integration occurs only at CLI orchestration
boundaries. This isolation ensures that changes to one layer cannot silently
break another.

### Compile Layer — voxel-dag

**[projects/voxel-dag/](projects/voxel-dag/)** transforms 3D scene meshes into
compressed Signed Distance Fields stored as a Directed Acyclic Graph (DAG) with
Sparse Voxel Octree (SVO) hierarchy. The output `.gmdag` binary is the canonical
compiled asset format consumed by all downstream components.

The compiler pipeline:

1. **Mesh Ingestion** — Loads `.glb`, `.obj`, `.ply`, `.stl`, or any
   Assimp-supported format. For scenes with scene-graph transforms (e.g.,
   ReplicaCAD Baked Lighting), optional mesh repair applies
   `trimesh.Scene.dump()` to bake transforms to world space, merge vertices,
   fill holes, and fix normals.

2. **SDF Generation** — Multi-threaded Fast Sweeping Method (FSM) Eikonal solver
   computes a dense signed distance field on a uniform cubic grid. The mesh is
   auto-centered and padded into a perfect cube with uniform voxel size across
   all axes. Resolution is adjusted to the nearest power of 2 for GPU bit-shift
   compatibility.

3. **DAG Compression** — MurmurHash3 candidate grouping with structural equality
   as the correctness authority deduplicates identical subtrees. This typically
   achieves 3–10× compression versus the dense SDF grid while preserving
   bit-exact geometry.

4. **Binary Output** — The `.gmdag` file contains: header metadata (resolution,
   bounding box, node count), the DAG node array, leaf distance payloads, and a
   validation checksum. The format is documented in
   [docs/GMDAG.md](docs/GMDAG.md).

Two compiler implementations exist: a Python verification compiler for monorepo
tests and a C++17 native compiler (with CMake build) for production throughput.
Both produce byte-identical output for the same inputs.

**Build:** `cd projects/voxel-dag && uv sync` (Python) or
`cmake .. && cmake --build .` (C++). The C++ build requires CMake 3.18+ and a
C++17 compiler. Assimp is fetched automatically via CMake.

### Trace Layer — torch-sdf

**[projects/torch-sdf/](projects/torch-sdf/)** executes bounded batched sphere
tracing against compiled `.gmdag` assets on the GPU. It writes results directly
into preallocated PyTorch CUDA tensors through zero-copy PyBind11/LibTorch
bindings.

Key runtime properties:

- **Bounded stackless traversal** — Iterative DAG descent with explicit
  `max_steps`, horizon, and hit-epsilon semantics. No recursion, no stack
  allocation, no data-dependent branching that would stall GPU warps.

- **Macro-cell void cache** — Reuses empty child-cell bounds so repeated
  samples in the same void region skip redundant root traversals. This
  dramatically accelerates rays that traverse large empty spaces.

- **Zero-copy execution** — Reads ray origins/directions and writes hit
  distances/semantics directly through raw PyTorch CUDA tensor pointers. No
  CPU staging buffers, no `cudaMemcpy`, no host synchronization.

- **Strict validation** — Device, dtype, rank, shape, and contiguity are
  validated before kernel launch. Input tensors must be contiguous CUDA
  `float32` shaped `[batch, rays, 3]`; output tensors are preallocated CUDA
  `[batch, rays]`.

- **Hot-path direction validation skip** — When ray directions are
  mathematically guaranteed normalized (e.g., yaw-rotated unit vectors from the
  environment), `skip_direction_validation=True` eliminates a GPU→CPU
  synchronization barrier.

**Build:** `cd projects/torch-sdf && pip install -e .`. Requires CUDA Toolkit
11.8+ and PyTorch 2.0+ with CUDA support. On Windows, set `CUDA_HOME` before
building.

### Simulation Layer — navi-environment

**[projects/environment/](projects/environment/)** provides headless batched
environment stepping against compiled `.gmdag` assets using the CUDA `sdfdag`
backend. This is where scene geometry meets agent physics.

Responsibilities:

- **Batched stepping** — Steps all parallel actors simultaneously on the GPU.
  Each actor has independent pose, velocity, and episode state. Actions are
  4-DOF velocity commands `[forward, vertical, lateral, yaw]`.

- **Drone kinematics** — Translates actions into 3D position/orientation updates
  with configurable speed limits. Maximum drone speed defaults to 5.0 m/s so the
  proximity speed limiter has adequate reaction time before geometry contact.

- **Spherical observation** — After each step, casts rays in a spherical pattern
  (default 256 azimuth × 48 elevation bins) from each actor's position to
  produce a `DistanceMatrix v2` observation — the agent's complete
  omnidirectional view of nearby geometry.

- **Reward computation** — Computes per-step reward signals combining collision
  penalty, forward progress, proximity shaping, exploration incentive, and
  information-foraging bonuses (see [Reward Architecture](#reward-architecture)).

- **Corpus management** — Discovers, compiles, validates, and qualifies `.gmdag`
  assets for training. Manages scene rotation across the corpus during training.

- **Episode lifecycle** — Handles truncation (hard 2000-step limit), scene
  rotation (16 episodes per scene before switching), spawn selection
  (quality-gated positions with low starvation), and void grace periods
  (10 steps after reset before starvation penalty).

The environment exposes two runtime modes:

1. **In-process backend** — Instantiated directly by the trainer with no IPC
   overhead (canonical training path).
2. **ZMQ server** — Publishes observations on PUB/SUB for the legacy 3-process
   inference stack.

**Build:** `cd projects/environment && uv sync`

### Brain Layer — navi-actor

**[projects/actor/](projects/actor/)** houses the sacred cognitive engine — the
architecturally immutable brain that processes spherical distance observations
and produces navigation actions. The actor is the most complex project in the
system, encompassing:

- **Cognitive pipeline** — A 5-stage neural architecture (see
  [The Cognitive Engine](#the-cognitive-engine)) that transforms raw distance
  observations into 4-DOF action commands.

- **PPO trainer** — Proximal Policy Optimization with GPU-resident rollout
  storage, sequence-aware minibatching, and truncated backpropagation through
  time (BPTT). 4 parallel actors by default, configurable up to hardware limits.

- **In-process environment stepping** — The trainer instantiates the `sdfdag`
  environment backend directly, stepping all actors and computing PPO updates in
  the same process with no ZMQ serialization.

- **Checkpoint management** — v3 checkpoint format with step count, reward EMA,
  training lineage, temporal core type, corpus summary, and wall time. Supports
  auto-resume from the model registry and auto-promote when training improves.

- **Inference evaluation** — Bounded or continuous inference running the same
  in-process backend as training, for evaluating trained policies.

- **Behavioral cloning** — Supervised pre-training from human `.npz`
  demonstrations, producing a BC checkpoint that can be RL fine-tuned.

- **Temporal core selection** — Selectable sequence engine (Mamba-2 SSD default,
  cuDNN GRU and pure-Python Mamba available as comparison backends).

The actor's cognitive pipeline is **immutable by policy** — new data sources,
observation formats, or environment backends must adapt their output to the
canonical `(1, Az, El)` DistanceMatrix format through a `DatasetAdapter` in
`environment/backends/`. The brain never changes to accommodate data.

**Build:** `cd projects/actor && uv sync`, then `.\scripts\setup-actor-cuda.ps1`
to install CUDA-enabled PyTorch.

### Gallery Layer — navi-auditor

**[projects/auditor/](projects/auditor/)** provides passive observability
without ever gating simulation throughput or modifying training behavior.

Components:

- **Live Dashboard** — PyQtGraph-based GUI that subscribes to the actor's ZMQ
  PUB telemetry stream and renders actor 0's forward-hemisphere observation at
  5–10 Hz. Shows active actor count, throughput metrics, observation freshness,
  and auto-detected mode (TRAINING / INFERENCE / OBSERVER). During training, the
  dashboard runs in passive actor-only mode — it never opens environment control
  paths or depends on environment PUB availability.

- **Interactive Explorer** — Keyboard-controlled (WASD) drone flight through
  compiled scenes for inspection and demonstration recording. Spawns the sdfdag
  backend directly with no training process required.

- **Session Recording** — Records live ZMQ streams to Zarr v3 archives for
  later analysis and replay.

- **Session Replay** — Replays recorded Zarr sessions via ZMQ PUB, recreating
  the original observation/action stream for post-hoc review.

- **Demonstration Capture** — Multi-scene navigation workflow that flies through
  corpus scenes sequentially, auto-recording `.npz` demonstrations for
  behavioral cloning pre-training.

The gallery is designed for resilience: it handles missing ZMQ streams gracefully
(displaying a WAITING state), never crashes when upstream processes are
unavailable, and processes all rendering transforms (slicing, palette, labels)
internally without modifying environment or actor contracts.

**Build:** `cd projects/auditor && uv sync`

### Contracts — navi-contracts

**[projects/contracts/](projects/contracts/)** defines the canonical wire-format
models that all other projects consume. This is the single source of truth for
inter-service communication.

| Model | Description |
|-------|-------------|
| `DistanceMatrix` | Spherical depth observation `(n_envs, Az, El)` with semantic + delta-depth channels |
| `Action` | 4-DOF velocity command `[forward, vertical, lateral, yaw]` |
| `RobotPose` | 6-DOF pose `(x, y, z, roll, pitch, yaw, timestamp)` |
| `StepRequest` | Step request from Brain → Environment (REQ/REP) |
| `StepResult` | Step result with reward, done, truncated, episode return |
| `TelemetryEvent` | Keyed numeric telemetry for dashboarding and replay |

Wire topics use MessagePack serialization over ZMQ:

| Topic | Direction | Transport |
|-------|-----------|-----------|
| `distance_matrix_v2` | Environment → Brain, Gallery | PUB/SUB |
| `action_v2` | Brain → Environment, Gallery | PUB/SUB |
| `step_request_v2` | Brain → Environment | REQ/REP |
| `step_result_v2` | Environment → Brain | REQ/REP |
| `telemetry_event_v2` | Any → Gallery | PUB/SUB |

Default network ports: `5559` (environment PUB), `5560` (environment REP),
`5557` (actor PUB for actions + telemetry). The unified trainer uses only
port `5557` for the passive dashboard telemetry stream.

**Build:** `cd projects/contracts && uv sync` — library package with no
long-running service process.

---

## The Cognitive Engine

The actor's brain implements `CognitiveMambaPolicy`, a 5-stage neural pipeline
that is architecturally sacred — it is never modified to accommodate new data
sources or environment backends.

```
DistanceMatrix (B, 3, Az, El)
         │
    ┌────▼──────────┐
    │ RayViTEncoder  │  Vision Transformer: (B, 3, Az, El) → (B, 128)
    └────┬──────────┘  Spherical positional encoding + patch projection
         │             Cached sin/cos encodings (no recomputation)
    ┌────▼──────────┐
    │ RND Curiosity  │  Random Network Distillation intrinsic reward
    └────┬──────────┘  Drives exploration by rewarding embedding novelty
         │
    ┌────▼──────────┐
    │EpisodicMemory  │  Tensor-native cosine-similarity query
    └────┬──────────┘  Detects spatial loop patterns, penalizes revisitation
         │
    ┌────▼──────────┐
    │ TemporalCore   │  Sequence engine (Mamba-2 SSD / cuDNN GRU / mambapy)
    └────┬──────────┘  Maintains hidden state across steps for temporal memory
         │
    ┌────▼──────────────┐
    │ ActorCriticHeads   │  4-DOF Gaussian action distribution + value estimation
    └───────────────────┘  [forward, vertical, lateral, yaw] + V(s)
```

**Stage 1 — RayViTEncoder:** A Vision Transformer that processes the 3-channel
distance observation `(depth, semantic, delta_depth)` through fixed spherical
positional encodings. Patch projection with `patch_size=8` converts the
`(Az, El)` grid into a token sequence; self-attention produces a 128-dimensional
spatial embedding. Sin/cos positional encodings are cached at initialization to
avoid redundant recomputation on every forward pass.

**Stage 2 — RND Curiosity:** Random Network Distillation compares the spatial
embedding against a fixed random target network. When the agent encounters
unfamiliar geometry, the prediction error is high, generating an intrinsic
reward that drives exploration into novel regions of the scene.

**Stage 3 — EpisodicMemory:** A tensor-native cosine-similarity buffer that
stores recent spatial embeddings. When the current embedding closely matches a
stored memory (indicating the agent is revisiting a location), the episodic
signal penalizes the reward to discourage spatial loops. Eviction does not
trigger full-index rebuilds on every post-capacity insert.

**Stage 4 — TemporalCore:** A recurrent sequence engine that maintains hidden
state across timesteps, giving the agent temporal memory beyond the current
observation. The canonical default is **pure-PyTorch Mamba-2 SSD** (Selective
State-Space Duality), chosen for superior learning quality over cuDNN GRU in a
25K-step head-to-head comparison (reward_ema −0.88 vs −1.48). Hidden states are
**never** reset upon geometry grazing, preserving situational context.

**Stage 5 — ActorCriticHeads:** Produces a 4-DOF Gaussian action distribution
`[forward, vertical, lateral, yaw]` and a scalar value estimate `V(s)` for PPO.

Input contract: the cognitive engine consumes only `depth` and `semantic`
channels from `DistanceMatrix`. All observation preprocessing (normalization,
axis ordering, horizon clipping) is handled upstream in the environment backend.

---

## Ghost-Matrix Persistence

The training philosophy is built around **continuous learning without death
resets.** Collisions with geometry do not kill the agent — instead, the agent
must learn to escape, recover, and navigate away through continuous per-step
negative reward signals.

- **No collision death** — `done=True` is never triggered by geometry contact
  during training. The agent stays alive and must learn recovery behavior.

- **Escape incentive** — A positive shaping signal rewards the agent for
  increasing obstacle clearance while near geometry, so recovery is learned
  in-scene instead of via reset churn.

- **Proximity-discounted progress** — Forward progress reward is discounted by
  proximity ratio so approaching walls yields diminishing forward credit instead
  of rewarding unsafe approaches.

- **Velocity-scaled collision** — Collision penalty scales with movement speed
  so fast crashes are punished more severely than gentle grazing, creating a
  natural speed-awareness incentive.

- **Context preservation** — Temporal hidden states (Mamba-2 SSD) are never
  reset upon grazing geometry, preserving the agent's situational awareness and
  accumulated context through collision events.

- **Hard truncation** — Episodes end after 2000 steps to ensure episodic
  diversity. This is the only mandatory episode termination mechanism.

- **Void grace period** — Starvation truncation is suppressed for 10 steps
  after each episode reset so agents can navigate away from high-starvation
  spawn positions before being penalized.

- **Scene residency** — During corpus training, each actor stays on a scene for
  16 completed episodes before rotation, balancing scene diversity with local
  mastery.

---

## Reward Architecture

Each training step produces a composite reward signal from multiple shaped
components. The reward design balances navigation progress against safety,
exploration, and information quality.

| Component | Description |
|-----------|-------------|
| **Collision penalty** | Negative signal scaled by movement speed — fast crashes are punished more than gentle grazing |
| **Forward progress** | Positive reward for forward movement, discounted by proximity ratio near geometry |
| **Proximity shaping** | Gradient signal that increases punishment as the agent approaches surfaces |
| **Escape incentive** | Positive reward for increasing clearance while already near geometry |
| **Exploration bonus** | Rewards visiting novel spatial regions, gated by current clearance to prevent wall-hugging |
| **Information foraging** | Penalizes horizon-saturated (blind) views and near-field wall-hugging using spherical observation statistics |
| **Structure seeking** | Positively rewards stable mid-range structure visibility and reorientation toward informative geometry |
| **Existential tax** | Small per-step negative penalty (default −0.02) that prevents the agent from idling |
| **RND curiosity** | Intrinsic reward from Random Network Distillation embedding novelty |
| **Episodic memory** | Negative signal when cosine similarity to stored spatial embeddings indicates loop revisitation |

Forward velocity reward weight defaults to `0.0` to eliminate inherent approach
bias toward obstacles. The conservative drone max speed (5.0 m/s) gives the
proximity speed limiter adequate reaction time.

---

## SDF/DAG Compiled Asset Pipeline

The compile-once-trace-forever approach is central to Navi's throughput story.
Any source mesh becomes a GPU-ready `.gmdag` asset that supports thousands of
ray queries per frame with no geometry processing at runtime.

### Compilation Flow

```
Source Mesh (.glb/.obj/.bsp)
     │
     ├─ Optional: mesh repair (scene-graph transforms → world space)
     │
     ▼
Dense SDF Generation (Fast Sweeping Method, resolution N³)
     │
     ▼
DAG Compression (MurmurHash3 deduplication, SVO hierarchy)
     │
     ▼
Binary Output (.gmdag: header + DAG nodes + leaf payloads + checksum)
     │
     ▼
Observation Quality Gate (qualify-gmdag: CUDA ray casting validation)
```

### Compilation Details

- **Resolution** — Canonical compile resolution is `512` (512³ voxels). The
  resolution is adjusted to the nearest power of 2 for GPU bit-shift
  compatibility. Higher resolutions produce finer geometric detail but larger
  assets.

- **Mathematical cubicity** — The compiler auto-centers and pads the scene
  bounding box into a perfect cube with uniform voxel size across all axes.

- **Mesh repair** — Datasets with scene-graph scale transforms (e.g., ReplicaCAD
  Baked Lighting) must be compiled with `--repair`, which applies
  `trimesh.Scene.dump()` to produce correct world-space geometry before
  voxelization.

- **Quality gate** — After compilation, `qualify-gmdag` probes a 5×7×5 spawn
  candidate grid with 48×12 ray directions and scores candidates using
  `_observation_profile` and `_spawn_candidate_score`. Scenes with 0 viable
  spawn candidates or best starvation ≥ 70% are rejected.

- **Determinism** — Repeated compilation of the same source mesh at the same
  resolution produces byte-identical `.gmdag` output.

- **Binary integrity** — The `.gmdag` loader rejects malformed headers,
  non-finite bounds, impossible pointer layouts, out-of-range child references,
  cycles, and trailing or truncated payloads.

### Supported Source Formats

| Format | Source |
|--------|--------|
| `.glb` / `.gltf` | HuggingFace datasets (Habitat, ReplicaCAD) |
| `.obj` | Converted from BSP via `bsp-to-obj` |
| `.bsp` | Quake 3 Arena community maps |
| `.ply`, `.stl` | Any Assimp-supported geometry |

See [docs/GMDAG.md](docs/GMDAG.md) for the binary format specification and
[docs/COMPILER.md](docs/COMPILER.md) for compiler internals.

---

## Observation Contract

The canonical observation is a **256×48 spherical distance matrix** (256 azimuth
bins × 48 elevation bins), representing a full 360° panoramic view of obstacle
proximity in all directions from the agent's position.

Each observation is a 3-channel tensor `(depth, semantic, delta_depth)`:

- **depth** — Distance to nearest surface along each ray, normalized by horizon
- **semantic** — Surface type classification (hit vs. miss)
- **delta_depth** — Frame-to-frame distance change for temporal awareness

The observation passes through `RayViTEncoder` with `patch_size=8`, producing a
token sequence of length `(Az / 8) × (El / 8)`. Self-attention cost grows
roughly with the square of the token count:

| Profile | Resolution | Tokens | Status |
|---------|-----------|--------|--------|
| **Standard** | `256×48` | 192 | **Production default** |
| High | `384×72` | 432 | Benchmark comparison |
| Ultra | `512×96` | 768 | Benchmark comparison |
| Extreme | `768×144` | 1,728 | Exceeds trainer VRAM on MX150 |

Higher observation resolutions are environment-viable (the CUDA ray tracer
scales linearly) but create quadratic attention cost in the RayViT encoder and
proportional PPO memory growth. The `256×48` default balances geometric fidelity
with training throughput on current hardware.

See [docs/RESOLUTION_BENCHMARKS.md](docs/RESOLUTION_BENCHMARKS.md) for the full
observation-resolution sweep results.

---

## Temporal Core

The sequence engine that gives the agent temporal memory across timesteps.
Without this, the agent would be purely reactive — responding only to the
current observation with no memory of where it has been or what it has seen.

### Canonical Default: Mamba-2 SSD

A pure-PyTorch implementation of Mamba-2 Selective State-Space Duality. Chosen
as the canonical default after a 25K-step head-to-head training comparison:

| Metric | Mamba-2 SSD | cuDNN GRU |
|--------|-------------|-----------|
| Final reward_ema | **−0.88** | −1.48 |
| Throughput | ~72 SPS | ~100 SPS |
| CUDA kernels per forward | ~55–60 | 2–4 (fused) |

Mamba-2 delivers significantly better learning quality at a modest throughput
cost. The throughput difference is dominated by PPO optimizer cost rather than
the temporal core itself. The higher kernel count is a structural limitation
of the pure-PyTorch implementation, resolvable only by hardware-fused
`mamba-ssm` Triton kernels (not available on Windows).

### Available Backends

| Backend | Implementation | Use Case |
|---------|---------------|----------|
| `mamba2` | Pure-PyTorch SSD | **Default** — best learning, no build step required |
| `gru` | cuDNN fused GRU | Comparison — highest throughput, lower quality |
| `mambapy` | Pure-Python Mamba | Debugging and reference only |

The `--temporal-core` flag selects the backend on all training and inference
surfaces. Future hardware-fused `mamba-ssm` remains a deferred upgrade target.

See [docs/COMPARISON.md](docs/COMPARISON.md) for the full training comparison.

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
environment, and test suite. See each project's README for detailed CLI
reference, scripts, and operational guides:

| Project | Role | README |
|---------|------|--------|
| **contracts** | Wire-format models & serialization | [projects/contracts/](projects/contracts/README.md) |
| **environment** | Simulation, corpus preparation, benchmarking | [projects/environment/](projects/environment/README.md) |
| **actor** | Cognitive engine, PPO training, inference, BC | [projects/actor/](projects/actor/README.md) |
| **auditor** | Dashboard, exploration, recording, replay | [projects/auditor/](projects/auditor/README.md) |
| **voxel-dag** | Mesh → `.gmdag` offline compiler | [projects/voxel-dag/](projects/voxel-dag/README.md) |
| **torch-sdf** | CUDA sphere-tracing runtime | [projects/torch-sdf/](projects/torch-sdf/README.md) |

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

**GPU verification:**

```powershell
python scripts\check_gpu.py      # CUDA availability + kernel execution test
python scripts\check_embree.py   # Embree ray tracing backend check (optional)
```

---

## Configuration

All projects use `pydantic-settings` with a shared root `.env` file. Settings
search up the directory tree from each project to find the root `.env`, with
hard-coded fallback defaults in each project's `config.py`.

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NAVI_ENV_PUB_ADDRESS` | `tcp://localhost:5559` | Environment PUB socket (observation broadcast) |
| `NAVI_ENV_REP_ADDRESS` | `tcp://localhost:5560` | Environment REP socket (step request/response) |
| `NAVI_ACTOR_PUB_ADDRESS` | `tcp://localhost:5557` | Actor PUB socket (actions + telemetry) |
| `NAVI_AZIMUTH_BINS` | `256` | Observation azimuth resolution |
| `NAVI_ELEVATION_BINS` | `48` | Observation elevation resolution |
| `NAVI_GMDAG_RESOLUTION` | `512` | Canonical `.gmdag` compile resolution |
| `NAVI_MAX_STEPS_PER_EPISODE` | `2000` | Hard truncation step limit |
| `NAVI_SCENE_EPISODES_PER_SCENE` | `16` | Episodes per scene before rotation |
| `NAVI_MAX_DISTANCE` | `100.0` | Observation horizon (meters) |
| `NAVI_SDFDAG_TORCH_COMPILE` | `1` | Enable `torch.compile` for env hot path (`sm_70+`) |

### Default Network Ports

| Port | Service | Role |
|------|---------|------|
| `5559` | Environment | PUB (observation broadcast) |
| `5560` | Environment | REP (step request/response) |
| `5557` | Actor | PUB (action + telemetry broadcast) |

The unified trainer uses only port `5557` for the passive dashboard telemetry
stream. Ports `5559`/`5560` are used by the legacy 3-process inference stack.

---

## Corpus Preparation

The training corpus is a collection of compiled `.gmdag` scene files. Navi
supports multiple data sources and provides automated transactional pipelines
for corpus construction.

### Full Corpus Refresh

Downloads scenes from HuggingFace datasets, compiles to `.gmdag`, validates,
and promotes to the live corpus. Staged transaction: stale data is only replaced
after successful rebuild.

```powershell
.\scripts\refresh-scene-corpus.ps1
```

Canonical datasets: `ai-habitat/ReplicaCAD_dataset`,
`ai-habitat/ReplicaCAD_baked_lighting`. Note that `ai-habitat/hssd-hab` is
permanently excluded from the canonical corpus — its incomplete shell geometry
causes unrecoverable void-starvation death spirals at runtime resolution.

### Additional Data Sources

| Source | Script | Description |
|--------|--------|-------------|
| Habitat test scenes | `.\scripts\download-habitat-data.ps1` | 3 test scenes + ReplicaCAD stages for bootstrap |
| ReplicaCAD expansion | `.\scripts\expand-replicacad-corpus.ps1` | Incremental baked-lighting scenes |
| Quake 3 Arena maps | `.\scripts\download-quake3-maps.ps1` | 10+ community maps from lvlworld.com |

### Single Asset Compilation

```powershell
# Compile one mesh
uv run navi-environment compile-gmdag --source scene.glb --output scene.gmdag --resolution 512

# With mesh repair (for ReplicaCAD Baked Lighting)
uv run navi-environment compile-gmdag --source scene.glb --output scene.gmdag --resolution 512 --repair

# Validate
uv run navi-environment check-sdfdag --gmdag-file scene.gmdag

# Quality gate
uv run navi-environment qualify-gmdag --gmdag-file scene.gmdag
```

All compilation defaults to resolution 512. Scene-pool compilation, corpus
manifest management, and per-dataset quality gating are documented in
[projects/environment/ — Corpus Preparation](projects/environment/README.md#corpus-preparation).

---

## Training

### Unified In-Process Trainer

Navi's canonical training architecture runs everything in a single process:

1. The trainer loads all compiled `.gmdag` assets from the corpus
2. 4 parallel actors step the sdfdag environment backend on the GPU
3. Spherical observations remain as CUDA tensors — no CPU materialization
4. The cognitive engine produces actions via forward pass
5. Actions step the environment again, accumulating into GPU-resident rollout
   buffers
6. After `rollout_length` steps (default 256), PPO computes advantages and
   updates the policy over `ppo_epochs` epochs of minibatch gradient descent
7. The cycle repeats continuously until the step limit or Ctrl+C

No ZMQ, no IPC, no host↔device copies in the training hot loop. Materializing
Python `DistanceMatrix` or `Action` objects inside the rollout hot path is
forbidden — tensor-native runtime seams are mandatory when the backend is
already GPU-resident.

### Standard PPO Training

```powershell
.\scripts\train.ps1
```

Trains on the **full discovered corpus** and runs **continuously until stopped**.
Key parameters: `-NumActors 4`, `-TemporalCore mamba2`, `-RolloutLength 512`,
`-PpoEpochs 1`, `-LearningRate 5e-4`, `-CheckpointEvery 25000`.

### Ghost Stack Training (Orchestrated)

```powershell
.\scripts\run-ghost-stack.ps1 -Train                           # Training only
.\scripts\run-ghost-stack.ps1 -Train -WithDashboard             # With live dashboard
.\scripts\run-ghost-stack.ps1 -Train -Datasets "quake3-arenas"  # Dataset filter
```

Full-stack orchestration: kills stale processes, configures environment, launches
training, optionally attaches the passive dashboard.

### Auto-Resume

When no `--checkpoint` is specified, the trainer automatically resumes from
`artifacts/models/latest.pt` if it exists. This enables seamless accumulation
across training sessions — stop training, restart later, and it picks up where
it left off with full optimizer state.

### Checkpoint Behavior

Checkpoints are saved every `checkpoint_every` steps (default 25,000) to
`artifacts/runs/<run_id>/checkpoints/`. After training completes, the final
checkpoint is auto-promoted to the model registry if its `reward_ema` exceeds
the current best.

Full CLI parameter tables, ghost stack options, and overnight training
configuration are documented in
[projects/actor/ — Training](projects/actor/README.md#training).

---

## Inference & Evaluation

### In-Process Inference

Evaluate a trained checkpoint with the same in-process CUDA backend as training,
but without PPO, rollout buffers, or reward shaping:

```powershell
# With dashboard (default)
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\model.pt"

# Deterministic, bounded, no dashboard
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\model.pt" -Deterministic -TotalSteps 10000 -NoDashboard
```

### Model Evaluation

```powershell
# Evaluate a checkpoint with quality metrics
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000

# Compare two checkpoints side-by-side
uv run brain compare ./artifacts/models/v001.pt ./artifacts/models/v002.pt --steps 2000
```

Full inference options and standalone wrappers are documented in
[projects/actor/ — Inference](projects/actor/README.md#inference).

---

## Model Management

Trained models are managed through a registry at `artifacts/models/`:

- **`registry.json`** — Version catalog with metadata for every promoted model
- **`latest.pt`** — Pointer to the best model (highest `reward_ema`)
- **`vNNN.pt`** — Versioned checkpoint copies

```powershell
# Promote a checkpoint to the registry
uv run brain promote ./checkpoints/policy_step_0050000.pt --notes "50K RL run" --tags rl,mamba2

# List all promoted models
uv run brain models
```

### Checkpoint Format (v3)

Every checkpoint contains: `step_id`, `episode_count`, `reward_ema`,
`wall_time_hours`, `parent_checkpoint` (training lineage), `training_source`
(`rl`/`bc`/`inference`), `temporal_core`, `corpus_summary`, and `created_at`.
Only v3 checkpoints are accepted — older formats fail fast with a clear error.

### Auto-Continue & Auto-Promote

- **Auto-continue:** Training resumes from `artifacts/models/latest.pt` when no
  checkpoint is specified, enabling seamless multi-session accumulation.
- **Auto-promote:** After training, the final checkpoint is promoted to the
  registry if it exceeds the current best reward EMA.
- **Nightly promotion:** Successful nightly validation runs auto-promote their
  best checkpoint with a `nightly` tag.
- **Lineage tracking:** Every checkpoint records `parent_checkpoint`, maintaining
  full training lineage from BC pre-training through RL accumulation.

---

## Live Dashboard & Observability

The dashboard is a **passive observer** — it subscribes to the actor's ZMQ PUB
stream without affecting training throughput.

```powershell
.\scripts\run-dashboard.ps1    # Attach to a running trainer
```

### Dashboard Behavior

- Displays **actor 0** observations (no selector mechanism)
- Renders a direct centered 180° half-sphere slice from the canonical `256×48`
  observation (exact `128×48` forward hemisphere before viewport scaling)
- Shows active actor count in the status bar
- Shows observation freshness (`Obs=XXms`) and SPS throughput
- Auto-detects mode:
  - **TRAINING** — triggered by `actor.training.*` telemetry events
  - **INFERENCE** — triggered by `actor.inference.*` events
  - **OBSERVER** — default state when no actor telemetry is present
- Handles missing streams gracefully (WAITING state, no crash)
- Split-socket architecture: dedicated `zmq.CONFLATE` observation socket for
  latest-frame guarantee, separate telemetry socket for ordered metrics

### Resilience

The dashboard and all gallery tools are designed to be operational independent
of simulation and brain layers. They handle missing ZMQ streams, connection
timeouts, and process restart gracefully. This means you can start the dashboard
before training begins and it will pick up the stream when available.

---

## Behavioral Cloning

Human demonstration capture and supervised pre-training to bootstrap a base
policy before RL fine-tuning.

### Phase 1 — Collect Demonstrations

Fly through scenes using keyboard controls. Each scene auto-closes after
`MaxSteps` and the next opens immediately.

```powershell
.\scripts\run-explore-scenes.ps1                                                # Full corpus
.\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas  # Subset
```

Controls: WASD (movement), Space/Shift (up/down), Q/E (yaw), ESC (skip scene).
Demonstrations auto-record to `artifacts/demonstrations/` as `.npz` files.

### Phase 2 — Train BC Checkpoint

```powershell
.\scripts\run-bc-pretrain.ps1
```

Trains a supervised policy from `.npz` demonstrations using truncated BPTT.

### Phase 3 — Fine-Tune with RL

```powershell
uv run brain promote artifacts\checkpoints\bc_base_model.pt --notes "BC baseline"
.\scripts\run-ghost-stack.ps1 -Train    # Auto-continues from latest promoted model
```

Full BC workflow, demonstration recording options, and parameters are documented
in [projects/actor/ — Behavioral Cloning](projects/actor/README.md#behavioral-cloning)
and [projects/auditor/ — Demonstration Recording](projects/auditor/README.md#demonstration-recording).

---

## Benchmarking

### Environment Throughput Benchmark

Measure raw sdfdag stepping throughput independent of the actor brain:

```powershell
uv run navi-environment bench-sdfdag --gmdag-file scene.gmdag --actors 4 --steps 200
```

### Training Benchmarks

| Surface | Script | Measures |
|---------|--------|----------|
| Temporal core comparison | `.\scripts\run-temporal-compare.ps1` | End-to-end training SPS across Mamba-2, GRU, mambapy |
| Kernel microbenchmark | `.\scripts\run-temporal-bakeoff.ps1` | Isolated forward/backward pass timing |
| Resolution scaling | `.\scripts\run-resolution-compare.ps1` | Throughput vs. observation resolution |
| Actor scaling | `.\scripts\run-actor-scaling-test.ps1` | Optimal parallel actor count for hardware |
| Attribution matrix | `.\scripts\run-attribution-matrix.ps1` | Ablation: disable components to isolate bottlenecks |

Full parameter tables and result formats are documented in
[projects/actor/ — Benchmarking](projects/actor/README.md#benchmarking).

### Benchmark Interpretation

Environment viability at a higher ray count does NOT prove end-to-end trainer
viability. RayViT encoder attention and PPO update memory are the dominant cost
factors at higher resolutions, not the CUDA ray tracer. Benchmarks must always
distinguish environment runtime scaling from actor-side scaling.

---

## Validation & Qualification

### Stack Qualification

One-pass end-to-end proof: dataset audit → bounded training → checkpoint resume
→ replay.

```powershell
.\scripts\qualify-canonical-stack.ps1
```

### Nightly Validation

Comprehensive automated overnight validation (8+ hours):

```powershell
.\scripts\run-nightly-validation.ps1
```

Pipeline: CUDA/runtime preflight → regression suites → bounded qualification →
checkpoint + resume proof → environment drift benchmarks → long-duration soak →
summary emission. Nightly runs produce governed artifacts under
`artifacts/nightly/<run_id>/`.

Details: [projects/actor/ — Validation](projects/actor/README.md#validation--qualification)
and [docs/NIGHTLY_VALIDATION.md](docs/NIGHTLY_VALIDATION.md).

---

## Diagnostics

### GPU Preflight

```powershell
python scripts\check_gpu.py      # CUDA availability + kernel execution
python scripts\check_embree.py   # Embree backend (optional)
```

### Corpus Diagnostics

```powershell
# Shallow header probe (fast — reads .gmdag headers only)
python scripts\diagnose_gmdag_corpus.py

# Deep DAG structural analysis (walks up to 2M nodes per scene)
python scripts\diagnose_gmdag_deep.py
```

Flags: non-finite values, extreme scales, degenerate scenes (all-void,
all-surface), low compression, coarse/fine voxels, far centers.

### Asset Validation

```powershell
uv run navi-environment check-sdfdag --gmdag-file path/to/scene.gmdag
uv run navi-environment check-sdfdag --gmdag-root .\artifacts\gmdag\corpus
uv run navi-environment check-sdfdag --json --gmdag-file path/to/scene.gmdag
```

### Training Log Summarization

```powershell
.\scripts\summarize-bounded-train-log.ps1 -LogPath .\logs\navi_actor_train.log.1
```

Extracts per-step metrics (`sps`, `fwd_ms`, `env_ms`, `opt_ms`) and computes
mean/min/max statistics.

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

Quality gates are mandatory: `ruff`, `mypy --strict`, and `pytest` must pass
for all changes. No legacy wrappers, compatibility shims, or deprecated surfaces
are maintained — only the canonical runtime path.

---

## Artifact Layout

```
artifacts/
├── gmdag/corpus/               Compiled .gmdag scene files + manifest
├── checkpoints/                Stable model checkpoints
├── models/                     Promoted model registry (registry.json + vNNN.pt)
├── demonstrations/             Human flight recordings (.npz)
├── runs/<run_id>/              Per-run governed outputs
│   ├── logs/                   Process logs
│   ├── metrics/                Append-only machine-readable metrics
│   ├── manifests/              Process manifests (params, surfaces, output roots)
│   ├── checkpoints/            Run-scoped model checkpoints
│   └── reports/                Human-readable summaries
├── benchmarks/                 Benchmark results (temporal, resolution, scaling)
├── qualification/              Qualification results
├── nightly/<run_id>/           Nightly validation governed run roots
└── validation/                 Validation artifacts

logs/                           Stable top-level log surface for operator tailing
├── navi_actor_train.log.*      Actor training (rotating, 1MB × 10)
├── navi_actor_infer.log.*      Actor inference (rotating, 1MB × 10)
└── navi_auditor_dashboard.log.*  Dashboard (rotating, 1MB × 10)
```

Every canonical process stamps a shared `run_id` into logs, metrics, manifests,
and checkpoints so whole-run review is correlation-safe. Run-scoped logs mirror
into the active run root while stable top-level `logs/` remain available for
operator tailing.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, runtime boundaries, layer responsibilities |
| [docs/TRAINING.md](docs/TRAINING.md) | Corpus refresh, training algorithms, resume, recovery |
| [docs/ACTOR.md](docs/ACTOR.md) | Sacred cognitive engine: RayViT → Mamba-2 → Episodic Memory → PPO |
| [docs/SIM_TO_REAL_PARITY.md](docs/SIM_TO_REAL_PARITY.md) | Dual-pipeline architecture: simulation training ↔ real-world deployment |
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
| [docs/COMPARISON.md](docs/COMPARISON.md) | Temporal-core comparison results (Mamba-2 vs GRU) |
| [docs/NIGHTLY_VALIDATION.md](docs/NIGHTLY_VALIDATION.md) | Nightly validation pipeline specification |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | SDF/DAG validation standard |
| [docs/PARALLEL.md](docs/PARALLEL.md) | Parallel architecture notes |
| [docs/RENDERING.md](docs/RENDERING.md) | Rendering and visualization internals |
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
profiles are benchmark-viable but limited by RayViT self-attention and PPO
update cost. On the active MX150 (`sm_61`), GPU compute utilization is
structurally limited by eager PyTorch dispatcher overhead — each tensor
operation dispatches a separate CUDA kernel with ~10–100μs Python-side idle gap
between launches.

`torch.compile` (`sm_70+`) is the highest-ROI path to GPU utilization
improvement — it fuses multiple PyTorch operations into single kernels,
eliminating dispatcher gaps. PPO/rollout double-buffer overlap (running
environment steps during PPO backward passes) would eliminate ~1000ms of GPU
idle per PPO window but requires architecture work not yet implemented.

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full analysis.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use,
modify, and distribute this software, provided the original copyright notice and
license text are included in all copies or substantial portions of the software.
