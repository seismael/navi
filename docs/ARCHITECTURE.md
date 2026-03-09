# ARCHITECTURE.md — Ghost-Matrix System Architecture

**System:** Navi — Ghost-Matrix Throughput RL  
**Status:** Active canonical specification  
**Policy:** See [AGENTS.md](../AGENTS.md) for implementation rules and non-negotiables

---

## 1. System Objective & Design Philosophy

Navi is engineered for **ultimate-throughput reinforcement learning** in autonomous
agentic navigation. The architecture fundamentally decouples geometric simulation
from temporal cognition and asynchronous observability. It abandons legacy
rendering pipelines — specifically polygon rasterization, bounding volume
hierarchies, and pixel-grid convolutions — in favour of pure mathematical
proximity evaluation.

### 1.1. The Mathematical Purity Principle

The environment is treated as a continuous scalar field. The cognitive agent is
**geometry-blind**, perceiving the world exclusively through raw, mathematically
exact proximity tensors — spherical depth arrays and semantic class matrices.
Visual aesthetics (RGB textures, lighting, camera images) are structurally
eradicated from the training execution path, enforcing **zero-shot topological
generalization** across unseen environments.

### 1.2. The "No Stall" Mandate

Rendering is strictly diagnostic. The simulation engine must never wait for the
neural network's backward pass. Hardware utilization must achieve maximum parity
between GPU compute cores and the memory bus. Training throughput is the primary
architectural success metric.

### 1.3. The Sacred Engine Principle

The training engine (`CognitiveMambaPolicy`) is **immutable**. It is never
modified to accommodate new simulators, sensor types, or data sources. External
data always connects through a `DatasetAdapter` that transforms *to* the
engine's canonical `(1, Az, El)` DistanceMatrix format. The adapter is the
**only** place where axis transposes, depth normalization, semantic class
remapping, delta-depth computation, and env-dimension insertion are performed.

The temporal stage inside the sacred pipeline remains a single canonical runtime
backend. Backend selection is benchmark-governed and must satisfy cross-platform
support targets (native Windows, Linux, WSL2) and throughput floor requirements.

The active runtime evolution path is therefore below the actor boundary:
`projects/voxel-dag` compiles `.gmdag` assets, `projects/torch-sdf` executes
batched CUDA sphere tracing, and `projects/environment` adapts the output back
to the actor's canonical `(1, Az, El)` `DistanceMatrix` contract.

### 1.3.1. Canonical Training Boundary

Canonical high-throughput training now targets one unified execution surface:

- `projects/environment` still owns simulation backends and observation shaping.
- `projects/actor` still owns PPO, memory, curiosity, and the sacred policy.
- The canonical training CLI instantiates the environment backend directly and
  steps it in-process to remove the Environment<->Actor ZMQ control-path barrier
  from the rollout hot loop.

This is an orchestration-level integration only. It does not relax project
sovereignty for long-running services or change the actor contract. There is no
second canonical training architecture alongside it.

What still remains after that architectural cleanup is internal hot-path
cleanup: canonical training must now remove observation host staging,
action host staging, and per-actor Python transition work inside the single
trainer rather than introducing any new runtime surfaces.

### 1.4. Canonical Launch Commands

Runtime commands are standardized across the repository to keep launches
stateless, reproducible, and aligned with service isolation.

| Scope | Command | Notes |
|------|---------|-------|
| Full stack (inference, voxel) | `./scripts/run-ghost-stack.ps1 -Backend voxel` | Environment + Actor + Dashboard |
| Full stack (inference, mesh) | `./scripts/run-ghost-stack.ps1 -Backend mesh -HabitatScene C:/path/to/scene.glb` | Mesh backend requires scene path |
| Full stack (inference, sdfdag) | `./scripts/run-ghost-stack.ps1 -Backend sdfdag -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag` | Canonical compiled-path runtime using a prebuilt `.gmdag` asset |
| Full stack (training, canonical) | `./scripts/run-ghost-stack.ps1 -Train` | Standard 4-actor fleet on the single canonical in-process sdfdag trainer, using the full discovered corpus by default |
| Actor trainer | `uv run --project projects/actor navi-actor train` | Single canonical training entrypoint with direct in-process sdfdag stepping, full-corpus default, and continuous runtime unless explicitly bounded |
| Environment service | `uv run --project projects/environment environment` | Shortcut for `navi-environment serve` |
| Brain service | `uv run --project projects/actor brain` | Shortcut for `navi-actor serve` |
| Dashboard service | `uv run --project projects/auditor dashboard` | Shortcut for `navi-auditor dashboard` |
| Environment wrapper | `./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560` | Windows wrapper |
| Brain wrapper | `./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560` | Windows wrapper |
| Dashboard wrapper | `./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560` | Windows wrapper |
| Compile mesh assets | `uv run --project projects/environment navi-environment compile-world --source data/scenes/world.ply --output data/scenes/world.zarr` | Canonical mesh-to-Zarr pipeline |
| Compile `.gmdag` assets | `uv run --project projects/environment navi-environment compile-gmdag --source data/scenes/world.glb --output data/scenes/world.gmdag --resolution 2048` | Performance-path compiler orchestration |

For canonical training, the repository should resolve the full discovered scene
corpus by default, compile or refresh `.gmdag` assets as needed, and only use
`-GmDagFile`, manifest, or named-scene inputs as explicit narrowing overrides.

### 1.5. Canonical-Only Compatibility Policy

Ghost-Matrix runs on a strict canonical-only policy:

- Legacy/backward-compatible runtime paths are removed once migration is complete.
- Producers and consumers are updated in lock-step; dual-path routing is not retained.
- Compatibility wrappers/aliases/no-op methods are not preserved.
- Required dependencies fail fast when missing; fallback implementations are not part of canonical runtime.
- Tests and docs must reflect only active, used production paths.

For training specifically, distributed ZMQ stepping is no longer part of the
canonical performance architecture. One direct in-process sdfdag trainer is the
production training surface.

Inside that production training surface, `DistanceMatrix` remains the external
diagnostic and wire contract, but canonical rollout is allowed to keep an
equivalent tensor-native CUDA representation internally. Materializing Python
wire objects every step is now treated as a performance bug, not as an
architectural requirement.

### 1.6. Temporal Backend Selection Policy (Mar 2026)

- Actor temporal-core backend selection is benchmark-gated under strict interface parity `(B,T,D)->(B,T,D)`.
- Candidate comparison tooling is allowed during migration; production runtime stays single-path canonical.
- Migration is accepted only if 4-actor rollout throughput remains `>=60 SPS`.
- Platform support target is first-class native Windows, Linux, and WSL2.
- Current production runtime is `Mamba2TemporalCore` on `mambapy` (single canonical path, no runtime fallback branch).
- Actor PPO runtime is CUDA-only in canonical training mode and must pass CUDA kernel preflight at startup.

---

## 2. Mathematical Foundations

### 2.1. The Signed Distance Field (SDF)

The ideal environment $\Omega$ is defined as a continuous mathematical function:

$$\Omega : \mathbb{R}^3 \to \mathbb{R}$$

For any point $\mathbf{p} \in \mathbb{R}^3$, $\Omega(\mathbf{p})$ returns the
shortest absolute geometric distance to the nearest surface:

- $\Omega(\mathbf{p}) > 0$: the point is in open space.
- $\Omega(\mathbf{p}) = 0$: the point is exactly on a surface boundary.
- $\Omega(\mathbf{p}) < 0$: the point is penetrating a solid mass.

> **Current implementation:** The canonical training runtime now runs through
> `projects/voxel-dag` for offline `.gmdag` compilation and `projects/torch-sdf`
> for CUDA sphere tracing, with `projects/environment` preserving the current
> actor contract. Sparse voxel grids and mesh traversal remain diagnostic or
> regression references only.

### 2.2. Actor Geometric State

The actor is a strict geometric entity defined by:

- $\mathbf{p}$: position vector $(x, y, z)$.
- $\mathbf{q}$: orientation (roll, pitch, yaw Euler angles; quaternion for
  Habitat coordinate mapping).
- $r_{min}$: physical minimum radius enforcing volumetric clearance.

The runtime `RobotPose` dataclass captures full 6-DOF state:
`(x, y, z, roll, pitch, yaw, timestamp)`. Kinematic stepping is handled by
`MjxEnvironment` using body-frame velocity integration with configurable
time delta.

### 2.3. Spherical Observation Model

The agent observes the world through a spherical depth panorama — a
multidimensional tensor of normalized distances along $N$ directional rays
$\hat{d}_i$ radiating from the actor's position.

**Current implementation:** Rays are projected into a uniform 3-channel 2D 
spherical grid with shape `(azimuth_bins, elevation_bins)`, default `(256, 48)`, 
via the `RaycastEngine`.
- Channel 0: Normalized Depth $[0, 1]$
- Channel 1: Semantic Class ID
- Channel 2: Frontier Mask (Valid/Invalid)

The perception engine is the **Ray-ViT Encoder** (§8.4), a Vision Transformer 
that treats patches of this sphere as tokens with fixed spherical positional 
encodings.

---

## 3. Algorithmic Design Decisions

To deliver the extreme frame rates necessary for reinforcement learning,
intersection testing must be hyper-optimized. These sections document why
traditional simulation paradigms introduce unacceptable bottlenecks and what
solutions are adopted or targeted.

### 3.1. Why Not Polygon Meshes / BVH

**Mechanism:** Bounding Volume Hierarchies solve algebraic ray-triangle
intersection tests. Each query descends a tree of axis-aligned bounding boxes
until it reaches individual triangles.

**Fundamental flaws for RL throughput:**

- **Volumetric enforcement is expensive.** To enforce the actor's physical
  volume ($r_{min}$), the engine must perform Continuous Collision Detection
  (CCD) or expand millions of triangles via Minkowski Sums. This destroys
  parallel pipeline efficiency and introduces massive per-frame overhead
  proportional to triangle count.
- **Memory access is irregular.** BVH traversal is pointer-chasing through
  tree nodes with divergent branching — the worst-case access pattern for GPU
  SIMD warps, causing severe warp divergence.
- **Scale is bounded.** Triangle counts for real-world scenes (Matterport3D,
  ScanNet) reach tens of millions, making per-ray intersection expensive even
  with optimized BVH libraries.

> **Current usage:** `MeshSceneBackend` remains a diagnostic and regression
> comparison backend only. Canonical training does not route through CPU mesh
> traversal.

### 3.2. Why Not Dense Voxel Occupancy / 3D DDA

**Mechanism:** The Digital Differential Analyzer (DDA) algorithm steps uniformly
through a discrete 3D occupancy grid, checking each voxel cell for occupation.

**Fundamental flaws for RL throughput:**

- **Resolution-accuracy tradeoff.** If the actor moves in $\epsilon$-unit
  increments, the grid must be partitioned at $\epsilon$ scale, leading to an
  $O(n^3)$ memory explosion. A 1 cm resolution grid for a 100 m³ scene requires
  $10^9$ cells — far exceeding GPU memory.
- **Empty-space waste.** DDA traverses empty space iteratively, cell by cell.
  In typical indoor scenes, 95%+ of the volume is air, yet every empty cell
  still incurs a memory read.
- **No native volume check.** Verifying the volumetric clearance of a 3D sphere
  against a voxel grid requires sampling multiple cells around the actor —
  there is no $O(1)$ solution analogous to the SDF subtraction trick (§4).

> **Current implementation:** The `VoxelBackend` mitigates these limitations
> using a **sparse chunked** voxel grid (`SparseVoxelGrid` — dict-based chunks)
> combined with a `SlidingWindow` that only materializes geometry near the
> actor. This avoids the $O(n^3)$ memory explosion for large worlds. The
> `RaycastEngine` then uses vectorized `np.minimum.at` scatter-reduce projection
> instead of per-ray DDA stepping, trading exact ray-march accuracy for
> throughput by projecting all voxels onto the spherical grid in a single
> vectorized pass. It is retained only for targeted diagnostics and regression
> comparison, not for canonical training.

### 3.3. Target: Sphere Tracing (Ray Marching)

Sphere Tracing exploits the fundamental property of SDFs: the distance
$\Omega(\mathbf{p})$ acts as a guaranteed *safe radius* — the ray can advance by
exactly that distance without intersecting any surface.

For any ray $\mathbf{p} + t\hat{d}$:

1. Evaluate $d = \Omega(\mathbf{p} + t\hat{d})$.
2. Advance instantly: $t \leftarrow t + d$.
3. Repeat until $d < \epsilon$ (surface contact).

**Mathematical superiority over DDA:**

- **Convergence:** Sphere Tracing reaches the target surface in $O(\log n)$ steps
  for empty regions, skipping vast volumes of air *mathematically* rather than
  iteratively. DDA requires $O(n)$ steps proportional to the number of cells
  traversed.
- **Adaptive step size:** Steps are large in open space and automatically
  shrink near surfaces — no resolution parameter to tune.
- **Exact boundaries:** Collision is detected at the continuous mathematical
  surface, not at a discrete grid edge.

> **Status:** Sphere tracing is the active integration target for the GPU
> simulation kernel through the internal `projects/torch-sdf` runtime and
> `projects/voxel-dag` compiler domains.

### 3.4. Foveated Ray Distribution

The ideal observation model casts rays with **variable angular density** —
dense in the central forward vector and sparse in the periphery — optimizing
computational load while maintaining high-fidelity forward collision detection.

This is mathematically superior to a uniform grid because:

- Ray density matches the information value of each direction (forward obstacles
  are critical; rear obstacles are less urgent).
- Total ray count can be reduced 2–4× without degrading navigation performance.
- It naturally maps to the attention-weighted processing of a Vision Transformer.

> **Current implementation:** Uses a uniform `(azimuth_bins, elevation_bins)` 2D
> grid. Foveated distribution is a roadmap optimization tied to the Ray-ViT
> encoder migration (§8.4).

---

## 4. Trivial Volume Handling & Collision Mathematics

A critical requirement is that the actor must not intersect walls, respecting
its minimum spatial dimensions ($r_{min}$). In polygon or discrete voxel
engines, verifying volume clearance of a 3D sphere against arbitrary geometry
is an expensive multi-sample integration task.

Under the SDF paradigm, accounting for the actor's volume is resolved with a
**single scalar subtraction** applied uniformly to the environment equation:

$$\Omega_{eff}(\mathbf{p}) = \Omega(\mathbf{p}) - r_{min}$$

**Consequences:**

- **Observation:** When sphere tracing evaluates $\Omega_{eff}$, rays terminate
  exactly when the *outer boundary* of the actor touches the wall. The depth
  proxy reports $d = 0$ precisely at volumetric contact.
- **Collision detection:** Checking whether the actor is currently in collision
  requires only one function call: $\Omega(\mathbf{p}_{actor}) < r_{min}$. This
  is an $O(1)$ operation executable in nanoseconds.
- **No Minkowski expansion:** Unlike BVH systems which must expand every
  triangle by $r_{min}$ (an $O(n)$ operation per frame), the SDF approach
  requires zero per-geometry computation — the subtraction is applied once at
  query time.

> **Current implementation:** `MjxEnvironment` performs collision checking via
> minimum-depth thresholds on the projected spherical view. `VoxelBackend`
> implements `_constrain_translation()` using nearest-occupied-voxel distance
> with a configurable `barrier_distance` standoff. The exact $\Omega_{eff}$
> subtraction will be native to the future SDF kernel (§8.1).

---

## 5. Layer Architecture

Ghost-Matrix employs three strictly isolated layers connected by canonical
zero-copy message contracts over ZeroMQ. No layer imports another layer's
package — they are sovereign services communicating exclusively via serialized
wire messages.

### 5.1. Simulation Layer — `environment`

Headless mathematical execution. Evaluates the environment and produces
canonical `DistanceMatrix v2` observations.

| Component | Role |
|-----------|------|
| `SimulatorBackend` ABC | Pluggable strategy interface for simulation |
| `SdfDagBackend` | Canonical compiled `.gmdag` runtime backed by `torch-sdf` batched CUDA sphere tracing |
| `VoxelBackend` | Procedural voxel worlds + `RaycastEngine` (diagnostic-only) |
| `HabitatBackend` | Meta `habitat-sim` + `HabitatAdapter` (`DatasetAdapter`) |
| `MeshSceneBackend` | `trimesh` ray-mesh intersection for `.glb`/`.obj`/`.ply` |
| `RaycastEngine` | Vectorized `np.minimum.at` scatter-reduce spherical projection |
| `DistanceMatrixBuilder` | Heading-relative rotation, delta-depth, overhead minimap |
| `MjxEnvironment` | Body-frame kinematic pose stepping |
| `SparseVoxelGrid` | Dict-based sparse chunked voxel storage |
| `SlidingWindow` | Materialises geometry only near the actor |
| `FrustumLoader` + `LookAheadBuffer` | Predictive chunk prefetching |
| `pruning` | Chunk pruning utilities for memory-efficient streaming |
| World generators | Arena, City, Maze, Rooms, Open3D, FileLoader |
| `WorldModelCompiler` | Offline PLY/OBJ/STL → sparse Zarr compilation |

External dataset backends use a `DatasetAdapter` (Protocol) internally to
convert raw sensor data into canonical DistanceMatrix format. The adapter
handles axis transpose, depth normalization, semantic remapping, delta-depth,
valid-mask, and env-dimension insertion. See [SIMULATION.md](SIMULATION.md) for
full detail.

### 5.2. Brain Layer — `actor`

Sacred cognitive engine. Consumes only `depth` and `semantic` from
`DistanceMatrix`. No RGB frames, camera images, or non-canonical fields enter
this engine.

**Cognitive pipeline (5-stage):**

1. **RayViTEncoder** — Transformer Encoder: `(B, 3, Az, El)` → `(B, 128)`
   spatial embedding $z_t$.
2. **RND Curiosity** — frozen target MLP + trainable predictor: intrinsic
   exploration reward from embedding novelty.
3. **EpisodicMemory** — tensor-native cosine ring buffer:
  loop-closure detection via batched on-device similarity search.
4. **Mamba2TemporalCore** — Canonical selective state-space temporal core
  implemented via `mambapy` on this Windows CUDA profile: $O(n)$ linear-time
  temporal integration.
5. **ActorCriticHeads** — Gaussian policy for 4-DOF `[fwd, vert, lat, yaw]` +
   scalar value head.

All runtime modes (smoke test, training, evaluation) use this single cognitive
pipeline. See [ACTOR.md](ACTOR.md) for full detail.

### 5.3. Gallery Layer — `auditor`

Passive subscriber. Reads ZMQ streams without ever blocking the simulation or
training loops. Visualization types (RGB frames, camera images) exist only here
— they are explicitly banned from the canonical training wire protocol.

| Component | Role |
|-----------|------|
| `StreamEngine` | Multi-topic ZMQ subscriber with ring-buffer state management |
| `Recorder` | Records `distance_matrix_v2`, `action_v2`, `telemetry_event_v2` to Zarr |
| `Rewinder` | Replays recorded Zarr sessions via ZMQ PUB |
| `LiveDashboard` | OpenCV-based first-person pseudo-3D depth renderer with teleop controls |
| `GhostMatrixDashboard` | PyQtGraph live selected-actor depth view (actor 0 default), optional actor selector, and compact status-line telemetry (stall, SPS, EMA, episodes, step, optimizer ms, zero-wait) |

---

## 6. The "No Stall" Protocol

### 6.1. Synchronous Batched Step-Mode

The `EnvironmentServer` operates in **step mode** for training.  Each rollout
tick follows a single-round-trip batched protocol:

1. **Actor** stacks all $N$ actors' observations, runs **one** batched forward
   pass `(N, 2, Az, El)` → `(N, 4)` actions.
2. **Actor** sends a single `BatchStepRequest` containing $N$ `Action` objects.
3. **Environment** calls `batch_step()` on the backend, stepping all actors,
   then replies with a single `BatchStepResult` containing $N$ `StepResult`
   + $N$ `DistanceMatrix` observations.
4. **Actor** unpacks per-actor results, processes reward shaping + episodic
   memory, and appends transitions to per-actor `TrajectoryBuffer`s.

This replaces the former per-actor round-robin design where each actor
required its own REQ/REP round-trip, reducing ZMQ latency from $O(N)$ to
$O(1)$ per rollout tick.

Single-actor `StepRequest`/`StepResult` remains available for single-
environment and inference workflows.

An **async mode** is also supported for inference: the server subscribes to
`action_v2`, applies actions continuously, and publishes distance matrices.

### 6.2. Canonical Rollout-Boundary PPO Updates

The measured canonical actor runtime now uses one rollout/update loop:

1. **Parallel rollout:** The simulation engine steps $N_{envs}$ parallel
  environments simultaneously.
2. **Instantaneous inference:** The Actor processes the batch in a single CUDA
  pass, yielding $N_{envs}$ continuous action vectors.
3. **Inline optimization:** Once the rollout buffer is full, PPO and RND
  updates run immediately on that canonical buffer, weights are synced to the
  rollout copy, and the next rollout begins.

> **Status:** Implemented. The earlier background optimizer-worker design was
> removed after tensor-native rollout storage and episodic-memory updates
> showed that coordination overhead, not transport, had become the dominant
> stall source. Zero-wait ratio remains a diagnostic metric, but the canonical
> runtime no longer depends on overlap to make progress.

### 6.3. Execution Topology

The full runtime execution flow spans three phases:

1. **Offline compilation:** `DatasetAdapter` / `WorldModelCompiler` ingest
   arbitrary external formats (`.ply`, `.obj`, `.glb`, procedural algorithms)
   and produce canonical sparse Zarr chunks or voxel grids.
2. **Parallel ray execution:** The simulation backend evaluates the environment
   and produces `(n_envs, Az, El)` spherical depth tensors.
3. **Neural interface:** The depth tensor enters the cognitive pipeline
   directly, with the target of zero CPU-to-GPU data transfer via shared
   PyTorch memory.

---

## 7. Canonical Wire Contracts

The only canonical models permitted on the inter-process wire are:

| Model | Direction |
|-------|-----------|
| `RobotPose` | Embedded in DistanceMatrix |
| `DistanceMatrix` | Simulation → Brain, Simulation → Gallery |
| `Action` | Brain → Simulation, Brain → Gallery |
| `StepRequest` | Brain → Simulation (REQ/REP) |
| `StepResult` | Simulation → Brain (REQ/REP) |
| `TelemetryEvent` | Any → Gallery (PUB/SUB) |

No additional models may be added without explicit approval. No visualization
types may appear in canonical contracts — ever.

### 7.1. Transport Backbone

- **Protocol:** ZeroMQ (ZMQ) over IPC/TCP.
- **Serialization:** msgpack with custom ext types for numpy arrays.
  Topic-prefixed multipart frames: `[topic_bytes, payload_bytes]`.
- **PUB/SUB topics:** `distance_matrix_v2`, `action_v2`, `telemetry_event_v2`.
- **REQ/REP topics:** `step_request_v2` → `step_result_v2`.
- **Default ports:** `5559` (environment PUB), `5560` (environment REP),
  `5557` (actor PUB).

See [CONTRACTS.md](CONTRACTS.md) for full field-by-field specifications.

### 7.2. Telemetry & Logging Architecture

- **Unified Standard:** All console and file logging is strictly centralized through `navi_contracts.logging.setup_logging()`.
- **Cyclic Bounds:** File logs are perpetually capped (RotatingFileHandler, e.g., 1MB chunks, 10-file retention) to ensure infinite uptime without disk bloat.
- **Professional Tracing:** Standardized log formats provide deterministic timestamps, sub-module precision, and level-aligned text for rapid regex analysis.

---

## 8. Roadmap: Future Architecture

The following sections describe the target evolution of the system. All items
are clearly marked as **not yet implemented**. They preserve and upgrade the
theoretical foundations from the original design documents.

### 8.1. SDF Compiler & Sparse Voxel DAG Engine

The simulation layer will migrate from discrete voxel grids to continuous SDF
evaluation via a multi-stage offline compiler pipeline:

```text
DatasetAdapter (Mesh / Pointcloud / Procedural)
  │
  ▼
SdfCompiler (Mathematical distance field evaluation)
  │
  ▼
Sparse Voxel DAG (.gmdag binary cache)
```

**SDF Compiler:** Calculates the exact distance to the nearest geometric
boundary for all points in simulated space, yielding a continuous Signed
Distance Field.

### 8.2. Sparse Voxel DAG Compression

While mathematical equations easily represent infinite spheres or planes,
mapping complex datasets (cave meshes, real-world point clouds) requires
storing distance values in VRAM. Storing a dense 3D float matrix is
prohibitively expensive due to the massive volume of empty air.

**Sparse Voxel Octrees (SVO):** The environment is recursively subdivided. If
a large region contains no geometry and a uniform SDF gradient, subdivision
halts — empty space is represented by a single high-level node, drastically
reducing memory footprint.

**Directed Acyclic Graph (DAG) Deduplication:** Procedural environments (mazes,
corridors) contain highly repetitive structural logic (identical walls, corners).

- An Octree creates a unique memory branch for every wall instance.
- A **DAG** identifies mathematically identical SDF branches across the
  environment and merges their memory pointers.

By folding redundant nodes into a Directed Acyclic Graph, environment memory
consumption is compressed by **orders of magnitude**. This ensures that massive
or functionally infinite environments can reside entirely within the L1/L2
caches of modern GPUs, preventing VRAM bandwidth bottlenecks during parallel
raycasting.

**Binary format:** The compiler outputs `.gmdag` (Ghost-Matrix DAG) binary
cache files. Leaf nodes contain the tuple `[float32 distance, int32 semantic_id]`.

### 8.3. GPU Sphere Tracing Kernel (CUDA/Triton)

A highly optimized, low-level kernel executes sphere tracing across the DAG
structure:

- **Input:** Tensor of actor poses $(x, y, z, \theta)_i$ for $N_{envs}$
  parallel environments.
- **Execution:** Memory access patterns are localized to exploit the DAG's
  aggressive deduplication. The kernel is designed for GPU SIMD warps with
  minimal branch divergence.
- **Normalization:** Raw distances $d$ are clamped and normalized:
  $d_{norm} = d / d_{max}$.
- **Output:** Structured float tensor directly in PyTorch's memory space,
  requiring zero CPU-to-GPU data transfers.

### 8.4. Ray-ViT Encoder

Traditional CNNs are mathematically incompatible with foveated
(variable-density) raycasting because they require uniform 2D grids. The target
encoder is a Vision Transformer (ViT) that natively ingests foveated rays:

- Each ray (or local angular cluster) is linearly projected into a
  high-dimensional token.
- Deterministic positional encoding representing absolute spherical angles
  $(\theta, \phi)$ is added to each token.
- A lightweight Transformer Encoder processes the sequence, allowing rays to
  attend to one another to resolve 3D topology (e.g., a cluster of similar
  distances forming a solid wall).

### 8.5. Tensor-Native Episodic Memory

Canonical training now uses batched tensor cosine search plus fixed-capacity
ring-buffer writes, eliminating the previous GPU -> CPU -> FAISS memory hop in
the rollout loop.

### 8.6. Canonical Inline PPO Updates

`PpoTrainer` now performs PPO updates inline at rollout boundaries after the
rollout buffer is finalized. The previous async double-buffered optimizer path
was removed once measurements showed it had become the dominant coordination
stall on the canonical tensor-native runtime.

---

## 9. Performance Metrics

Architectural success is measured by:

| Metric | Description |
|--------|-------------|
| **Ray throughput** | Rays evaluated per second (aggregate across parallel $N_{envs}$) |
| **Sequence latency** | Inference time for the Encoder + episodic memory + Mamba2 pipeline |
| **Zero-wait ratio** | Fraction of simulation steps not blocked by optimization |
| **Memory efficiency** | Ratio of active geometry to allocated VRAM |
