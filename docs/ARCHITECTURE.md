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

> **Current implementation:** The runtime approximates the SDF via discrete
> sparse voxel grids (`SparseVoxelGrid`) with vectorized spherical projection.
> See §8.1 for the roadmap to continuous SDF evaluation.

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
spherical grid with shape `(azimuth_bins, elevation_bins)`, default `(64, 32)`, 
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

> **Current usage:** `MeshSceneBackend` supports high-throughput training on complex meshes (HSSD, ReplicaCAD) via **batched raycasting**. It leverages `trimesh` with a vectorized query structure to maintain high rollout throughput across multiple actors at 128x24 resolution.

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
> vectorized pass.

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

> **Status:** Sphere tracing is the target architecture for the GPU simulation
> kernel. See §8.1 for the full SDF compiler and kernel roadmap.

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
| `VoxelBackend` | Procedural voxel worlds + `RaycastEngine` (default) |
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
3. **EpisodicMemory** — CPU FAISS `IndexFlatIP` (numpy fallback): KNN
   loop-closure detection via cosine similarity.
4. **Mamba2TemporalCore** — Selective State Space Model (`mamba_ssm` with GRU
   fallback): $O(n)$ linear-time temporal integration.
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
| `GhostMatrixDashboard` | PyQtGraph live dashboard with spherical depth views + 10 PPO metric plots |

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

This replaces the former sequential round-robin design where each actor
required its own REQ/REP round-trip, reducing ZMQ latency from $O(N)$ to
$O(1)$ per rollout tick.

Single-actor `StepRequest`/`StepResult` is still supported for backward
compatibility and inference modes.

An **async mode** is also supported for inference: the server subscribes to
`action_v2`, applies actions continuously, and publishes distance matrices.

### 6.2. Target: Asynchronous Double-Buffered Training

To guarantee that the Actor does not bottleneck the simulation engine:

1. **Parallel rollout:** The simulation engine steps $N_{envs}$ parallel
   environments simultaneously.
2. **Instantaneous inference:** The Actor processes the batch in a single CUDA
   pass, yielding $N_{envs}$ continuous action vectors.
3. **Thread bifurcation:** Upon filling the episodic sequence `TrajectoryBuffer`,
   the pipeline branches:
   - **Optimization thread:** Executes Backpropagation Through Time (BPTT) for
     PPO and RND gradients on Buffer 1.
   - **Simulation thread:** Instantly swaps memory pointers to Buffer 2 and
     continues stepping environments using the latest policy weights —
     zero latency.

> **Status:** Implemented. `PpoTrainer` uses dual `TrajectoryBuffer` sets
> (A/B) with an `_optimisation_worker` background thread. At rollout
> boundaries the simulation thread swaps the active buffer pointer and
> signals the optimisation thread — zero latency. Weight updates are
> guarded by `threading.Lock`; PyTorch inference is lock-free.
> Zero-wait ratio is tracked and logged at the end of each training run.

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

### 8.5. GPU FAISS Episodic Memory

Migration from CPU-based `faiss.IndexFlatIP` to `faiss.StandardGpuResources`,
residing 100% in GPU VRAM. This eliminates PCIe bus latency stalls during
rollout by keeping all KNN lookups on-device.

### 8.6. ~~Asynchronous Double-Buffered Training~~ (Implemented)

> Implemented in `PpoTrainer._optimisation_worker()`. See §6.2 for details.
> The dual-`TrajectoryBuffer` pointer-swapping protocol enables true
> zero-wait optimization where the backward pass never stalls the
> simulation engine.

---

## 9. Performance Metrics

Architectural success is measured by:

| Metric | Description |
|--------|-------------|
| **Ray throughput** | Rays evaluated per second (aggregate across parallel $N_{envs}$) |
| **Sequence latency** | Inference time for the Encoder + FAISS + Mamba2 pipeline |
| **Zero-wait ratio** | Fraction of simulation steps not blocked by optimization |
| **Memory efficiency** | Ratio of active geometry to allocated VRAM |
