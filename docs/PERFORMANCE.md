# Ghost-Matrix: Canonical Architecture & Performance Specification

This document serves as the absolute, canonical source of truth for the Navi "Ghost-Matrix" ecosystem. It unifies the structural architecture and the mathematical performance baselines into a single specification.

This architecture supersedes all legacy designs involving discrete voxel stepping, polygon rasterization, bounding volume hierarchies (BVH), and standard convolutional neural networks. It is engineered to achieve the theoretical maximum throughput for Reinforcement Learning (RL) in autonomous spatial navigation.

---

## 1. Executive Summary & Core Philosophies

Ghost-Matrix fundamentally decouples continuous geometric physics from temporal cognition and asynchronous observability. The architecture is bound by two inviolable mandates:

1. **The Mathematical Purity Principle:** The environment is treated strictly as a continuous scalar field. The cognitive agent is **geometry-blind**, perceiving the world exclusively through raw, mathematically exact proximity tensors. Visual aesthetics (RGB, textures, lighting) are structurally eradicated from the training execution path, enforcing a zero-shot topological generalization capability.
2. **The "No Stall" Execution Mandate:** Hardware utilization must achieve absolute parity between the GPU compute cores and the SRAM memory bus. Rendering is strictly diagnostic. The simulation engine operates asynchronously and must never wait for the neural network's backward pass.

Any future architectural proposals must be benchmarked against the metrics and computational complexities defined in this document.

---

## 2. Global System Topology

The architecture is partitioned into four strictly isolated operational domains. Inter-domain communication at runtime is executed exclusively via zero-copy inter-process message contracts over ZeroMQ. Memory pointers are never shared across domain boundaries.

1. **Domain I: Offline Ingestion & Compilation Pipeline.** Adapts heterogeneous external datasets into unified mathematical structures.
2. **Domain II: Core Simulation Engine (Runtime).** Executes highly parallel mathematical sphere-tracing against a compressed memory structure.
3. **Domain III: Cognitive Actor Engine (Brain).** A stateful, VRAM-locked neural pipeline processing 1D sequential ray topologies into continuous spatial actions.
4. **Domain IV: Auditor Engine (Visualization/Replay).** A passive, asynchronous subscriber reconstructing mathematical data into human-readable point-cloud geometries for debugging.

---

## 3. Domain I: Universal Environment Ingestion

To train a generalized agent, the system consumes massive, heterogeneous 3D datasets (e.g., Meta's Habitat, LiDAR point clouds, procedural algorithms). **Raw external datasets never enter the training loop.**

### 3.1. The Adapter Ecosystem (SOLID Conformance)

The ingestion pipeline implements a strict Adapter Pattern (`IDatasetAdapter`) to enforce the Open/Closed Principle.

* **`HabitatAdapter`:** Parses `.glb` / `.gltf` meshes, extracting spatial geometries and semantic scene graphs.
* **`PointCloudAdapter`:** Ingests raw `.las` / `.ply` LiDAR scans, computing continuous surface normals.
* **`ProceduralAdapter`:** Wraps internal generator algorithms, translating algorithmic structures into geometric boundaries.

### 3.2. The SDF Compiler & DAG Compression

Raw geometries are processed offline by the **SDF Compiler**.

1. **Mathematical Integration:** Calculates the exact distance to the nearest geometric boundary for all points, yielding a continuous Signed Distance Field (SDF).
2. **Sparse Voxel DAG Compression:** Empty space is pruned via an Octree structure. The Octree is mathematically folded into a **Directed Acyclic Graph (DAG)** by deduplicating identical structural branches. Target compression ratio is **10:1 to 50:1**.
3. **Binary Output:** Outputs a highly compressed `.gmdag` cache file. Leaf nodes contain the tuple `[float distance, int32 semantic_id]`.

---

## 4. Domain II: Core Simulation Engine (The Mathematical Core)

The runtime environment operates exclusively on the compiled `.gmdag` binary caches.

### 4.1. Sphere Tracing vs. Legacy Simulators

For a foveated array of  direction vectors  originating from actor position , the engine utilizes **Sphere Tracing**.

| Methodology | Ray Intersection Complexity | Memory Scaling | Physical Volume Computation |
| --- | --- | --- | --- |
| **Polygon Meshes (BVH)** |  tree traversal | Linear to geometry detail |  Minkowski sum / CCD |
| **Discrete Voxels (3D DDA)** |  iterative grid stepping |  Cubic explosion | Discrete approximation |
| **Signed Distance Field (SDF)** | ** empty space leaps** | **DAG Compressed (High)** | ** scalar subtraction** |

Because an SDF  guarantees the absolute minimum distance to any surface, a ray advances across empty space instantaneously: .

### 4.2. Trivial Volume Verification

The actor's minimal spatial clearance (radius ) is enforced natively without Minkowski sum expansions:



Collision detection evaluates in nanoseconds: . A distance of  signifies absolute physical contact.

---

## 5. Domain III: Cognitive Actor Architecture

The neural pipeline must ingest mathematical states and calculate gradients without causing a pipeline stall.

### 5.1. Foveated Ray-ViT (The Spatial Core)

* **The CNN Bottleneck:** Convolutional Neural Networks require uniform 2D grids, distorting variable-density spherical ray casts.
* **The Ray-ViT Solution:** A Vision Transformer processes the input tensor  (`[distance, semantic_id]`). Absolute spherical angles  are injected as positional encodings. Output is a latent vector .

### 5.2. VRAM-Locked Episodic Memory (The Path Core)

* **The Markovian Flaw:** Standard PPO agents suffer from spatial amnesia and infinite looping.
* **The Ghost-Matrix Solution:** Maintains a historical buffer of spatial embeddings  entirely within GPU/Unified Memory. Loop detection evaluates cosine similarity . This executes in strictly bounded  parallel matrix operations.

### 5.3. Mamba-2 Temporal Sequence Core

* **The Transformer/GRU Bottleneck:** Self-attention scales , causing VRAM explosion during Backpropagation Through Time (BPTT). GRUs scale  but degrade over long horizons.
* **The Mamba-2 Solution:** Selective State Space Models (SSM) with FlashAttention hardware IO fusion provide infinite-horizon memory. BPTT gradient calculation executes in **** linear time. Output is .

### 5.4. Actor-Critic & Tripartite Reward Shaping

MLPs project  into continuous velocity distributions  and a state-value estimation .

1. **Extrinsic Physics:** Collision termination (), existential tax (), forward momentum ().
2. **Intrinsic Curiosity (RND):** Rewards prediction errors in mathematically unmapped territories: .
3. **Explicit Loop Penalty:** Instantly repels the agent from cyclical paths: .

---

## 6. Canonical Contracts & Transport Topology

Inter-process integration relies on stateless, zero-copy messaging via ZeroMQ.

### 6.1. `DistanceMatrix v2` (The Sacred Tensor)

The canonical format is a 1D sequence array representing the pure mathematical output of the sphere tracing kernel. Visual fields (RGB, minimaps) are strictly prohibited.

* `matrix_shape`: tuple[int] — `(N_rays,)` defining the total foveated ray count.
* `depth`: NDArray[float32] — `(n_envs, N_rays)`, normalized distance.
* `semantic`: NDArray[int32] — `(n_envs, N_rays)`, class IDs.
* `robot_pose`: RobotPose — .

---

## 7. Execution Runtime: The "No Stall" Hardware Protocol

### 7.1. Unified Memory vs. PCIe Target

While functional on discrete NVIDIA GPUs (e.g., RTX 4090), Ghost-Matrix is structurally optimized for Unified Memory architectures (e.g., Apple Silicon M-Series via MLX). Evaluating the SDF and executing the Cognitive Actor on shared silicon completely eradicates the PCIe transfer bottleneck.

### 7.2. Asynchronous Double-Buffering

To guarantee  idle time on the simulation compute cores:

1. **Parallel Rollout:** Traces rays for  (or hardware limit) environments simultaneously.
2. **Thread Bifurcation:** Upon filling the `TrajectoryBuffer`, the pipeline branches. The **Optimization Thread** executes BPTT gradients, while the **Simulation Thread** seamlessly swaps to a secondary buffer and continues evaluating the SDF using the latest policy weights.

### 7.3. Strict Latency Budgets

Total end-to-end forward pass must execute in ****:

* SDF DAG Ray Evaluation: 
* Ray-ViT Forward Pass: 
* Episodic Memory Cosine Similarity: 
* Mamba-2 Forward + Actor/Critic: 

---

## 8. The Benchmark Mandate (Architectural Replacement)

This architecture defines the Ghost-Matrix standard. It may not be deprecated or replaced based on subjective preference for legacy pre-compiled simulators. Any proposed replacement architecture must satisfy **ALL** of the following:

1. **Computational Proof:** Must demonstrate a worst-case ray intersection complexity better than or equal to  for empty space, and  volume clearance without CCD.
2. **Temporal Scaling Proof:** Must maintain infinite-horizon contextual memory while guaranteeing  or better scaling during backpropagation.
3. **Empirical Superiority:** Must achieve a statistically significant increase in aggregate **Steps Per Second (SPS)** during active PPO training on locked hardware, navigating the exact same mathematical topology, with identical random seeds.