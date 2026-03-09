# ARCHITECTURE_OVERVIEW.md — TopoNav Ecosystem

## 1. Executive Summary

**TopoNav** (Topological Navigation) is an enterprise-grade reinforcement learning framework designed to achieve true continuous spatial cognition for autonomous actors.

Legacy spatial simulation heavily relies on graphics-first paradigms (polygon rasterization or BVH raycasting). These methods incur severe compute penalties ($O(\log N)$ or $O(V)$) and are fundamentally misaligned with the mathematical reality of blind spatial routing. TopoNav completely bypasses standard rendering APIs by converting 3D environments into purely mathematical **Signed Distance Fields (SDF)** compressed via **Sparse Voxel Directed Acyclic Graphs (DAG)**.

By offloading the environment into an $O(1)$ bounded continuous field and processing foveated sensor arrays through a **Mamba-2 + Vision Transformer (Ray-ViT)** cognitive core, the architecture achieves a highly accelerated ($>10,000$ SPS) Sim-to-Real training pipeline strictly compliant with Partially Observable Markov Decision Process (POMDP) constraints.

---

## 2. System Taxonomy (The Monorepo)

The architecture is partitioned into strictly isolated domains to enforce the Open/Closed Principle. Inter-process communication operates strictly via predefined memory contracts, decoupling the physics engine from the neural network.

```text
toponav/
├── torch-sdf/           # Domain I: The CUDA Sphere-Tracing Extension (Compute Engine)
├── voxel-dag/           # Domain II: The C++ Offline Mesh Compiler
├── toponav-actor/       # Domain III: The Cognitive Core (Mamba-2 / RayViT)
├── toponav-env/         # Domain IV: The Python Simulator Server
├── toponav-auditor/     # Domain V: Asynchronous Telemetry & Replay Dashboard
└── toponav-contracts/   # Domain VI: ZMQ Schemas & MsgPack Serialization

```

---

## 3. Domain I & II: The Mathematical Physics Engine

The core innovation of the architecture resides in `torch-sdf` and `voxel-dag`. Standard polygon arrays (meshes) are discarded in favor of continuous volumetric equations.

### 3.1. The Offline Compiler (`voxel-dag`)

Arbitrary 3D geometry must be transformed into a compressed memory layout prior to training.

1. **The Eikonal Solver:** The compiler ingests `.glb` or `.obj` files and evaluates the true Euclidean distance field using the Fast Sweeping Method (FSM), solving the Eikonal equation $\|\nabla f(\mathbf{x})\| = 1$ in linear time $O(V)$.
2. **SVO to DAG Folding:** The dense spatial matrix is recursively subdivided into a Sparse Voxel Octree (SVO). A cryptographic hashing function (MurmurHash3) evaluates every node. If two spatial branches are mathematically identical (e.g., repeating hallways), the duplicate is dropped and its pointer redirected.
3. **The Result:** Spatial geometry is compressed by factors exceeding $50\times$, allowing massive topological mazes to reside entirely within the GPU's L1/L2 cache.

### 3.2. The Runtime Execution Kernel (`torch-sdf`)

The evaluation engine operates natively on PyTorch GPU memory allocations.

* **Zero-Copy Architecture:** Using PyBind11 and LibTorch, the execution layer accepts Python PyTorch tensors directly. Mathematical operations modify raw pointers (`.data_ptr<float>()`). Zero data traverses the PCIe bus during the training loop.
* **Stackless Sphere-Tracing:** The CUDA kernel initiates one thread per ray. Threads evaluate the continuous space via point queries against the DAG. Because ray-tracing against an SDF relies solely on coordinate location and local distance ($d = f(\mathbf{x})$), the descent requires no recursive stacks, guaranteeing zero hardware thread divergence.

---

## 4. Domain III: The Cognitive Architecture (`toponav-actor`)

The actor network enforces absolute mathematical parity between the simulation and physical hardware (inference deployment).

### 4.1. POMDP Isolation Protocol

The `toponav-actor` never receives absolute coordinate vectors ($X, Y, Z$) or top-down maps.

* The observation boundary strictly filters the `DistanceMatrix` payload, accepting only `depth`, `semantic`, and `valid_mask` arrays.
* The actor perceives space exactly as a physical depth-camera or LiDAR array would, preventing the network from overfitting to the simulator's global coordinate grid.

### 4.2. Cognitive Topology (Mamba-2 & Ray-ViT)

* **Spatial Encoding:** Spherical foveated arrays are tokenized by a Vision Transformer (`RayViTEncoder`). Azimuth and elevation are injected via absolute positional encodings, permitting asymmetric sensor allocations (e.g., $128 \times 24$ resolution) without geometric distortion.
* **Temporal Horizon:** The sequence of encoded latent vectors is processed by a Mamba-2 Selective State Space Model. This bypasses the catastrophic $O(T^2)$ VRAM explosion of traditional Transformers, granting the actor near-infinite temporal horizon memory for complex maze navigation.
* **Episodic Foraging:** VRAM-locked nearest-neighbor matching (FAISS) evaluates cosine similarity between current and historical spatial embeddings, providing internal auxiliary rewards that eliminate "Void Exploits" and mathematically deter infinite circling.

---

## 5. Inter-Process Communication Layer

The architecture embraces high-frequency lock-step data exchange over ZeroMQ (`toponav-contracts`), effectively preventing Python's Global Interpreter Lock (GIL) from stalling the compute pipelines.

### 5.1. Protocol Topologies

* **Synchronous Lock-Step (REQ/REP):** The primary training loop. `toponav-actor` pushes a `BatchStepRequest` containing continuous velocity floats. `toponav-env` evaluates physical constraints against `torch-sdf` and returns a batched `DistanceMatrix`.
* **Asynchronous Telemetry (PUB/SUB):** `toponav-env` broadcasts the environment state strictly for human observation. `toponav-auditor` drops rendering frames as necessary to accommodate human visual limits without artificially throttling the $>10,000$ SPS hardware capabilities of the core engine.

---

## 6. Performance Guarantees & Hardware Utilization

By shedding standard rasterization APIs and avoiding cross-actor trajectory bleeds in BPTT optimization, the framework establishes the following guarantees on standard hardware configurations:

1. **Throughput:** $O(1)$ sphere-tracing guarantees stable hardware utilization, unbounding the engine from the traditional $O(\log N)$ polygon-intersection CPU limits.
2. **VRAM Integrity:** The static allocation of PyTorch buffers and execution of in-place CUDA mutations ensures a mathematically static memory footprint during the main PPO loop, neutralizing `CUDA Out Of Memory` (OOM) fragmentation.