# SYSTEM_COMPONENTS.md — TopoNav Ecosystem Architecture

## 1. Executive Summary

The TopoNav ecosystem is strictly partitioned into six primary domains to enforce the Open/Closed Principle (SOLID) and guarantee zero architectural bleed between the physical simulation and the cognitive neural network.

The architecture bridges offline C++ geometry compilation, bare-metal CUDA execution, and asynchronous Python networking, culminating in a continuous-space Reinforcement Learning framework.

---

## 2. Domain I: `voxel-dag` (Offline Mesh Compiler)

**Execution:** Pure C++ (CPU-Bound, Offline)
**Responsibility:** Ingest discrete polygon meshes and mathematically compress them into the continuous Ghost-Matrix Directed Acyclic Graph (`.gmdag`) binary format.

### Internal Sub-Components

* **`MeshIngestor`**:
* Integrates `tinyobjloader` to parse `.obj`/`.glb` files.
* Extracts vertex buffers, index buffers, and semantic material IDs.
* Calculates the global bounding box and dynamic voxel resolution ($dx$).


* **`EikonalSolver` (The Distance Transform)**:
* Initializes the narrow-band distances for voxels intersecting triangles.
* Executes the Fast Sweeping Method (FSM) to propagate Euclidean distances outward, solving $\|\nabla f(\mathbf{x})\| = 1$ in $O(V)$ time.


* **`DagCompressor`**:
* Constructs a bottom-up Sparse Voxel Octree (SVO).
* Implements `MurmurHash3` to generate deterministic 64-bit signatures for every spatial node based on its children's pointers.
* Merges identical spatial branches via a global `std::unordered_map`, guaranteeing absolute topological deduplication.


* **`BinarySerializer`**:
* Flattens the DAG into a contiguous `std::vector<uint32_t>`.
* Writes the `.gmdag` file containing the bit-masked topology and semantic payloads.



---

## 3. Domain II: `torch-sdf` (Runtime Compute Engine)

**Execution:** C++ / CUDA (GPU-Bound, Runtime)
**Responsibility:** The agnostic mathematical kernel that exposes $O(1)$ stackless sphere-tracing directly to PyTorch via zero-copy memory bindings.

### Internal Sub-Components

* **`PyTorchBridge` (`bindings.cpp`)**:
* Utilizes `pybind11` and `LibTorch` to expose the C++ API to Python.
* Intercepts `torch::Tensor` inputs and validates contiguous memory alignment and device placement (`is_cuda()`).


* **`MemoryManager`**:
* Loads the `.gmdag` binary directly into the GPU.
* Binds the array to the CUDA Read-Only Data Cache (`__restrict__`) to maximize L1/L2 cache hit rates across millions of parallel point queries.


* **`SphereTraceKernel` (`kernel.cu`)**:
* The heavily optimized CUDA shader. Maps 1 Thread = 1 Ray.
* Executes a stackless point-query traversal of the DAG: $d = f(\mathbf{x})$.
* Advances the ray $\mathbf{x}_{k+1} = \mathbf{x}_k + d \cdot \mathbf{v}$ without thread divergence.
* Writes the final depth and semantic integers directly to the pre-allocated output tensor.



---

## 4. Domain III: `toponav-env` (Simulation Server)

**Execution:** Python (Asynchronous, High-Frequency)
**Responsibility:** Manages the ZMQ network sockets, maintains the state of the physical world, and marshals data into the `torch-sdf` compute engine.

### Internal Sub-Components

* **`ZmqBroker`**:
* Binds to the `REQ/REP` step endpoint and `PUB/SUB` telemetry endpoint.
* Deserializes incoming `BatchStepRequest` payloads.


* **`DynamicsTracker` (Dynamic CSG)**:
* Maintains the coordinate state of non-static entities (other actors, moving obstacles).
* Projects dynamic collision capsules into the `torch-sdf` engine to allow multi-actor occlusion.


* **`StepCoordinator`**:
* Integrates continuous velocity commands into spatial displacement using first-order exponential smoothing (inertia).
* Executes the batched $O(1)$ collision check.
* Invokes the `torch-sdf` raycast and constructs the final `DistanceMatrix`.



---

## 5. Domain IV: `toponav-actor` (The Cognitive Core)

**Execution:** Python / PyTorch (GPU-Bound, Optimization)
**Responsibility:** The Reinforcement Learning pipeline. It maps foveated arrays to continuous velocity actions while maintaining infinite-horizon temporal context.

### Internal Sub-Components

* **`RayViTEncoder`**:
* A Vision Transformer adapted for 1D/2D spherical distance arrays.
* Ingests the `DistanceMatrix` and applies explicit azimuthal and elevational positional embeddings to prevent topological distortion.


* **`TemporalCore` (Mamba-2)**:
* The Selective State Space Model.
* Maintains the recurrent hidden state across the episode without the $O(T^2)$ attention bottleneck of standard transformers.


* **`EpisodicMemory` (FAISS)**:
* The non-parametric loop-detection database.
* Stores historical spatial embeddings and executes $O(1)$ batched queries to generate Information Foraging intrinsic rewards.


* **`PpoLearner` & `MultiTrajectoryBuffer**`:
* The optimization engine. Implements double-buffering and strictly isolated actor trajectory tracking to prevent BPTT Cross-Actor Bleed.



---

## 6. Domain V: `toponav-contracts` (IPC & Data Schemas)

**Execution:** Python (Serialization)
**Responsibility:** Defines the immutable data contracts that flow across the ZeroMQ boundaries, ensuring type-safety between the environment and the neural network.

### Internal Sub-Components

* **`Action` Schema**: Defines the continuous `linear_velocity` and `angular_velocity` $NDArrays$.
* **`DistanceMatrix` Schema**: Defines the $128 \times 24$ `depth`, `semantic`, and `valid_mask` tensors.
* **`TelemetryEvent` Schema**: Defines the decoupled payload used for monitoring reward shapes, loop detection, and system metrics.
* **`Serializer`**: Implements high-speed `msgpack` packing/unpacking to minimize serialization latency over the IPC loopback.

---

## 7. Domain VI: `toponav-auditor` (Telemetry Dashboard)

**Execution:** Python (Asynchronous, GUI)
**Responsibility:** Allows human researchers to monitor the locked-step execution without introducing artificial rendering bottlenecks to the hardware.

### Internal Sub-Components

* **`TelemetrySubscriber`**: Connects to the `PUB/SUB` ZMQ socket and buffers incoming `TelemetryEvent` data.
* **`MatrixVisualizer`**: Renders the 2D equirectangular arrays (Depth and Semantic) into human-readable RGB heatmaps at a decoupled framerate (e.g., 30 FPS), ignoring the underlying $>10,000$ SPS engine throughput.
* **`MetricAggregator`**: Computes moving averages of the Proximal Policy Optimization (PPO) metrics (Policy Loss, Value Loss, Entropy, Steps-Per-Second) and interfaces with logging frameworks (e.g., TensorBoard or Weights & Biases).