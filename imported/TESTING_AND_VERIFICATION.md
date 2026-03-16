> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/VERIFICATION.md`, `docs/PERFORMANCE.md`, `docs/SDFDAG_RUNTIME.md`, and `AGENTS.md`.

# TESTING_AND_VERIFICATION.md — TopoNav Ecosystem

## 1. Executive Summary

The testing matrix for the TopoNav ecosystem is bifurcated across the language boundary: **GoogleTest (C++)** asserts the absolute mathematical purity of the offline compiler and CUDA execution, while **PyTest (Python)** asserts memory integrity, API contracts, and high-frequency training throughput.

A failure at any layer of this matrix halts the CI/CD pipeline. The engine must guarantee zero VRAM leaks, strict $O(1)$ execution time bounds, and zero-copy memory interoperability before being deployed to the `toponav-actor` reinforcement learning loop.

---

## 2. Layer I: Mathematical & Algorithmic Invariants (C++ GoogleTest)

Before sphere-tracing can be utilized for spatial navigation, the underlying distance transforms must be proven correct against known geometric equations.

### 2.1. The Analytical Sphere Proof

* **Objective:** Validate the Fast Sweeping Method (FSM) and Eikonal solver within the `voxel-dag` compiler.
* **Setup:** Generate a synthetic mathematical sphere mesh of radius $R=5.0$ centered at the origin $(0,0,0)$.
* **Execution:** Compile the mesh into the dense SDF grid.
* **Assertion:** For $100,000$ randomly sampled coordinates $\mathbf{x}$ inside the bounding volume, the computed distance $d(\mathbf{x})$ must satisfy:

$$|d(\mathbf{x}) - (\|\mathbf{x}\| - R)| < \epsilon$$



where $\epsilon$ is strictly bounded by the voxel resolution diagonal ($\sqrt{3} \cdot dx$).

### 2.2. The Cryptographic Deduplication Defense

* **Objective:** Ensure the Directed Acyclic Graph (DAG) compressor never merges geometrically distinct topological branches.
* **Setup:** Construct two identical $16 \times 16 \times 16$ voxel spatial blocks. Alter exactly one voxel's semantic ID in the second block.
* **Execution:** Route both blocks through the `MurmurHash3` node evaluator.
* **Assertion:** Prove `hash(Block_A) != hash(Block_B)`.
* **Failsafe Test:** Deliberately mock the hash function to force a collision. Assert that the deep-equality byte-comparison fallback triggers and rejects the pointer redirection.

---

## 3. Layer II: Topological Edge Cases (C++ GoogleTest)

Real-world meshes (`.glb` or `.ply` scans) frequently contain topological defects. The engine must handle these gracefully without throwing segmentation faults.

### 3.1. Non-Manifold Geometry & "Holes"

* **Objective:** Verify distance generation on infinitely thin planes.
* **Setup:** Ingest a single 2D quad (a flat wall with zero thickness) placed in a 3D volume.
* **Assertion:** The FSM must compute the absolute Euclidean distance to the plane's surface without requiring a defined "inside" or "outside" volumetric hull.

### 3.2. Out-of-Bounds Queries & The Internal Spawn

* **Objective:** Prevent out-of-bounds memory accesses and handle invalid drone spawn coordinates.
* **Setup A (Out of Bounds):** Query the DAG at coordinate $(10^6, 10^6, 10^6)$.
* **Assertion A:** Must safely return the `MAX_DISTANCE` constant, avoiding unallocated memory access.


* **Setup B (Internal Spawn):** Cast a ray from an origin strictly *inside* a solid voxel mass ($d < 0$).
* **Assertion B:** The sphere-tracing kernel must immediately terminate the loop and return a negative distance, triggering the immediate collision penalty in the RL environment.



---

## 4. Layer III: Memory Integrity & Hardware Boundaries (PyTest)

The zero-copy PyTorch bridge is the most critical systems-engineering bottleneck. Silent memory leaks here will instantly crash the GPU during extended training.

### 4.1. The Zero-Copy Pointer Validation

* **Objective:** Prove data never traverses the PCIe bus during environment steps.
* **Setup:** Pass a PyTorch `origins` tensor to the `torch-sdf` extension.
* **Execution:** Extract the C++ memory address via `tensor.data_ptr<float>()`. Modify the tensor using a CUDA kernel.
* **Assertion:** Return the modified tensor to Python. Assert that Python's `.data_ptr()` exactly matches the integer address accessed by C++, confirming an in-place mutation.

### 4.2. The VRAM Leak Detector

* **Objective:** Guarantee absolute memory stability across millions of steps.
* **Setup:** Record the baseline VRAM via `torch.cuda.memory_allocated()`.
* **Execution:** Run the batched `torch_sdf.cast_rays()` function $10,000$ times in a tight loop.
* **Assertion:** The final allocated VRAM must exactly equal the baseline VRAM. A discrepancy of 1 byte constitutes a critical failure.

---

## 5. Layer IV: API Contracts & Failsafes (PyTest)

The user-facing Python API must defensively reject invalid inputs from the RL pipeline before invoking raw CUDA pointers.

### 5.1. Device and Type Traps

* **Objective:** Prevent illegal memory access caused by CPU/GPU mismatch.
* **Setup:** Pass `origins = torch.zeros((64, 3072, 3), device='cpu')` or `dtype=torch.float64` to the C++ backend.
* **Assertion:** The C++ layer (`TORCH_CHECK`) must intercept the invalid tensor and raise a descriptive Python `RuntimeError` before launching the CUDA grid.

### 5.2. Dimensionality Enforcement

* **Objective:** Ensure the raycasting tensor dimensions match the expected spherical distribution.
* **Setup:** Pass an input tensor of shape `(Batch, Rays)` instead of the required `(Batch, Rays, 3)`.
* **Assertion:** Must raise a `ValueError` detailing the expected matrix shape.

---

## 6. Layer V: Performance & Load Boundaries (Benchmarks)

Performance degradation represents a regression in the core architecture. The engine must mathematically outperform legacy rasterizers.

### 6.1. The Throughput Floor ($>10,000$ SPS)

* **Objective:** Validate the speed of the $O(1)$ sphere-tracing kernel under full load.
* **Setup:** Initialize $64$ actors. Allocate $3,072$ rays per actor ($128 \times 24$ resolution).
* **Execution:** Execute $1,000$ batched steps through a highly complex `.gmdag` topology.
* **Assertion:** The total wall-clock execution time for the C++ backend must yield a minimum throughput of $10,000$ Steps-Per-Second.

### 6.2. CUDA Thread Divergence Stress Test

* **Objective:** Ensure the sphere-tracing loop does not cause hardware lockups due to warp divergence.
* **Setup:** In a single batch, place $32$ actors inside a tight enclosed box (rays hit geometry instantly, terminating the loop) and $32$ actors in an infinite open plain (rays iterate until `MAX_STEPS`).
* **Assertion:** The variance in batch execution time must remain bounded, proving that the maximum loop limit (`MAX_STEPS`) correctly prevents hardware stalling.
