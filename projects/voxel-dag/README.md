# voxel-dag: The TopoNav Offline Mesh Compiler

`voxel-dag` (Domain II) is an enterprise-grade, framework-agnostic geometric compiler. It is responsible for bridging the gap between discrete polygon meshes (`.obj`, `.glb`) and the continuous mathematical domain of TopoNav. 

The compiler transforms 3D environments into **Signed Distance Fields (SDF)** using the **Fast Sweeping Method (FSM)** and compresses them into a high-performance **Directed Acyclic Graph (DAG)** utilizing a Sparse Voxel Octree (SVO) hierarchy.

---

## 1. Architectural Mandates

This component strictly adheres to the following agnostic design principles:

*   **Mathematical Cubicity:** The compiler dynamically centers the mesh and pads the world into a perfect cube. This ensures a uniform voxel size ($h$) across all axes, eliminating spatial distortion in downstream $O(1)$ sphere-tracing kernels.
*   **Power-of-2 Alignment:** Input resolutions are automatically adjusted to the next power of 2 ($2^n$). This is a hardware-optimization mandate, ensuring spatial subdivisions align perfectly with GPU bit-shifting logic.
*   **Zero-Copy Memory Specification:** The output `.gmdag` binary follows the 64-bit node architecture defined in `docs/MEMORY_LAYOUT_AND_DAG_SPEC.md`, optimized for GPU L1/L2 cache residency.

---

## 2. Key Features

*   **High-Performance FSM Solver:** Multi-threaded (via `numba` or C++) Eikonal solver reaching throughputs of **~0.16 MVoxels/s**.
*   **Intelligent Initialization:** Robust triangle-to-voxel seeding that guarantees mathematical convergence even for non-manifold or zero-thickness geometry.
*   **Aggressive Deduplication:** MurmurHash3-based DAG folding achieving **>5,000x compression** for uniform spatial regions and **>7x** for highly complex, non-convex environments.
*   **Dual-Language Strategy:** Provides a high-performance Python compiler for rapid development and a full C++17 source tree for native deployment.
*   **Repository Test Surface:** Exposes a `voxel_dag.compiler` Python package so monorepo tests can validate compiler/runtime integration without shelling out to the native CLI.

---

## 3. Installation & Requirements

### Python Environment (High-Performance Backend)
*   Python 3.10+
*   Dependencies: `numpy`, `numba`, `pytest`

The repository-local Python package exposes:

```python
from voxel_dag.compiler import MeshIngestor, compute_dense_sdf, compress_to_dag, write_gmdag
```

This surface is used by `projects/voxel-dag` verification tests and by
`projects/torch-sdf` integration tests.

```bash
pip install numpy numba pytest
```

### C++ Build Environment
*   CMake 3.18+
*   C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+)
*   Dependencies: `assimp` (automatically fetched via CMake)

---

## 4. Build & Usage

### Building the C++ Compiler
```bash
cd voxel-dag
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

PowerShell equivalent:
```powershell
cd voxel-dag
if (-not (Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Windows Note:** The executable `voxel-dag.exe` will be located in `build/Release/`. Ensure that the `assimp-vc143-mt.dll` (typically found in `build/_deps/assimp-build/bin/Release/`) is copied to the same directory as the executable before running it.

### Compiling a Mesh (CLI)
```bash
# Run the compiler directly
./build/Release/voxel-dag --input my_mesh.glb --output my_mesh.gmdag --resolution 128
```

PowerShell:
```powershell
.\build\Release\voxel-dag.exe --input my_mesh.glb --output my_mesh.gmdag --resolution 128
```

---

## 5. Testing & Verification

The integrity of the compiler is guarded by two intensive test suites:

### Functional & Scale Suite (`tests/test_suite.py`)
Asserts mathematical accuracy against analytical spheres/planes and verifies $O(N^3)$ scaling.
```bash
python tests/test_suite.py
```

### Intensive Integrity & Stress Suite (`tests/test_integrity.py`)
Stress-tests the compiler using "Swiss Cheese" geometry (50+ internal holes and thin walls) to ensure topological stability and native-level performance.
```bash
python tests/test_integrity.py
```

---

## 6. Performance Benchmarks (Verified)

| Metric | Target Resolution | Performance |
| :--- | :--- | :--- |
| **FSM Throughput** | $128^3$ (2.1M Voxels) | **0.158 MVoxels/s** |
| **Deduplication Rate** | Uniform Space | **>5,400x** |
| **Topological Stability** | Complex Geometry | **Pass** (50+ Edge Cases) |
| **Memory Footprint** | $128^3$ DAG | **~300k Nodes** |

---

## 7. Deployment & Artifacts

The generated `.gmdag` files are the "Source of Truth" for the `torch-sdf` (Domain I) runtime engine. These files contain:
1.  **32-byte Header:** Metadata including resolution, cubic bounding box, and voxel size ($h$).
2.  **Contiguous Node Array:** Flattened 64-bit `uint64` nodes bit-packed with Type Flags, Child Masks, and `float16` distance payloads.

---

## 8. Development Standards

All contributors must adhere to the [**AGENTS.md**](../AGENTS.md) operating manual. The `docs/` directory remains the immutable reference for all memory layouts and API contracts.
