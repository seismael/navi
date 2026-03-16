> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/SIMULATION.md`, `docs/SDFDAG_RUNTIME.md`, `docs/VERIFICATION.md`, and `AGENTS.md`.

# VOXEL_COMPILER.md — The Universal Asset Factory

## 1. Purpose

The `voxel-dag` C++ compiler acts as the strict gateway between human-designed 3D assets (like Habitat datasets) and the $O(1)$ stackless CUDA execution environment. It converts arbitrary polygon meshes into compressed `.gmdag` binaries.

---

## 2. The 3D Memory Paradox

### 2.1. Raw Polygon Mesh
- **Size:** Small on disk (~5 MB)
- **Problem:** GPU cannot query "distance to wall" efficiently. Requires BVH traversal — $O(\log N)$, memory-fragmented, slow for parallel rays

### 2.2. Dense Voxel Grid (Naïve)
- **Math:** 2 cm resolution for a 40×40×10m house = 2 billion voxels × 4 bytes = **8.0 GB** per house
- **Problem:** 100 houses = 800 GB VRAM → impossible

### 2.3. Ghost-Matrix DAG (The Solution)
- 90% of a house is empty air; only the surface matters
- **SVO + DAG deduplication** collapses 8 GB → **30–50 MB**
- 200 distinct environments fit on a single consumer GPU

---

## 3. Architecture (OOP, C++17)

### 3.1. Directory Structure
```text
voxel-dag/
├── CMakeLists.txt
├── src/
│   ├── main.cpp             # CLI Entry Point (cxxopts)
│   ├── compiler.hpp         # voxeldag::Compiler class
│   └── compiler.cpp         # Full pipeline implementation
└── third_party/
    └── cxxopts.hpp          # Header-only CLI parser
```

### 3.2. Dependencies
- **Assimp** (Open Asset Import Library): Parses `.glb`, `.obj`, `.ply`, `.fbx`
- **cxxopts**: Lightweight header-only CLI argument parser

### 3.3. The `voxeldag::Compiler` Class

Encapsulates the 4-stage compilation pipeline:

1. **`loadMesh()`** — Assimp-based multi-format loader with bounding box computation
2. **`computeSDF()`** — Eikonal equation via Fast Sweeping Method, solving $\|\nabla f(\mathbf{x})\| = 1$ in $O(N)$
3. **`compressToDAG()`** — Recursive SVO construction with FNV hash-based deduplication
4. **`writeBinary()`** — 32-byte packed header + contiguous 64-bit node array

### 3.4. Binary Header Format

```
Offset  Size    Field          Description
0       4       magic          "GDAG"
4       4       version        1
8       4       resolution     Grid N
12      4       bmin_x         Bounding box min X
16      4       bmin_y         Bounding box min Y
20      4       bmin_z         Bounding box min Z
24      4       voxel_size     dx (meters/voxel)
28      4       node_count     Total 64-bit nodes
```

---

## 4. CLI Interface

```bash
voxel-dag -i scene.glb -o scene.gmdag -r 2048
```

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input 3D mesh (`.glb`, `.obj`, `.ply`) | Required |
| `-o, --output` | Output binary (`.gmdag`) | Required |
| `-r, --resolution` | Voxel grid resolution | `2048` |

---

## 5. Build Instructions

### Windows (MSVC)
```powershell
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
```

### Linux (GCC)
```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

Assimp is automatically downloaded via CMake `FetchContent`.
