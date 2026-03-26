# voxel-dag

Offline mesh-to-`.gmdag` compiler for the Navi ecosystem. Transforms 3D scene
meshes into compressed Signed Distance Fields stored as a Directed Acyclic Graph
(DAG) with Sparse Voxel Octree hierarchy.

**Format specification:** [docs/GMDAG.md](../../docs/GMDAG.md)  
**Compiler internals:** [docs/COMPILER.md](../../docs/COMPILER.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Overview

The compiler bridges discrete polygon meshes (`.glb`, `.obj`, and any Assimp-
supported format) to the continuous mathematical domain required by Navi's CUDA
sphere-tracing runtime. The output `.gmdag` binary is the canonical compiled
asset consumed by `torch-sdf` at training time.

Pipeline: **Mesh Ingestion → SDF Generation (FSM) → DAG Compression → `.gmdag` Output**

---

## Key Features

- **Fast Sweeping Method (FSM)** — multi-threaded Eikonal solver (~0.16 MVoxels/s)
- **Deterministic deduplication** — MurmurHash3 candidate grouping with
  structural equality as the correctness authority
- **Mathematical cubicity** — auto-centers and pads the mesh into a perfect cube
  with uniform voxel size across all axes
- **Power-of-2 alignment** — resolutions adjusted to $2^n$ for GPU bit-shift
  compatibility
- **Dual-language strategy** — Python verification compiler for monorepo tests
  plus a C++17 native compiler for production throughput

---

## Installation

### Python Package

```bash
cd projects/voxel-dag
uv sync  # or: pip install -e .
```

Exposes:

```python
from voxel_dag.compiler import MeshIngestor, compute_dense_sdf, compress_to_dag, write_gmdag
```

### C++ Build

```bash
cd projects/voxel-dag
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Requires: CMake 3.18+, C++17 compiler. Assimp is fetched automatically via CMake.

**Windows note:** copy `assimp-vc143-mt.dll` from
`build/_deps/assimp-build/bin/Release/` next to the `voxel-dag.exe` executable.

---

## Usage

### CLI (C++ Compiler)

```bash
./build/Release/voxel-dag --input scene.glb --output scene.gmdag --resolution 512
```

### Via Environment CLI

```bash
uv run --project ../environment navi-environment compile-gmdag \
    --source ../../data/scenes/hssd/102343992.glb \
    --output ../../artifacts/gmdag/corpus/hssd/102343992.gmdag \
    --resolution 512
```

### Via Corpus Refresh

```powershell
./scripts/refresh-scene-corpus.ps1
```

---

## Testing

```bash
# Functional + mathematical accuracy (analytical spheres/planes)
python tests/test_suite.py

# Intensive integrity + stress (Swiss Cheese geometry, 50+ edge cases)
python tests/test_integrity.py
```

---

## Output Format

The generated `.gmdag` files contain:

1. **32-byte header** — resolution, cubic bounding box, voxel size
2. **Contiguous node array** — flattened 64-bit `uint64` nodes bit-packed with
   type flags, child masks, and `float16` distance payloads

See [docs/GMDAG.md](../../docs/GMDAG.md) for the complete binary specification.

---

## Performance

| Metric | Value |
|--------|-------|
| FSM throughput ($128^3$) | ~0.158 MVoxels/s |
| Deduplication rate (uniform space) | >5,400× |
| Memory footprint ($128^3$ DAG) | ~300k nodes |

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
