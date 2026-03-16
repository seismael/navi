> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/SDFDAG_RUNTIME.md`, `docs/SIMULATION.md`, `docs/VERIFICATION.md`, and `AGENTS.md`.

# MEMORY_LAYOUT_AND_DAG_SPEC.md — TopoNav DAG Core

## 1. Executive Summary

To achieve $>10,000$ Steps-Per-Second (SPS) on consumer GPUs, the `TorchSDF` compute engine cannot rely on pointer chasing or dynamically allocated objects. The entire spatial topology must be flattened into a strictly 1-dimensional, contiguous block of memory.

This document defines the **64-bit Node Architecture** for the Ghost-Matrix Directed Acyclic Graph (`.gmdag`). By aggressively bit-packing the tree hierarchy and semantic payloads into `uint64_t` integers, the entire topological map fits into the GPU's L1/L2 Read-Only Data Cache, enabling instantaneous memory broadcasts across parallel CUDA thread warps.

---

## 2. Global Memory Architecture

The DAG is serialized and loaded into VRAM as a single, contiguous array: `std::vector<uint64_t>`.

* **The Pointer Contract:** Memory offsets are strictly relative to the start of this array.
* **Hardware Binding:** In CUDA, this array is passed using the `__restrict__` keyword (e.g., `const uint64_t* __restrict__ dag_memory`). This instructs the compiler to route memory fetches through the constant cache, neutralizing the latency penalty of random memory access during sphere tracing.

---

## 3. The 64-Bit Node Specification (Bit-Packing)

Every node in the DAG is exactly 64 bits (8 bytes). To avoid branching (which causes CUDA thread divergence), the Most Significant Bit (MSB) acts as a hardware switch to define the node's type: **Internal** or **Leaf**.

### 3.1. Node Type Flag

* **Bit 63:** Node Type Indicator
* `0` = Internal Node (Contains pointers to spatial subdivisions)
* `1` = Leaf Node (Contains mathematical distance and semantic payloads)



---

### 3.2. Internal Node Layout (Type `0`)

An internal node defines spatial subdivisions. If a region of space is completely empty or completely solid, it is pruned, and no children exist.

| Bit Range | Size | Field Name | Description |
| --- | --- | --- | --- |
| **63** | 1 bit | `Type` | Must be `0`. |
| **55-62** | 8 bits | `Child Mask` | Bitmask indicating which of the 8 octants exist. If bit $i$ is `1`, the child exists. |
| **32-54** | 23 bits | `Reserved` | Reserved for future spatial hashing or padding. |
| **0-31** | 32 bits | `Child Pointer` | The array index offset pointing to the contiguous block of children. |

**Child Memory Layout:** Children are stored contiguously in memory. If the `Child Mask` is `0b10001001` (Octants 0, 3, and 7 exist), the pointer points to an array of 3 nodes. The query logic simply counts the population of set bits (`__popc` in CUDA) before the target octant to find the exact memory offset of the desired child.

---

### 3.3. Leaf Node Layout (Type `1`)

A leaf node contains the terminal mathematical and semantic truth for a given spatial voxel.

| Bit Range | Size | Field Name | Description |
| --- | --- | --- | --- |
| **63** | 1 bit | `Type` | Must be `1`. |
| **32-62** | 31 bits | `Reserved` | Padding to ensure 64-bit alignment. |
| **16-31** | 16 bits | `Semantic ID` | `uint16_t` mapping to Habitat/ReplicaCAD object IDs (e.g., 1=Wall, 2=Floor). |
| **0-15** | 16 bits | `Distance` | `float16` (IEEE 754 half-precision) Signed Distance Field value. |

**Why `float16`?** High-precision `float32` is mathematically unnecessary for collision and raycasting logic. A `float16` provides sub-millimeter accuracy within typical indoor dimensions ($<65$ meters), saving exactly 50% of the VRAM footprint.

---

## 4. Spatial-to-Memory Mapping (The Mathematics)

To traverse the DAG without a recursive function call, the CUDA shader maps a continuous 3D coordinate $\mathbf{p} = (x, y, z)$ into a discrete octant index $idx \in [0, 7]$.

### 4.1. The Bounding Box

The DAG operates within an absolute bounding box defined by $\mathbf{B}_{min}$ and $\mathbf{B}_{max}$.
The grid resolution $N$ must be a power of 2 (e.g., $N = 1024, 2048$).

### 4.2. Stackless Traversal Logic

For any point $\mathbf{p}$, the shader executes a bounded loop based on the maximum tree depth $D_{max} = \log_2(N)$.

At each depth level $L$, the spatial midpoint $\mathbf{m}$ is calculated. The octant index is derived using non-branching bitwise math:

$$idx = (x \ge m_x) \ll 2 \mid (y \ge m_y) \ll 1 \mid (z \ge m_z)$$

1. **Read Node:** Fetch `uint64_t node = dag_memory[current_ptr]`.
2. **Check Type:** If `node >> 63 == 1`, cast bits 0-15 to a float, and return the distance.
3. **Check Mask:** Evaluate `(node >> 55) & (1 \ll idx)`. If `0`, the space is empty; return the distance to the bounds.
4. **Advance Pointer:** Use intrinsic hardware population count to find the child offset:
`child_offset = __popc((node >> 55) & ((1 \ll idx) - 1))`
`current_ptr = (node & 0xFFFFFFFF) + child_offset`
5. Repeat until a Leaf Node is hit.

---

## 5. Serialization Contract (`.gmdag` Format)

The offline C++ compiler (`voxel-dag`) serializes the processed DAG to disk. The `torch-sdf` engine reads this binary file directly into PyTorch tensors.

The file is strictly little-endian and composed of two sections: the **Header** and the **Payload**.

### 5.1. The Binary Header (32 Bytes)

| Offset | Type | Description |
| --- | --- | --- |
| `0x00` | `char[4]` | Magic Bytes: `G`, `D`, `A`, `G` |
| `0x04` | `uint32` | Format Version (Currently `1`) |
| `0x08` | `uint32` | Resolution $N$ (e.g., 2048) |
| `0x0C` | `float32` | Bounding Box $X_{min}$ |
| `0x10` | `float32` | Bounding Box $Y_{min}$ |
| `0x14` | `float32` | Bounding Box $Z_{min}$ |
| `0x18` | `float32` | Voxel Size $dx$ |
| `0x1C` | `uint32` | Total Node Count ($C$) |

### 5.2. The Payload

Directly following the header, the payload is an array of $C \times 8$ bytes representing the `uint64_t` nodes. This payload is mathematically identical to the layout expected by the CUDA `__restrict__` pointer.