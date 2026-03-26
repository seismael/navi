# `.gmdag` — Ghost-Matrix Directed Acyclic Graph
### A Tensor-Native Spatial Standard for Massive-Scale Reinforcement Learning

---

## 1. The Paradigm Shift: Why `.gmdag`?

In standard Deep Reinforcement Learning (RL) and embodied AI research, spatial
environments are represented using legacy formats borrowed from the video game
and VFX industries: **polygon meshes (.obj, .gltf)**, **dense voxel grids**, or
**standard octrees**.

These formats are structurally hostile to modern GPU hardware when scaled to
thousands of concurrent agents:

* **Meshes + BVH (e.g. Habitat, MuJoCo):** Raycasting requires testing
  intersections against thousands of floating-point triangles.  This causes
  severe **CUDA warp divergence** (threads within a warp branching in different
  directions) and extreme VRAM bandwidth starvation.
* **Dense Voxel Grids:** Provide $O(1)$ lookup time, but memory scales cubically
  $O(N^3)$.  A high-resolution $1024^3$ grid consumes gigabytes of VRAM for a
  single room, making multi-environment or massive-scale memory pinning
  impossible.
* **Standard Octrees:** Save memory, but require "pointer chasing."  A ray must
  navigate down pointers, fetching memory unpredictably, which destroys the
  GPU's L1/L2 cache hit rate.

### The `.gmdag` Solution

The `.gmdag` format is a **hardware-aligned, bit-packed Directed Acyclic
Graph**.  It is not a visual format; it is a *mathematical spatial index* built
explicitly for CUDA integer ALU and bitwise operations.  It achieves the memory
efficiency of a sparse tree, the near-$O(1)$ lookup speed of a dense grid, and
completely eliminates physics engine CPU overhead.

The entire 3D world is collapsed into a single contiguous array of 64-bit
integers — no structs, no pointers, no floating-point vertex coordinates at
runtime.  The GPU decodes spatial relationships using single-cycle bitwise
shifts and the hardware population-count instruction.  This design means the
simulation is bounded **only by GPU memory bandwidth**, not CPU throughput or
polygon complexity.

---

## 2. Mathematical Foundation & Architecture

The core innovation of the `.gmdag` format is **Spatial Hash Deduplication**
combined with **Bit-Packed 64-bit Nodes**.

### 2.1 The "Ghost" Matrix

A standard octree stores every spatial subdivision as a unique node even when
large regions of space are structurally identical — a flat empty wall spanning
100 metres generates millions of identical empty sub-trees.

In a `.gmdag`, the compiler evaluates the geometric topology of every
$2 \times 2 \times 2$ spatial block from the bottom up.  If two blocks are
identical (empty space, solid wall, or an identical corner), the compiler
**hashes** them and merges their pointers.  Instead of storing $O(N^2)$ surface
nodes, the DAG compresses repetitive structures down to near $O(1)$ for
identical macro-patterns.

The GPU addresses the environment as though it were a massive, dense
multidimensional matrix, but physically it only queries a tiny footprint of
deduplicated memory.  This is the "Ghost" Matrix — a phantom address space that
appears dense but is physically sparse.

A naïve $512^3$ voxel grid would consume **1 GB** of raw SDF data.  DAG
deduplication typically compresses a compiled scene to **6–15 MB** (see §5 for
the full deduplication pipeline).

### 2.2 The 64-Bit Node Architecture

To prevent the GPU from wasting cycles reading large data structs, every node in
the `.gmdag` is strictly compressed into a single `uint64_t` (8 bytes).  The bit
layout is designed so that every field can be extracted with a single shift-and-mask
instruction, and the hardware `__popc()` intrinsic replaces all child-index loops
with a single-cycle population count.

The full bit layout is detailed in §3 below.

---

## 3. Binary Layout

### 3.1 Header (32 bytes, packed)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `char[4]` | `magic` | `"GDAG"` (`0x47 0x44 0x41 0x47`) |
| 4 | 4 | `uint32_t` | `version` | Format version (`1`) |
| 8 | 4 | `uint32_t` | `resolution` | Voxel grid resolution (e.g. `512`) |
| 12 | 4 | `float` | `bmin_x` | Bounding box minimum X |
| 16 | 4 | `float` | `bmin_y` | Bounding box minimum Y |
| 20 | 4 | `float` | `bmin_z` | Bounding box minimum Z |
| 24 | 4 | `float` | `voxel_size` | Side length of a single voxel (metres) |
| 28 | 4 | `uint32_t` | `node_count` | Number of 64-bit nodes in the DAG pool |

Python struct format: `<4sIIffffI`.

Only `bmin` is stored — consumers reconstruct the upper bound as
`bmax = bmin + resolution × voxel_size`.  The bounding box is cubified with
a +2 m padding envelope centered on the source mesh geometry.

### 3.2 Node Pool

Immediately after the header: `node_count × 8` bytes of little-endian
`uint64_t` values.  This flat array **is** the DAG — every node, every child
pointer, every leaf distance lives here.

Because the file payload is a strict 1D array of native `uint64_t` values, the
loader performs a single bulk read (`np.fromfile` / `fread`) that maps almost
directly to a `memcpy`.  This is preceded by a 32-byte header parse and binary
integrity validation (magic, version, bounds finiteness, node count
consistency, trailing-byte check) — overhead that is negligible relative to the
bulk transfer but ensures malformed or truncated files cannot silently corrupt
VRAM.  The node array is then transferred to GPU global memory via a single
PCIe `cudaMemcpy` and stays GPU-resident across all simulation steps.

### 3.3 Binary Integrity Validation

The loader rejects files that fail any of these gates:

* Magic bytes ≠ `"GDAG"`.
* Header version ≠ `1`.
* Bounding-box components are not finite.
* Voxel size is not finite or not positive.
* Node count exceeds available payload bytes.
* Trailing bytes exist after the declared node pool.

---

## 4. Node Encoding

Every 64-bit word in the pool is either an **inner node** (spatial router), a
**leaf node** (physical geometry), or a **child pointer** (indirection entry
stored in the contiguous child array of an inner node).

### 4.1 Inner Node (Bit 63 = 0)

```
 63  62        55  54       32  31                 0
┌───┬──────────┬────────────┬───────────────────────┐
│ 0 │ChildMask │  (unused)  │    ChildBase (u32)    │
└───┴──────────┴────────────┴───────────────────────┘
```

| Bits | Width | Field | Description |
|------|-------|-------|-------------|
| 63 | 1 | Leaf flag | `0` → this is an inner node |
| 62–55 | 8 | Child mask | One bit per octant.  `1` = geometry present, `0` = pure void |
| 54–32 | 23 | (reserved) | Zero |
| 31–0 | 32 | Child base | Index into the DAG pool where this node's child pointer array begins |

The child pointer array is stored contiguously in the DAG pool immediately after
allocation.  Only **occupied** octants have entries — the array length equals
`popcount(child_mask)`.  Each entry is a `uint64_t` pool slot whose low 32 bits
hold the index of the corresponding child node in the same pool.

**The Bitwise Lookup Algorithm:** When a ray queries the space, the GPU does
not loop through children.  It uses instantaneous bitwise ALU mathematics.  To
check if child index `idx` (0–7) exists, the CUDA thread executes:

```cpp
bool exists = (child_mask & (1 << idx)) != 0;
```

If it exists, the memory offset is the child pointer plus the number of
populated bits before `idx`, calculated using the hardware-accelerated
`__popc()` intrinsic (single clock cycle on all modern NVIDIA GPUs):

```cpp
int offset = __popc(child_mask & ((1 << idx) - 1));
uint32_t child_ptr = child_base + offset;
```

Empty sibling octants consume zero memory and zero lookup time.

**Octant indexing:** `octant = (x_bit) | (y_bit << 1) | (z_bit << 2)` where
`bit = 1` when the query point is in the upper half of that axis (≥ midpoint).

### 4.2 Leaf Node (Bit 63 = 1)

```
 63  62        32  31       16  15                0
┌───┬───────────┬────────────┬────────────────────┐
│ 1 │  (unused) │ SemanticID │  SDF half-float    │
└───┴───────────┴────────────┴────────────────────┘
```

| Bits | Width | Field | Description |
|------|-------|-------|-------------|
| 63 | 1 | Leaf flag | `1` → this is a leaf node |
| 62–32 | 31 | (reserved) | Zero |
| 31–16 | 16 | Semantic ID | Surface proximity flag: `1` if SDF < 2×voxel_size (near surface), `0` otherwise |
| 15–0 | 16 | Distance | IEEE 754 half-precision float — signed distance to nearest surface (metres) |

The semantic gate is critical for sphere-tracing correctness: only leaves with
`semantic ≠ 0` can register a surface hit, preventing false positives at void
octant boundaries.

---

## 5. DAG Deduplication — Why "Directed Acyclic Graph"

The core mathematical innovation that distinguishes `.gmdag` from a standard
octree is **bottom-up structural deduplication**.  The compiler traverses the
tree from leaves to root, hashing each node's topology.  Structurally identical
sub-trees are merged so that both parents reference a single shared node — this
is what makes the tree a **directed acyclic graph** rather than a simple tree.

### 5.1 Leaf Deduplication

Every leaf node is hashed via **MurmurHash3 x64/128** (a fast, high-quality
non-cryptographic hash designed for speed and excellent distribution) over its
raw 8-byte value.  If two leaves encode the same distance and semantic, they
share a single pool entry.

### 5.2 Inner Node Deduplication

Inner nodes are hashed over their **child pointer array** content.  Two inner
nodes with identical child masks pointing to identical children (by index) are
collapsed to a single pool entry, and both parents reference the shared node.

### 5.3 Additional Compression

- **Void octant pruning:** If the minimum SDF within an octant exceeds
  the octant's diagonal extent (`half_size × voxel_size × √3`), the octant's
  child mask bit is cleared and no child is stored.
- **Early-leaf collapse:** Uniform sub-blocks where all voxel distances are
  within `1e-6` of each other are collapsed to a single leaf at any tree level,
  not only at the bottom.

### 5.4 Compression Result

A naïve $512^3$ voxel grid would consume **1 GB** of raw SDF data.  DAG
deduplication typically compresses a compiled scene to **6–15 MB**, fitting
entirely within the GPU's L2 cache and enabling thousands of concurrent ray
queries without memory bandwidth starvation.

---

## 6. The Compilation Flow (Offline)

The creation of a `.gmdag` file happens via the C++ `voxel-dag` compiler
(`projects/voxel-dag/src/compiler.cpp`).  It translates human-readable geometry
into GPU-native math.

### 6.1 Stage 1 — Voxelization & SDF Generation

The compiler ingests a source mesh via Assimp (`aiProcess_Triangulate |
JoinIdenticalVertices | PreTransformVertices`), supporting any Assimp-importable
format (`.glb`, `.obj`, `.ply`, `.fbx`, and dozens more).  It then computes the
exact Signed Distance Field in two passes:

1. **Seed pass:** For every triangle in the mesh, every voxel within range is
   seeded with the exact parametric point-to-triangle closest-point distance
   (full edge/face case handling).
2. **Eikonal fast sweep:** Eight diagonal sweeps propagate distances outward
   using the standard 3D Eikonal solver, filling the entire $N^3$ grid with
   accurate signed distance values.

### 6.2 Stage 2 — Topological Hashing

The compiler groups voxels into $2 \times 2 \times 2$ blocks from the bottom up.
Each block's topology is hashed with MurmurHash3 x64/128 (returning a 64-bit
digest from the 128-bit body).

### 6.3 Stage 3 — Bottom-Up DAG Merging

Moving up the tree, the compiler looks for hash collisions.  If two
$16 \times 16 \times 16$ sectors of a room hash to the same value, one is
discarded and both parent branches point to the single surviving block.  Void
octant pruning and early-leaf collapse further reduce node count.

### 6.4 Stage 4 — Serialization

The deduplicated graph is flattened into a strict 1D array of `uint64_t`
integers and written to the `.gmdag` file preceded by the 32-byte header.

---

## 7. The Execution Flow (Online Runtime)

Because the file payload is already a 1D array of 64-bit integers, loading is
near-instantaneous: 32-byte header parse + bulk `fread` + a single PCIe
`cudaMemcpy` into GPU global memory.  The DAG stays GPU-resident across all
simulation steps; the CPU never sees geometry again.

### 7.1 Cooperative Shared-Memory Caching

Each CUDA thread block cooperatively loads the first 1024 DAG words (8 KB) into
`__shared__` memory.  Since the top levels of the tree are the most frequently
accessed by every ray, this eliminates global memory traffic for the majority of
traversal steps.  Accesses beyond the cached prefix fall through to global
memory, where the GPU's L1/L2 hardware caches provide further hit-rate benefit.

### 7.2 Stackless Ray Traversal (`query_dag_stackless`)

Standard tree traversal uses a recursive stack (e.g. `traverse(node->child)`).
On a GPU, recursive stacks force memory to spill into local VRAM, destroying
warp-level occupancy and throughput.

`.gmdag` utilizes **stackless traversal**.  As the drone casts a ray, the
coordinates $(x, y, z)$ are mathematically mapped to the octree depth bounds.
The kernel descends the tree using a `while` loop and floating-point
bounding-box math, relying entirely on the GPU's ultra-fast internal registers —
no explicit stack, no DDA grid walk:

**Step 1 — Octant determination.** Compute the midpoint of the current node's
bounding box.  Select the octant index using:

```
octant = (px ≥ mx ? 1 : 0) | (py ≥ my ? 2 : 0) | (pz ≥ mz ? 4 : 0)
```

**Step 2 — Bitwise occupancy check.** Extract the child mask and test the
octant bit:

$$\text{occupied} = \left(\frac{\text{node}}{2^{55}}\right) \mathbin{\&} \left(1 \ll \text{octant}\right)$$

If the bit is zero, the entire sub-cube is known void — the ray advances by the
bounding box size with zero further computation.  This is why raycasting through
empty space costs almost no ALU cycles.

**Step 3 — Popcount child lookup.** When the octant is occupied, the thread
locates the child using the hardware `__popc` instruction (single clock cycle on
all modern NVIDIA GPUs):

$$\text{mask}_\text{lower} = \text{child\_mask} \mathbin{\&} \left((1 \ll \text{octant}) - 1\right)$$

$$\text{child\_index} = \text{child\_base} + \texttt{\_\_popc}\!\left(\text{mask}_\text{lower}\right)$$

This yields the exact array offset of the child pointer without conditional
branching or loop iteration — empty sibling octants consume zero memory and
zero lookup time.

**Step 4 — Leaf decode.** When `bit 63 = 1`, the thread extracts the half-float
distance via `__half2float(node & 0xFFFF)` (single instruction) and the semantic
via `(node >> 16) & 0xFFFF`.

### 7.3 Multi-Level Caching

The traversal maintains three fast-path caches to avoid redundant descent:

| Cache | Purpose | Benefit |
|-------|---------|---------|
| **Prefix cache** (depth 4) | Saves the node pointer and bounds at the 4th tree level | Nearby queries skip the top 4 levels of descent |
| **Leaf cache** | Stores the last hit leaf's bounds, distance, and semantic | A new point in the same leaf voxel returns instantly |
| **Void cache** | Stores the last void octant's bounds | Points still inside the same void skip traversal entirely |

### 7.4 The Deduplication Hardware Dividend

DAG deduplication creates a profound hardware advantage beyond memory savings.
When multiple agents are observing different parts of a structurally similar
region (e.g. a large flat wall), their individual rays query *different* spatial
coordinates — but because the flat wall was deduplicated into a single block by
the compiler, all CUDA threads are physically reading the **same memory
addresses** in VRAM.

Since the deduplicated DAG is small enough to fit in L2 cache (6–15 MB, well
within the L2 of any modern GPU), threads across different warps and SMs serve
from the same cached lines.  Combined with the explicit `__shared__` memory
prefix cache, this means the top levels of the tree are served from on-chip
SRAM (shared memory, ~4 clock cycles) and the lower levels from L2 (~100 clock
cycles) rather than global DRAM (~400+ clock cycles).  This bypasses the VRAM
bandwidth bottleneck entirely.

### 7.5 Sphere Tracing

The outer loop (`sphere_trace_kernel`) marches along each ray in SDF-safe
steps.  At each step it queries the DAG for the signed distance at the current
point and advances `t += distance`.  A hit is registered when
`semantic ≠ 0 && distance < ε` (ε = 0.01 m).  The ray terminates when `t`
exceeds `max_distance` or a hit is found.

---

## 8. End-to-End Pipeline

```
Source Mesh (.glb/.obj/.ply/.fbx — any Assimp-importable format)
    │
    ▼   Assimp loader (aiProcess_Triangulate | JoinIdenticalVertices | PreTransformVertices)
Unified triangle soup (vertices + indices)
    │
    ▼   Exact point-to-triangle SDF + 8-direction Eikonal fast sweep
Signed Distance Field (N³ grid)
    │
    ▼   Recursive octree build + MurmurHash3 DAG deduplication
.gmdag binary (32B header + uint64_t[] node pool)
    │
    ▼   Single PCIe transfer at load time (bulk read + cudaMemcpy)
GPU VRAM (dag_memory) + __shared__ prefix cache (top 1024 nodes, 8 KB)
    │
    ▼   Batched sphere_trace_kernel (N_actors × Az × El rays)
torch.Tensor output — DistanceMatrix [batch, rays] on CUDA
```

### Why This Enables Massive Scale

1. **Complete CPU decoupling.** The CPU never sees geometry.  It sends batched
   ray origins/directions as CUDA tensors and receives distance tensors back.
   The neural network (Mamba-2 SSD / GRU) and the physics engine (`torch-sdf`)
   execute asynchronously on the GPU.  The CPU is completely removed from the
   hot path.
2. **No page faults.** All references are absolute integer indices into a
   contiguous array — the GPU memory controller never triggers page faults
   or TLB misses from pointer chasing.
3. **Perfect tensor output.** The sphere-tracing kernel writes directly
   into a preallocated `torch.Tensor`, producing the `DistanceMatrix`
   observation contract consumed by the actor's `RayViTEncoder` with zero
   intermediate copies.
4. **Cache-friendly deduplication.** DAG compression keeps the entire world
   in L2 cache, allowing thousands of concurrent ray queries without global
   memory bandwidth starvation.

By enforcing the `.gmdag` standard, the simulation is bounded only by the
memory clock speed of the GPU — rendering CPU bottlenecks structurally
impossible.

---

## 9. Competitive Dominance in RL & Sim-to-Real

### 9.1 The Core Problem: The RL Scaling Wall

To understand why `.gmdag` was invented, one must understand the absolute
physical limits of GPU hardware during Deep Reinforcement Learning training.

In a standard embodied AI simulation (e.g. a drone navigating an apartment),
the agent "sees" the world by casting rays — simulating LiDAR or depth cameras.
Effective training demands massive scale:

* **The math:** $N_\text{actors}$ concurrent agents $\times$ $12{,}288$ rays per
  agent (a $256 \times 48$ spherical sensor) = $N_\text{actors} \times 12{,}288$
  physics queries per single environment step.
* At $1{,}000$ SPS on current hardware (the active target), even 4 actors
  produce **~49 million** ray-DAG queries per second.  The distributed-hardware
  target of $10{,}000$ SPS across thousands of agents would push the query rate
  into the hundreds of billions per second.

Legacy 3D formats were engineered by the video game and film industries to
render *light and pixels* for the human eye.  They were never designed to
execute billions of physical distance queries per second for a neural network.
When pushed to these limits, legacy formats cause complete GPU memory bandwidth
starvation.

`.gmdag` was created because massive-scale RL is not a rendering problem; it is
a **highly parallelised, memory-bound tensor mathematics problem.**

### 9.2 Head-to-Head: How `.gmdag` Outperforms Legacy Formats

#### 9.2.1 `.gmdag` vs. Polygon Meshes + BVH (MuJoCo, Habitat, Unreal)

**The legacy flaw.** Physics engines rely on Bounding Volume Hierarchies (BVH)
wrapped around floating-point triangles.  When millions of rays are cast, the
GPU must execute complex floating-point line-plane intersection tests.  Because
rays fan out in 360°, neighbouring GPU threads hit different polygons at
different depths.  This causes catastrophic **CUDA warp divergence** — threads
sit idle waiting for their neighbours to finish complex math on divergent
branches.

**The `.gmdag` advantage.** `.gmdag` contains zero polygons and zero
floating-point intersection math at runtime.  Geometry is pre-calculated into a
spatial SDF grid at compile time.  The GPU uses single-cycle integer bitwise
operations (`__popc()`, bit-shifting) to step through space.  A bitwise shift
executes in one GPU clock cycle; a floating-point polygon intersection takes
dozens of cycles with unpredictable branching.

#### 9.2.2 `.gmdag` vs. Dense Voxel Grids (3D Tensors)

**The legacy flaw.** The fastest way to look up spatial data is a dense 3D
matrix (e.g. `tensor[x][y][z]`).  This gives $O(1)$ memory access.  However,
memory scales cubically $O(N^3)$.  A millimetre-accurate $1024^3$ voxel grid
requires roughly **4 GB of VRAM**.  Training agents across 10 different
environments to prevent overfitting would consume 40 GB of VRAM just for the
maps, leaving no room for rolling out agents or storing the neural network.

**The `.gmdag` advantage.** `.gmdag` uses MurmurHash3 to deduplicate the
spatial grid at compile time.  Empty air, flat walls, and repetitive geometric
patterns are hashed and merged.  That 4 GB dense grid is compressed into a
**6–15 MB** `.gmdag` file.  The near-$O(1)$ lookup speed of a grid is retained
via the three-level traversal cache (§7.3), but thousands of unique,
high-resolution environments fit onto a single consumer GPU simultaneously.

#### 9.2.3 `.gmdag` vs. Standard Pointer-Based Octrees

**The legacy flaw.** Standard octrees save memory, but they require "pointer
chasing."  To find a voxel, the code must read a memory address, jump to it,
read the next address, jump again, etc.  On a GPU, random memory jumps destroy
the L1/L2 cache hit rate.  The SMs (Streaming Multiprocessors) stall while
waiting hundreds of clock cycles for data to arrive from global VRAM.
Furthermore, tree traversal requires a recursive stack, which spills into slow,
thread-local memory on GPUs.

**The `.gmdag` advantage.** Every node is strictly packed into exactly 64 bits
(8 bytes).  There are no structs, no padding, and no wasted bytes — this
enforces perfect memory alignment.  Furthermore, `.gmdag` uses **stackless
traversal** (§7.2) — it steps through the tree using mathematical bounding-box
logic stored entirely in the GPU's ultra-fast internal registers, avoiding local
VRAM stack spills completely.  The explicit `__shared__` memory prefix cache
(§7.1) ensures the most-accessed top levels never touch global memory at all.

#### 9.2.4 `.gmdag` vs. NeRFs / 3D Gaussian Splatting

**The legacy flaw.** Neural Radiance Fields (NeRFs) and Gaussian Splats are
state-of-the-art for *visual rendering*.  However, they represent space
probabilistically.  To determine if a drone hit a wall, a NeRF requires a heavy
neural network forward pass just to estimate surface density — adding latency
and probabilistic uncertainty to every physics query.

**The `.gmdag` advantage.** `.gmdag` is absolute physical truth.  It stores the
exact Signed Distance Field as a half-precision float (IEEE 754 `float16`)
directly in the leaf node's bits 15–0 (§4.2).  There is zero probabilistic
guessing; the physics engine knows the exact distance to the nearest surface
instantly, in a single memory fetch + one `__half2float` instruction.

### 9.3 The Ultimate Standard for Autonomous Discovery

Beyond pure throughput, `.gmdag` is the structurally correct format for training
agents destined for real-world deployment (Sim-to-Real).

#### 9.3.1 The "Native LiDAR" Data Structure

Real-world autonomous drones do not "see" polygons.  They shoot lasers (LiDAR)
or use depth cameras to receive a **Distance Matrix** — a 2D grid of distance
values.

Because `.gmdag` inherently stores the Signed Distance Field of the
environment, raycasting through a `.gmdag` outputs the exact mathematical
tensor structure of a real-world hardware sensor.  The neural network learns on
the exact data format it will receive in physical reality, requiring zero
translation layers between simulation and deployment.  The sphere-tracing
kernel writes directly into a preallocated `torch.Tensor`,
producing the `DistanceMatrix` observation contract consumed by the actor's
`RayViTEncoder` with zero intermediate copies.

#### 9.3.2 The Self-Healing Traversal Cache

Because the SDF value is baked into the bits of every `.gmdag` leaf (bits
15–0), the algorithm inherently acts as its own spatial acceleration structure.
If an agent is in the middle of a large empty room, the queried leaf node
explicitly tells the ray: *"The nearest geometry is 4.5 metres away."*  The
sphere-tracing outer loop instantly advances `t += 4.5` in a single
mathematical step, skipping millions of empty spatial voxels.  Performance
dynamically accelerates in open spaces — exactly the regions where discovery
and navigation tasks spend most of their time.

Combined with the void cache (§7.3) that remembers the last empty octant
bounding box, consecutive queries inside the same void region return instantly
without even entering the DAG traversal.

### 9.4 Complete CPU Decoupling

Traditional simulators require a Python API to query a C++ physics engine for
collisions, triggering a PCIe data transfer that stalls the pipeline.  `.gmdag`
resides entirely in PyTorch's `cuda` memory space.  The neural network and the
physics engine execute on the GPU asynchronously; the CPU is needed only for
orchestration, never for geometry.

---

## 10. Summary

The `.gmdag` format is not a 3D asset; it is a silicon-optimised spatial index.
By fusing non-cryptographic hash deduplication (DAG), hardware-aligned byte
packing (64-bit nodes), and stackless bitwise traversal with explicit
shared-memory caching, it solves the fundamental memory-bandwidth bottlenecks
of embodied AI.  It allows researchers to treat physical simulation not as a
game-engine problem, but as a pure, high-throughput tensor operation.

You cannot brute-force modern RL with legacy VFX rendering formats.  By
abandoning polygons for bit-packed, deduplicated DAGs, `.gmdag` transforms
physical simulation into pure tensor math, allowing researchers to simulate
centuries of autonomous flight time over a single weekend.

---

## 11. Compilation Reference

| Tool | Command |
|------|---------|
| **Compile single scene** | `uv run navi-environment compile-gmdag --source scene.glb --output scene.gmdag --resolution 512` |
| **Validate compiled asset** | `uv run navi-environment check-sdfdag --gmdag-file scene.gmdag` |
| **Benchmark runtime** | `uv run navi-environment bench-sdfdag --gmdag-file scene.gmdag` |
| **Expand corpus** | `powershell scripts/expand-replicacad-corpus.ps1` |
| **Full corpus refresh** | `powershell scripts/refresh-scene-corpus.ps1` |

Canonical compile resolution: `512` (matches the `256×48` observation contract).

---

*Source of truth: `projects/voxel-dag/src/compiler.cpp`, `projects/torch-sdf/cpp_src/kernel.cu`*
