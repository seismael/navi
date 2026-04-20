# Sim-to-Real Parity: The Dual-Pipeline Architecture

> Unifying Simulation Training and Hardware Deployment

*Status: Architectural specification (April 2026)*

---

## 1. The Architectural Paradigm: Distance Matrix Parity

A fundamental challenge in embodied AI is the **Reality Gap**. Simulators
generate perfect geometric data at extreme speeds, while real-world hardware
(radar, depth cameras, LiDAR) generates noisy, probabilistic, and conflicting
data at lower frequencies.

If an actor is trained on perfect data but deployed on raw noisy sensors, the
policy will collapse. To solve this, the navi engine enforces **Distance Matrix
Parity**.

The neural network (the Actor) is mathematically isolated from the origin of
its sensory data. It operates against a strict contract: it only ever consumes
an `(n_envs, Az, El)` `DistanceMatrix` tensor — canonically `(B, 256, 48)` in
training. Two entirely distinct mathematical pipelines — one optimised for
infinite training throughput, one designed for real-world sensor fusion —
converge on this identical tensor structure.

See [CONTRACTS.md](CONTRACTS.md) for the full wire-format specification.

---

## 2. Pipeline A: The Simulation Hot-Path (Training)

**Goal:** Maximum theoretical throughput toward the 1,000 SPS current-hardware
target and 10,000 SPS advanced-hardware target.

**Input:** The compiled `.gmdag` spatial index and the Actor's pose
$(\mathbf{R}, \vec{t})$.

In simulation, there is no uncertainty — probabilistic mathematics are
structurally prohibited. The environment is treated as an absolute geometric
oracle.

### 2.1 Spherical Ray Direction LUT

Ray directions are statically pre-calculated into a lookup table via
`build_spherical_ray_directions(azimuth_bins, elevation_bins)` at backend
initialisation. The resulting `(Az × El, 3)` unit-vector array is converted to
a CUDA tensor once and reused across all rollout steps — no per-step
trigonometric recomputation.

### 2.2 CUDA Sphere Tracing

The `torch_sdf.cast_rays()` C++ CUDA kernel performs **stackless DAG
traversal** over the loaded `.gmdag` asset. Each ray advances through the
spatial index using the distance field stored in DAG leaf nodes:

$$P_{\text{next}} = P_{\text{current}} + d_{\text{sdf}} \cdot \hat{r}$$

The actual implementation is significantly more sophisticated than a naive SDF
march:

- **Stackless DAG query** — traverses the bit-packed 64-bit node tree without
  recursion or an explicit stack.
- **Multi-level caching** — shared-memory prefix cache (1024 nodes), void-region
  cache, and leaf cache eliminate redundant DAG traversals.
- **Semantic gating** — hit detection requires `semantic != 0 && dist < ε`
  to distinguish real geometry from void/boundary false positives.
- **Bounded execution** — the kernel terminates on geometry hit, horizon
  (`max_distance`), or iteration limit (`max_steps`).

The Python GIL is released during kernel execution via
`pybind11::gil_scoped_release`.

### 2.3 Tensor Contract

```
Inputs:   origins    [B, R, 3]  float32  CUDA
          dirs       [B, R, 3]  float32  CUDA (normalised unit vectors)
Outputs:  distances  [B, R]     float32  CUDA (preallocated, in-place)
          semantics  [B, R]     int32    CUDA (preallocated, in-place)
```

Where `B` = number of actors, `R` = `Az × El` = `256 × 48 = 12,288` rays at
canonical resolution.

The kernel writes directly to preallocated CUDA tensors. The DAG is loaded to
GPU memory once at asset load time and stays resident. **No CPU bounce occurs
in the canonical training hot path** — distance matrices flow from CUDA kernel
output through PyTorch reshape to the actor's `RayViTEncoder` without host
materialisation.

CPU materialisation (`.cpu().numpy()`) is only triggered on-demand for
diagnostics, dashboard viewing, or replay recording.

### 2.4 Measured Performance

On the active MX150 (`sm_61`) with 4 parallel actors at `256×48`:

| Metric | Value |
|--------|-------|
| Environment batch step | ~36.5 ms |
| End-to-end training (Mamba-2) | ~50 SPS |
| End-to-end training (GRU) | ~100 SPS |
| PPO update cost | ~1,020 ms |

The environment step is **not** the bottleneck — PPO update cost and
actor-side RayViT attention dominate end-to-end wall time. See
[PERFORMANCE.md](PERFORMANCE.md) and
[RESOLUTION_BENCHMARKS.md](RESOLUTION_BENCHMARKS.md) for detailed analysis.

---

## 3. Pipeline B: The Deployment Hot-Path (Real-World Hardware)

> **Status:** Architectural design. Not yet implemented — this section
> describes the planned deployment pipeline.

**Goal:** Resolve real-world sensor noise, hardware conflicts, and multipath
interference into a clean `DistanceMatrix` that the actor cannot distinguish
from simulation output.

**Input:** Asynchronous streams from physical sensors (mmWave radar, depth
cameras, LiDAR).

When deployed on a physical drone, the `.gmdag` oracle does not exist. The
hardware must construct its distance matrix by dynamically fusing sensor data
in real-time.

### 3.1 Log-Odds Bayesian Accumulation

As diverse sensors stream data, the onboard GPU maintains a **Transient
Spherical Voxel Grid** in VRAM. Rather than recording absolute truth, this grid
stores the *probability* of occupancy using log-odds integer addition:

$$\text{Grid}[m] \mathrel{+}= w_{\text{sensor}}$$

This allows instant fusion of conflicting data — for example, when radar
detects a glass door that a depth camera misses.

### 3.2 SDF Synthesis via Jump Flooding Algorithm (JFA)

The neural network cannot ingest a probabilistic log-odds grid; it requires
the standard `DistanceMatrix`. To convert the transient grid into distances,
the GPU executes the **Jump Flooding Algorithm (JFA)**.

JFA is a massively parallel CUDA technique that calculates the distance to the
nearest occupied voxel for every point in the grid in $O(\log N)$ steps,
effectively synthesising an SDF on the fly.

**Estimated performance:** Bound by hardware sensor frequencies (typically
30–60 Hz), with JFA conversion executing in < 2.0 ms on target deployment
hardware.

### 3.3 Output

A `DistanceMatrix` tensor with shape `(1, 256, 48)` in the same coordinate
space and normalisation as Pipeline A output — making the transition from
simulation to reality transparent to the actor.

---

## 4. The Unified Tensor Boundary

Because both pipelines terminate at the exact same tensor shape and coordinate
space, the transition from simulation to reality is seamless:

| Feature | Pipeline A (Training) | Pipeline B (Deployment) |
|:--------|:----------------------|:------------------------|
| **Status** | Production (canonical) | Architectural design |
| **Primary Mathematics** | Stackless DAG sphere tracing | Bayesian fusion + JFA |
| **Epistemology** | Deterministic / absolute | Probabilistic / heuristic |
| **Memory Structure** | Static `.gmdag` (GPU-resident) | Transient spherical VRAM grid |
| **Compute Bottleneck** | ALU ray intersections | Sensor I/O and bus bandwidth |
| **Final Contract** | `DistanceMatrix (n_envs, Az, El)` | `DistanceMatrix (1, Az, El)` |

By isolating the Bayesian logic to the real-world deployment node, the
architecture protects the tensor-native speed of the training loop while
guaranteeing that the drone's neural pathways will recognise the physical
world identically to the simulation.

The actor's sacred cognitive pipeline — `RayViTEncoder` → `TemporalCore` →
`EpisodicMemory` → `ActorCriticHeads` — remains completely immutable
regardless of which pipeline produced the observation.

---

## 5. Relationship to Existing Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Actor (Sacred Brain)          │
                    │  RayViT → Temporal → Memory → Heads     │
                    │         consumes DistanceMatrix          │
                    └────────────────┬────────────────────────┘
                                     │
                            DistanceMatrix(B, Az, El)
                                     │
                    ┌────────────────┴────────────────────────┐
                    │                                         │
           Pipeline A (Training)               Pipeline B (Deployment)
           .gmdag + CUDA sphere trace          Sensor fusion + JFA
           Deterministic, >50 SPS              Probabilistic, 30-60 Hz
           [PRODUCTION]                        [ARCHITECTURAL DESIGN]
```

### Related Documentation

- [CONTRACTS.md](CONTRACTS.md) — Wire-format specification (`DistanceMatrix`, `Action`, `RobotPose`)
- [GMDAG.md](GMDAG.md) — `.gmdag` binary format specification (Pipeline A asset format)
- [SDFDAG_RUNTIME.md](SDFDAG_RUNTIME.md) — CUDA sphere tracing runtime (Pipeline A execution)
- [SIMULATION.md](SIMULATION.md) — Environment runtime, kinematics, reward shaping
- [ACTOR.md](ACTOR.md) — Sacred cognitive pipeline (the consumer of both pipelines)
- [PERFORMANCE.md](PERFORMANCE.md) — Measured throughput and bottleneck analysis

---

*This document was promoted from project notes and validated against the
production codebase. Pipeline A specifications reflect measured implementation;
Pipeline B specifications describe planned deployment architecture.*
