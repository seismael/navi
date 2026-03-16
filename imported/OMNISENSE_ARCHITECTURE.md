> Historical imported reference only. This file preserves external design material and is not canonical Navi policy. For current repository behavior, use `docs/SIMULATION.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, and `AGENTS.md`.

# OMNISENSE_ARCHITECTURE.md — 3-Tier Data & Sensor Hierarchy

## 1. Executive Summary

To achieve absolute decoupling between offline dataset processing, high-speed training, and physical hardware deployment, the project implements the 3-Tier **OmniSense** Architecture. This ensures the C++ physics engine remains $O(1)$ while supporting arbitrary external datasets and real-world sensors.

---

## 2. The Sim-to-Real Modality Gap

The training runtime must remain dead simple — it *only* ingests the `.gmdag` format. Introducing dynamic mesh parsing or complex adapters during the PPO training loop would destroy CUDA optimization and collapse SPS. The advanced architecture belongs in a strict DataOps Pipeline and a Sensor Abstraction Layer.

---

## 3. Tier 1: The Universal Asset Factory (`omnisense-data`)

**Execution:** Python (Offline DataOps Pipeline)
**Responsibility:** An ETL (Extract, Transform, Load) pipeline that standardizes arbitrary 3D datasets into the agnostic `.gmdag` format.

### 3.1. The Universal Ontology Standard

A single `ontology.json` defines the absolute truth for the actor's semantic understanding:
- `0`: VOID
- `1`: STATIC_GEOMETRY
- `2`: FLOOR_NAVIGABLE
- `3`: DYNAMIC_OBSTACLE
- `4`: TARGET_OBJECT

### 3.2. The Dataset Translator

The pipeline uses standard Python libraries to:
1. **Load** external meshes (`.glb`, `.obj`, `.ply`)
2. **Standardize Coordinates** — Detect dataset origins and apply rotation matrices to Z-up, right-handed
3. **Translate Semantics** — Map dataset-specific IDs through the ontology translation matrix
4. **Invoke Compiler** — Call the `voxel-dag` CLI to produce `.gmdag`

### 3.3. The Math for Coordinate Standardization

Habitat uses Y-up. TopoNav uses Z-up right-handed. A 90° rotation around the X-axis transforms between them.

---

## 4. Tier 2: The Stochastic Sim Runtime (`omnisense-env`)

**Execution:** Python / PyTorch (Runtime GPU)
**Responsibility:** Injecting physical hardware noise into mathematically perfect C++ raycasts.

### 4.1. Domain Randomization Layer

Runs immediately *after* `torch-sdf` returns distances, *before* ZMQ dispatch:

- **Gaussian Depth Jitter:** $\pm 0.02$ meters standard deviation
- **Ray Dropout:** Random $2\%$–$5\%$ of pixels set to `MAX_DISTANCE` (simulates LiDAR absorption)

This ensures the Mamba-2 actor learns robust recovery policies without slowing the $O(1)$ physics engine.

---

## 5. Tier 3: The Sensor Abstraction Bridge (`omnisense-drivers`)

**Execution:** Python / C++ / ROS2 (Inference Runtime)
**Responsibility:** Projecting real-world 3D Point Clouds into the actor's foveated Sphere.

### 5.1. The Mathematical Projection

For a point $\mathbf{p} = (x, y, z)$:

$$r = \sqrt{x^2 + y^2 + z^2}$$
$$\theta = \arctan2(y, x) \quad \text{(Azimuth)}$$
$$\phi = \arcsin\left(\frac{z}{r}\right) \quad \text{(Elevation)}$$

Map $\theta \in [-\pi, \pi]$ to azimuth bins and $\phi \in [\phi_{min}, \phi_{max}]$ to elevation bins.

### 5.2. Hardware Scenarios

- **Physical 360° LiDAR** (Ouster/Velodyne): Point cloud → spherical projection → DistanceMatrix
- **Multi-Camera RGB-D** (Intel RealSense): Stitch depth maps into cylindrical layout

The neural network never realizes it has left the simulation.

---

## 6. Evaluation & Replay (`omnisense-eval`)

- **Deterministic Seeding:** Load specific `.gmdag`, seed spawn points predictably
- **Replay Buffer:** Save continuous observation, action, trajectory sequences to `.hdf5`
- **Auditor Playback:** Dashboard replays actor's spherical vision and 3D path over time

---

## 7. Why This Architecture Works

1. **Speed:** The C++ engine remains pure math
2. **Scale:** DataOps handles messy external datasets completely offline
3. **Realism:** PyTorch wrapper injects noise for Sim-to-Real robustness
4. **Agnosticism:** Real-world hardware only has to project data into the standard Sphere to become compatible
