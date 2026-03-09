# PARALLEL.md — Ghost-Matrix Parallelization & Multi-Actor Topologies

## 1. Executive Summary

This document defines the parallel execution architecture for the Ghost-Matrix ecosystem. Scaling Reinforcement Learning (RL) requires processing massive volumes of experiential data. When deploying multiple actors within the same mathematical environment (the Sparse Voxel DAG), the architecture must explicitly resolve whether these actors function as isolated data-gatherers or as a cooperative cognitive swarm.

This specification investigates the two distinct mathematical paradigms of parallelization—**Distributed Experience Accumulation** and **Swarm Cognition (Multi-Agent RL)**—and defines the exact, most efficient implementation paths for the Signed Distance Field (SDF) Core Engine.

---

## 2. The Physics of Shared Environments (SDF Core Efficiency)

Before defining the cognitive sharing mechanisms, the mechanical efficiency of placing multiple actors in the *same* environment must be established.

In traditional polygon engines, 1,000 actors in one environment cause geometric collision bottlenecks ( entity-to-entity checks). In the Ghost-Matrix SDF architecture, parallelizing actors in a single environment yields a massive computational advantage:

* **Zero Memory Duplication:** A single `.gmdag` cache resides in VRAM.
* **Batched Ray Execution:** The SDF Sphere Tracing kernel accepts a batched tensor of actor origins . It evaluates the foveated rays for all  actors simultaneously in a single highly parallelized CUDA/Metal wavefront.
* **Actor Independence in Math:**  is evaluated per-ray. Actors do not inherently block each other unless explicit actor-to-actor physics are programmed into the distance function.

---

## 3. Paradigm A: Distributed Experience Accumulation (Independent)

**Objective:** Run parallel actors exclusively to accelerate the accumulation of state-action-reward data, solving the sample inefficiency of PPO. Ultimately, only **one independent policy** is deployed during inference.

### 3.1. Theoretical Framework

* **Training Phase:**  actors spawn in the exact same procedural DAG but at different starting coordinates. They act as independent parallel threads exploring the space. They do not share live sensor data.
* **The Global Brain:** All  trajectories are periodically aggregated into a centralized `TrajectoryBuffer`. The PPO learner updates a single, global set of neural network weights () for the Ray-ViT and Mamba-2 core.
* **Inference Phase:** A single actor is deployed. It possesses the synthesized navigation mastery of all training actors but operates autonomously without network communication.

### 3.2. Architectural Implementation

1. **Isolated Episodic Memory:** Each actor  maintains its own isolated historical tensor  within VRAM. Actor A cannot see Actor B's loop closures.
2. **Batched Forward Pass:** The neural network executes a single forward pass on the batched ray tensor .
3. **Efficiency:** This is the industry-standard approach for scaling PPO. It guarantees stable gradient descent because the trajectories remain independent and identically distributed (i.i.d.).

---

## 4. Paradigm B: Swarm Cognition (Multi-Agent RL / Shared Knowledge)

**Objective:** Actors operate simultaneously in the same environment and explicitly share knowledge. What one actor maps is instantly known to the entire swarm. This applies to both training and deployed inference (e.g., a swarm of drones mapping a cave system cooperatively).

### 4.1. Theoretical Framework

This paradigm shifts the architecture from standard RL to **Centralized Training with Decentralized Execution (CTDE)** or a fully **Centralized Swarm**.

* **Shared Latent Memory:** Instead of isolated episodic buffers, the system maintains a **Global Episodic Memory Matrix** .
* **Instantaneous Loop Detection:** When Actor A traverses a corridor, its spatial embeddings  are appended to . If Actor B approaches the same corridor minutes later, its cosine similarity check  triggers a match against Actor A's history. Actor B instantly receives the  penalty and routes away, preventing redundant exploration.

### 4.2. Architectural Implementation

1. **The Global Tensor Memory Index:** A singular tensor-native cosine-similarity store accepts concurrent append operations from all actors.
2. **Cross-Attention Sequence Modeling:** To achieve true swarm intelligence, the Mamba-2 Temporal Core can be modified. Instead of processing  isolated hidden states , the architecture implements a cross-attention layer where Actor A's Ray-ViT output  can attend to Actor B's output  if their absolute mathematical coordinates  are within a specific proximity radius.
3. **Reward Sharing:** The reward function must be modified. If Actor A finds the objective, a discounted intrinsic reward is propagated to the hidden states of nearby actors, reinforcing cooperative swarm formations.

---

## 5. Strategic Comparison & Implementation Directive

Determining which paradigm to implement depends strictly on the final operational deployment of the autonomous system.

| Metric | Paradigm A: Distributed Accumulation | Paradigm B: Swarm Cognition |
| --- | --- | --- |
| **Primary Goal** | Train a single, highly robust agent quickly. | Deploy a cooperative fleet of agents. |
| **Training Speed** | Extremely fast. Independent trajectories yield stable, low-variance PPO gradients. | Complex. Shared rewards and global memory increase gradient variance (Credit Assignment Problem). |
| **Inference Mode** | 1 Actor (Isolated). |  Actors (Synchronized). |
| **Episodic Memory** |  isolated  tensors. | 1 global  tensor. |
| **Algorithm Class** | Standard PPO / Asynchronous Advantage. | MAPPO (Multi-Agent PPO) / QMIX. |

### 5.1. The Performance Bottleneck in Swarm (Paradigm B)

If Paradigm B is selected, the mathematical bottleneck shifts to the **Global Episodic Memory**. Writing to a single shared memory index from 1,024 parallel actors every microsecond requires rigorous tensor locking and atomic operations to prevent race conditions. The  similarity search grows  times faster, requiring aggressive pruning of the global memory matrix to maintain sub-millisecond latency.

### 5.2. Recommended Execution Path

To build a highly robust architectural foundation, **Paradigm A (Distributed Experience Accumulation)** must be implemented first.

It natively exploits the batched tensor operations of the SDF Core Engine and the Mamba-2 sequence model without requiring custom atomic memory locks. Once the Core Engine is successfully producing stable, generalized navigation policies for a single actor via batched training rollouts, the architecture can be gracefully extended into Paradigm B by decoupling the `EpisodicMemory` class into a unified globally-addressable MLX/CUDA tensor.