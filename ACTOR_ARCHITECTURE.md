# Architectural Specification: Cognitive Actor & Advanced RL Training Pipeline

## 1. Executive Summary

This document defines the structural architecture for transitioning the Ghost-Matrix RL agent from a reactive, shallow heuristic model to a **Deep Cognitive Architecture**. By synthesizing Selective State Space Models (Mamba2), Non-Parametric Episodic Memory, and Random Network Distillation (RND), this architecture enables the agent to perform long-horizon pathfinding, recognize loop closures, and autonomously explore structurally complex 3D environments (mazes/caves) with maximum sample efficiency.

---

## 2. Global Pipeline Topology

The Actor subsystem is redesigned as a modular pipeline, decoupling perception, temporal memory, and decision-making. The input is the raw `DistanceMatrix` (v2), and the output is a continuous `Action` distribution.

### The Five-Stage Cognitive Flow

1. **Foveated Perception Encoder:** Compresses the high-dimensional spherical depth matrix into a dense spatial embedding.
2. **RND Curiosity Module:** Evaluates the spatial embedding to generate intrinsic exploration rewards.
3. **Episodic Memory Matrix:** Queries the spatial embedding against historical states to detect loops and append context.
4. **Mamba2 Temporal Core:** Integrates the spatial embedding, episodic context, and historical hidden states into a unified temporal representation.
5. **Actor-Critic Heads:** Projects the temporal representation into velocity commands and state-value estimations.

---

## 3. Detailed Component Architecture

### 3.1. Foveated Perception Encoder (The Spatial Core)

Replaces the 13-feature manual extraction (`navi_actor/spherical_features.py`).

* **Input Tensor:**  where  represent the equirectangular resolution (azimuth and elevation bins) of the `DistanceMatrix` depth and semantic masks.
* **Architecture:** A lightweight Convolutional Neural Network (CNN) or a Vision Transformer (ViT) patch-encoder.
* **Dimensionality Reduction:** Processes the input into a fixed-size latent vector .
* **Objective:** To map identical geometric surroundings to identical latent vectors, regardless of the temporal sequence.

### 3.2. Episodic Memory Matrix (The Path Core)

A non-parametric buffer designed to explicitly solve the "map memory" and loop-closure problems without requiring SLAM.

* **Data Structure:** A discrete K-Nearest Neighbors (KNN) index (e.g., using FAISS) storing the historical spatial embeddings .
* **Buffering Logic:** The immediate past (e.g., the last  steps) is excluded from the query to prevent trivial matches from the agent simply standing still.
* **Retrieval Mechanism:**
At step , the system computes the maximum cosine similarity against the memory buffer:


* **Context Concatenation:** If  (where  is a confidence threshold, e.g., 0.85), the retrieved memory vector  is concatenated with , explicitly informing the downstream sequence model that a loop has occurred.

### 3.3. Mamba2 Temporal Core (The Sequence Engine)

Replaces standard Recurrent Neural Networks (GRUs) or standard Transformers to provide  linear-time sequence modeling.

* **Input:** The concatenated vector .
* **Mechanism:** Implements the Selective State Space equation:



* **Advantage:** The matrices  and  are dynamically parameterized by . The network mathematically learns to filter out irrelevant frames (e.g., staring at a flat wall while moving forward) and strictly updates its hidden state  only at critical junctions or decision points.
* **Output:** A temporal latent representation  encompassing both the current visual state and the relevant historical trajectory.

### 3.4. Actor-Critic Output Heads

* **Actor Head (Policy ):** An MLP processing  to output the parameters  of a Gaussian distribution for both linear velocity (forward) and angular velocity (yaw).
* **Critic Head (Value ):** An MLP processing  to output a scalar estimation of the expected discounted return, .

---

## 4. Reward Shaping & Intrinsic Curiosity Architecture

To drive exploration in vast, procedurally generated environments, the `PpoLearner` must synthesize extrinsic rules with intrinsic motivation.

### 4.1. Extrinsic Penalty Matrix (Environment Physics)

Hard-coded environmental constraints injected into the reward buffer:

* **Collision Termination:** . (Episode terminates instantly).
* **Existential Tax:**  per step. (Forces efficiency).
* **Velocity Heuristic:** . (Encourages forward momentum, penalizes spinning in place).

### 4.2. Intrinsic Curiosity (Random Network Distillation - RND)

Calculated continuously to force the agent into uncharted regions.

* **Target Network ():** Fixed, randomly initialized MLP mapping .
* **Predictor Network ():** Trainable MLP attempting to mimic .
* **Intrinsic Reward:**


* *Behavioral Effect:* Novel rooms produce high prediction error (high reward). Visited rooms produce low error (boredom).

### 4.3. Explicit Loop Penalty

Calculated using the Episodic Memory Matrix similarity score ():


* *Behavioral Effect:* If the agent re-enters a room it visited 500 steps ago, it receives a harsh, immediate penalty, forcing it to reverse course or choose an alternate branch.

### 4.4. Total Reward Formulation

The final scalar reward provided to the PPO advantage estimator is:



*(Note:  is an annealing schedule, decaying as the agent masters the environment).*

---

## 5. Software Implementation Blueprint

To integrate this within the existing `projects/actor` ecosystem securely and strictly:

### 5.1. Refactoring `navi_actor/policy.py`

* **Deprecate:** `ShallowPolicy` and the linear `LearnedSphericalPolicy`.
* **Implement:** `CognitiveMambaPolicy`. This class will instantiate the PyTorch modules for the CNN encoder, the Mamba2 block (via the `mamba-ssm` package), and the Actor-Critic MLPs.
* **State Management:** The policy must now maintain and return its hidden state  to the caller during inference, as the architecture is stateful.

### 5.2. Refactoring `navi_actor/rollout_buffer.py`

* Must be upgraded to store trajectories rather than independent steps. It must capture  sequences to allow for Backpropagation Through Time (BPTT) when training the Mamba2 temporal core.

### 5.3. Implementing `navi_actor/memory/episodic.py`

* A new isolated module encapsulating the FAISS KNN index.
* Must expose an interface to `reset()` at the start of each episode, `add(embedding)`, and `query(embedding)` to compute  and  in sub-millisecond time.

### 5.4. Updating `navi_actor/learner_ppo.py`

* The learner must be expanded to maintain the separate RND Predictor Network optimizer.
* The `train_epoch` function will execute gradient descent on both the PPO surrogate loss (updating the Actor/Critic/Mamba/CNN) and the RND Distillation loss (updating the Predictor).