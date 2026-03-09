# TODO — Navi Development Roadmap

**Status:** Active
**Last updated:** 2026-02-26

---

## 0. Training Correctness & Architecture Alignment (COMPLETED)

- **Vision Transformer Encoder (Ray-ViT):** Replaced CNN with a Transformer-based patch encoder (§8.4).  
- **Discovery-Focused Rewards:** High exploration bonus (+1.0) and damped velocity in known areas.        
- **Continuous All-Night Training:** Robust single-process scene cycling across 48 datasets.
- **Synchronous PPO Updates:** Fixed threading race conditions and vectorization bugs in raycasting.

---

## 1. Ongoing Monitoring & Iteration

### Issue 1: Missing Faiss Dependency
- **Observation:** `EpisodicMemory` is using the numpy fallback because `faiss` is not found.
- **Evaluation:** Memory query time (`mem`) grows linearly as the buffer fills (from 0.1ms at step 100 to 9.7ms at step 1700). This will eventually choke the PPO trainer loop.
- **Action:** Install `faiss-cpu` (or `faiss-gpu`) in the actor's virtual environment to enable O(1) similarity lookups.

### Issue 2: Mamba2 SSM
- **Status:** Still using GRU fallback (`mamba-ssm` not found).
- **Goal:** Move to a native PyTorch selective SSM implementation to replace GRU without needing the complex `mamba-ssm` CUDA build dependency, or successfully build and install the C++ kernels for Windows.

### Issue 3: Thread Starvation & Gradient Poisoning
- **Observation:** Async PPO was taking 10 minutes and starving the main thread (SPS dropped to 0.4). Agents were stuck in wall death loops getting -5.0 reward forever.
- **Action Taken:** 
    - **RayViT Optimization:** Increased patch size to 8 and reduced layers/heads to speed up CPU inference for 256x48 resolution.
    - **Collision Nudging:** Agents are now nudged 0.5m backwards upon collision to break death loops while keeping temporal memory.
    - **Async Fix:** Moved tensor stacking to the background thread and used granular locking for atomic weight swaps.
    - **Hyperparam Tuning:** Reduced ppo_epochs to 1 and increased minibatch_size to 128 for faster CPU updates.

---

## 2. Parallel Multi-Scene Training (Planned)

**Priority:** High
**Complexity:** Architectural

Design and implement CPU-parallel training across multiple scenes.
