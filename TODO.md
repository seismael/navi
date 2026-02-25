# TODO — Navi Development Roadmap

**Status:** Active  
**Last updated:** 2026-02-25

---

## 0. Training Correctness & Architecture Alignment (COMPLETED)

- **Vision Transformer Encoder (Ray-ViT):** Replaced CNN with a Transformer-based patch encoder (§8.4).
- **Discovery-Focused Rewards:** High exploration bonus (+1.0) and damped velocity in known areas.
- **Continuous All-Night Training:** Robust single-process scene cycling across 48 datasets.

---

## 1. Ongoing Monitoring & Iteration

### Issue 1: Fast Scene Cycling
- **Observation:** Scenes switch every `n_actors` episodes. At the start, this is very frequent (~100 steps).
- **Evaluation:** This provides high diversity but might prevent the agent from reaching deep parts of large scenes.
- **Action:** Monitor if the agent learns general navigation. If reward EMA stays low, consider increasing `episodes_per_scene` in `MeshSceneBackend`.

### Issue 2: CPU Throughput
- **Observation:** Current ViT + GRU on CPU achieves ~50-65 SPS.
- **Action:** Profile `RayViTEncoder`. Optimize patch projection and attention heads for CPU vectorization if needed.

### Issue 3: Mamba2 SSM
- **Status:** Still using GRU fallback.
- **Goal:** Move to a native PyTorch selective SSM implementation to replace GRU without needing the `mamba-ssm` CUDA dependency.

---

## 2. Parallel Multi-Scene Training (Planned)

**Priority:** High  
**Complexity:** Architectural

Design and implement CPU-parallel training across multiple scenes.
