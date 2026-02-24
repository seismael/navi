# TODO — Navi Development Roadmap

**Status:** Active  
**Last updated:** 2026-02-24

---

## 0. Training Correctness Fixes (Feb 22 2026)

**Status:** COMPLETED — all fixes implemented, 205 tests pass (75 actor + 86 environment + 12 contracts + 32 auditor)

### Context
During live monitoring of 4-actor multi-scene sequential training (49 scenes, mesh backend),
analysis from steps 0–26,000 revealed 4 confirmed issues affecting training quality.

- **Issue 1: GAE Truncation Bootstrap Bug — FIXED**
- **Issue 2: Double Collision Penalty — FIXED**
- **Issue 3: RND Exploration Collapse — FIXED**
- **Issue 4: Clip Fraction = 0.00 — RESOLVED**

---

## 1. Training Optimisation & Observability (Feb 24 2026)

**Status:** COMPLETED — all optimisations implemented, dashboard updated.

### Context
Detailed analysis of training metrics and behavior revealed sub-optimal convergence
patterns due to static learning rates and reward scaling imbalances.

### Issue 1: Static Learning Rate — FIXED
- **Problem:** Static LR (3e-4) prevented the policy from settling into high-precision
  navigation maneuvers during late-stage training.
- **Fix:** Implemented Linear LR Annealing for both policy and RND optimizers.
  Default schedule: 3e-4 -> 3e-5 over 500,000 steps.

### Issue 2: Reward Scaling Imbalance — FIXED
- **Problem:** Intrinsic curiosity reward (clamped at [-5, 5]) was dominating
  the extrinsic navigation signals.
- **Fix:** Adjusted `intrinsic_coeff_init` from 1.0 to 0.2. This balances the
  curiosity signal to be roughly equal to navigation progress.

### Issue 3: Missing LR Observability — FIXED
- **Problem:** The dashboard lacked indicators for learning rate and other health metrics.
- **Fix:** Added current LR to telemetry and a dedicated "Learning Rate" chart to the Auditor Dashboard.

---

## 2. Parallel Multi-Scene Training

**Priority:** High  
**Complexity:** Architectural

Design and implement CPU-parallel training across multiple scenes, aggregating
learned knowledge into a single unified policy.

### Requirements
- Spawn `N = num_cpus - 1` independent training workers.
- Implement gradient aggregation (IMPALA or Federate Weight Averaging).
- Knowledge accumulation checkpoints (v2 format) must merge gracefully.
