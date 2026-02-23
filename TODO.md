1- # TODO — Navi Development Roadmap

**Status:** Active  
**Last updated:** 2026-02-22

---

## 0. Training Correctness Fixes (Feb 22 2026)

**Status:** COMPLETED — all fixes implemented, 210 tests pass (82 actor + 86 environment + 10 contracts + 32 auditor)

### Context

During live monitoring of 4-actor multi-scene sequential training (49 scenes, mesh backend),
analysis from steps 0–26,000 revealed 4 confirmed issues affecting training quality.
Training was running (reward_ema rose from -1.53 to +1.37 over 12 scenes), but several
correctness bugs were limiting policy improvement.

### Issue 1: GAE Truncation Bootstrap Bug — FIXED

- **File:** `projects/actor/src/navi_actor/rollout_buffer.py`
- **Problem:** `PPOTransition` had a single `done` field; `ppo_trainer.py` merged `done or truncated`
  into it. GAE set `mask=0` for both, zeroing the bootstrap value on truncation instead of
  using V(s_T). This biased returns downward for time-limited episodes, discouraging long episodes.
- **Fix:** Added `truncated: bool = False` field to `PPOTransition`. Restructured GAE loop
  to handle 3 cases separately: truncation (bootstrap from V(s_T), cut trace), true termination
  (future value = 0, cut trace), normal step (full GAE recursion).
- **Tests:** `test_truncation_bootstrap`, `test_truncation_still_cuts_gae_trace`

### Issue 2: Double Collision Penalty — FIXED

- **Files:** `config.py`, `cli.py`, `reward_shaping.py`
- **Problem:** Backend already applies collision penalty (-1.0 mesh, -2.0 voxel) in `StepResult.reward`.
  `RewardShaper` then added another -10.0 on `done=True`. Total: -11.0 per collision.
  This inflated value loss, which dominated the combined loss and crushed the policy gradient signal.
  Symptom: clip_fraction = 0.00 (policy not updating).
- **Fix:** Changed `collision_penalty` default from -10.0 to 0.0 in config, CLI, and RewardShaper.
  Backend's penalty remains the sole collision signal. Parameter kept configurable.
- **Tests:** `test_default_collision_penalty_is_zero`

### Issue 3: RND Exploration Collapse — FIXED

- **Files:** `rnd.py`, `config.py`, `cli.py`, `learner_ppo.py`
- **Problem:** Predictor had 3 linear layers vs target's 2 — strictly more capacity, so it
  memorized the target perfectly within ~500 steps. RND LR was 1e-3 (3.3× policy LR), accelerating
  convergence. RND loss dropped to 0.0000 permanently, killing all curiosity-driven exploration.
- **Fix:** (a) Reduced predictor to 2 layers (matching target architecture — standard RND design
  per Burda et al. 2018). (b) Reduced RND LR from 1e-3 to 3e-5 (10× slower than policy LR).
  Equal capacity + slow learning ensures persistent residual prediction error.
- **Tests:** `test_predictor_matches_target_architecture`, `test_rnd_maintains_nonzero_loss_after_training`

### Issue 4: Clip Fraction = 0.00 — RESOLVED (downstream symptom)

- **Problem:** PPO clipping never activated because the -10.0 collision penalty inflated value loss
  to dominate the combined loss. With `max_grad_norm=0.5`, policy gradients were negligible.
  Ratio stayed ~1.0 across all PPO epochs.
- **Resolution:** No direct fix needed. Removing the double collision penalty (Issue 2) restores
  reward scale, so value loss no longer drowns policy gradients. Revived RND (Issue 3) provides
  additional exploration-driven gradient signal.

### Performance Impact

- Zero additional computation: no new forward passes, tensors, or layers
- RND predictor actually has fewer parameters (removed one 128×128 layer + ReLU)
- Sacred cognitive pipeline untouched

### Post-Fix Action Required

Training must restart from scratch. Existing checkpoints were calibrated for -11.0 collision
penalty and dead RND — continuing with new reward dynamics would cause severe value mismatch.

---

## 1. Parallel Multi-Scene Training

**Priority:** High  
**Complexity:** Architectural

Design and implement CPU-parallel training across multiple scenes, aggregating
learned knowledge into a single unified policy. This is a well-studied problem
in distributed RL (DeepMind IMPALA, OpenAI Rapid, Ray RLlib) and requires
careful mathematical validation.

### Requirements

- Spawn `N = num_cpus - 1` independent training workers, each running a
  separate Environment + Actor pair on a different scene.
- Each worker collects rollouts independently using the shared policy weights.
- Gradient aggregation must be mathematically sound — options include:
  - **Synchronous A2C/PPO:** All workers collect rollouts, gradients are
    averaged before a single optimizer step (IMPALA-style).
  - **Asynchronous gradient averaging:** Workers push gradients to a central
    learner which applies them with staleness correction (A3C-style).
  - **Federated weight averaging:** Workers train locally for K steps, then
    average model weights (FedAvg-style).
- Knowledge accumulation checkpoints (v2 format) must merge gracefully —
  RND running statistics, episodic memory, and reward shaper state must be
  combined across workers.
- Port allocation must be automated (each worker gets unique ZMQ ports).

### References

- IMPALA (Espeholt et al., 2018) — scalable distributed actor-learner
- Ape-X (Horgan et al., 2018) — distributed prioritized experience replay
- Ray RLlib — production-grade distributed RL framework
- FedAvg (McMahan et al., 2017) — federated learning weight aggregation

### Implementation Plan

1. Design port allocation and worker lifecycle management.
2. Implement central parameter server or weight-sharing mechanism.
3. Validate gradient aggregation mathematically (convergence guarantees).
4. Implement RND statistics merging (Chan's parallel algorithm for
   running mean/variance across workers).
5. Test on voxel arena first, then extend to multi-scene habitat training.
6. Benchmark throughput scaling: verify near-linear speedup with worker count.