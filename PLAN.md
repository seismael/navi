# PLAN.md — Canonical Throughput Investigation

This file records the completed tensor-native canonical training migration,
the current 400-SPS gap investigation, and the implementation steps now needed
to attribute the remaining loss between backend-only and end-to-end training.

## 1. Current State

- Canonical actor training already uses one public surface only:
  - `navi-actor train`
  - `scripts/train.ps1`
  - `scripts/train-all-night.ps1`
- Canonical training already runs in-process against `sdfdag` and no longer
  routes through the Environment<->Actor ZMQ control path.
- Canonical training now seeds rollout state from tensor-native `reset_tensor()`
  observations and steps through tensor-native `batch_step_tensor_actions()`.
- Canonical rollout storage, reward shaping, and episodic memory all operate on
  tensors on the canonical path.
- PPO optimization now runs inline at rollout boundaries after the async worker
  reassessment showed overlap had become the dominant coordination stall.
- Direct backend benchmarking has already demonstrated roughly `400 SPS` on the
  same machine/profile, while integrated canonical training still generally
  lands around `50-120 SPS`.
- The active bottleneck question has therefore moved from backend viability to
  actor/runtime attribution on the existing canonical training surface.

## 2. Resolved Bottlenecks

### 2.1 Environment Observation Bounce

- `projects/environment/src/navi_environment/backends/sdfdag_backend.py`
  now exposes tensor-native reset and step seams for canonical training.
- `projects/actor/src/navi_actor/training/ppo_trainer.py` now seeds rollout
  state from `reset_tensor()` and consumes tensor-native step batches directly.
- CPU `DistanceMatrix` materialization remains only for low-volume dashboard and
  telemetry publication.

### 2.2 Actor Action And Embedding CPU Extraction

- Canonical rollout actions, embeddings, log-probs, values, intrinsic rewards,
  and aux tensors now stay on-device through stepping, reward shaping, memory
  queries, and buffer append.
- Host extraction remains only for coarse telemetry payloads.

### 2.3 Former CPU Episodic Memory Boundary

- Canonical training now uses tensor-native episodic-memory query/add calls in
  `projects/actor/src/navi_actor/memory/episodic.py`.
- The previous GPU -> CPU -> FAISS hop has been removed from the rollout loop;
  remaining follow-up work is measurement and coordination tuning only.

### 2.4 Per-Actor Python Reward And Transition Loop

- Reward shaping, episodic-memory query/add, and rollout buffer appends are now
  batched on the canonical path.
- The remaining per-actor loop work is limited to coarse telemetry/accounting,
  not the former hot-path transition assembly.

### 2.5 Secondary Rollout/Update Coordination Cost

- The async optimization thread was revisited after the observation/action,
  rollout-buffer, and episodic-memory tensor-native work landed.
- That reassessment showed the background optimizer had become the remaining
  coordination bottleneck, so canonical training now uses inline
  rollout-boundary PPO updates.

## 3. Reference Comparison: Topo-nav

`c:\dev\projects\topo-nav` keeps the entire canonical rollout/update path
tensor-native:

- observations remain CUDA tensors
- actions remain CUDA tensors
- positions and quaternions remain CUDA tensors
- rewards are computed batched on CUDA
- rollout storage is preallocated in VRAM
- telemetry is the only acceptable coarse CPU-side extraction path

Navi must move toward the same GPU-first shape while preserving the sacred
actor-side policy and external `DistanceMatrix` wire contract for diagnostics.

## 4. Implementation Order

### Phase A - Documentation And Policy Alignment

1. Update `AGENTS.md` and `docs/*.md` to codify that the remaining canonical
   throughput gap is internal tensor/dataflow, not mode selection.
2. State explicitly that canonical training internals may use tensor-native
   CUDA representations while external diagnostics continue to use canonical
   `DistanceMatrix` wire objects.

### Phase B - Tensor-Native Runtime Seam (Completed)

1. Extend `projects/environment/src/navi_environment/backends/sdfdag_backend.py`
   with an internal training-facing tensor-native return path.
2. Preserve the existing public object-based environment path for service-mode
   diagnostics and replay tooling.
3. Keep the integration at the actor CLI boundary only.

### Phase C - Tensor-Native Trainer Input Path (Completed)

1. Update `projects/actor/src/navi_actor/cli.py` to wire the canonical trainer
   to the tensor-native runtime seam.
2. Remove the `_obs_to_tensor()`-driven hot-path rebuild from
   `projects/actor/src/navi_actor/training/ppo_trainer.py`.
3. Keep occasional observation materialization only for dashboard and coarse
   telemetry.

### Phase D - Tensor-Native Action, Reward, And Buffer Path (Completed)

1. Remove per-step CPU action packing from the canonical rollout loop.
2. Add batched reward-shaping execution in
   `projects/actor/src/navi_actor/reward_shaping.py`.
3. Rework rollout storage in
   `projects/actor/src/navi_actor/rollout_buffer.py` so canonical training uses
   indexed GPU writes instead of Python transition assembly in the hot path.

### Phase E - Episodic Memory And Optimizer Follow-Up

1. Remove the current CPU episodic-memory query/add boundary.
2. Reassess whether the async optimizer thread remains beneficial once rollout
  data flow is tensor-native.

Status:
Completed. Canonical training now keeps episodic memory tensor-native and runs
PPO updates inline at rollout boundaries because the old async optimizer path became
the dominant stall source once transport and host-staging overheads were
removed.

### Phase G - 400 SPS Gap Attribution (In Progress)

1. Add explicit diagnostic controls to the canonical trainer so telemetry,
  episodic-memory query/add, and reward shaping can be disabled one class at a
  time without introducing a second training surface.
2. Extend actor perf reporting so host extraction and telemetry publication are
  visible beside `fwd`, `env`, `mem`, and `trans`.
3. Remove any remaining avoidable device -> host -> device bounces still present
  inside the per-tick actor loop.
4. Use the resulting attribution matrix to decide the next implementation pass
  rather than guessing at environment-side micro-optimisations.

Status:
In progress. The next implementation pass starts by instrumenting and isolating
remaining actor-side costs on `navi-actor train` itself.

### Phase H - PPO Boundary And Perf-Only Robustness (In Progress)

1. Remove redundant `.to(device)` minibatch copies and related allocator churn
  from `projects/actor/src/navi_actor/learner_ppo.py` now that rollout storage
  is already CUDA-resident on the canonical path.
2. Improve optimizer-side execution on the existing learner surface rather than
  adding a second trainer mode or changing canonical training defaults first.
3. Harden perf telemetry publication in
  `projects/actor/src/navi_actor/training/ppo_trainer.py` so the reproduced
  `no_obs__no_train_tele` attribution case fails safely or runs cleanly rather
  than exiting at process level.
4. Re-run focused canonical validation and a direct smoke benchmark after the
  learner/telemetry changes to confirm the next attribution delta.

Status:
Ready for implementation. The completed 32-run matrix shows memory and reward
shaping matter for rollout SPS, but PPO update wall time is still the dominant
remaining end-to-end stall.

### Phase F - Validation (Completed)

1. Benchmark after each phase on the same `.gmdag` asset.
2. Track `sps`, `fwd`, `env`, `mem`, `trans`, and `ppo_update_ms` after every
   change.
3. Run focused `pytest`, `ruff`, and `mypy --strict` coverage for touched
   actor/environment/auditor slices.

Status:
Completed. Latest focused validation passed with `39 passed` across the touched
actor slice, `ruff` clean on edited files, `mypy --strict` clean on the edited
trainer, and a real canonical `sdfdag` smoke run on
`artifacts/gmdag/sample_apartment.gmdag` completed successfully with inline PPO
updates and tensor-native initial reset seeding.

## 5. Success Criteria

- Canonical training remains a single public surface.
- Canonical training no longer rebuilds observations through CPU objects inside
  the rollout hot path.
- Canonical training no longer performs repeated `.cpu()`, `.numpy()`, or
  Python object packing inside the main rollout loop except for coarse
  diagnostics.
- Dashboard and telemetry remain functional as non-hot-path observability.
- Throughput improves materially on the current MX150-based machine.
- The repo can explain, with direct measurements, how much SPS is lost to host
  extraction, telemetry emission, episodic memory, reward shaping, and PPO
  rollout-boundary updates.

Status:
Tensor-native migration criteria are met for the canonical `sdfdag` training
surface. Attribution criteria are in progress.

## 6. Hardware Expectation Rule

This PC should be pushed to its maximum stable throughput, but repo planning
must not promise `400 SPS` or `10K SPS` on MX150-class hardware. The software
direction should still target the GPU-first architecture proven by Topo-nav so
that stronger hardware can scale much further.