# PLAN.md - Architecture Merge And Implementation Roadmap

## 1. Purpose

This plan now serves two jobs:

- preserve the architecture decisions extracted from `imported/`
- convert those decisions into an implementation roadmap on the current codebase

`imported/` stays in the repository for reference, but planning now happens
against the in-repo documentation set.

## 2. Current Architecture Decision Summary

### 2.1 Accepted Directions

Accepted into current docs and planning:

- strict CUDA and tensor-boundary contracts
- explicit separation between wire contracts and tensor-native hot-path seams
- offline data normalization and corpus preparation outside the PPO hot loop
- passive dashboard and dataset-auditor direction
- stronger layered verification language

### 2.2 Rejected As Canonical

Rejected as production architecture:

- distributed ZMQ lock-step training as the main training surface
- unverified headline `>10,000 SPS` documentation claims

### 2.3 Benchmark-Gated Only

Kept as candidates, not settled decisions:

- truncation metadata
- alternate leaf-distance storage policies
- Morton or other layout redesigns
- stream-overlap or double-buffering changes that complicate the single canonical trainer

## 3. Imported-To-Navi Mapping

| Imported file | Navi home |
| --- | --- |
| `ARCHITECTURE_OVERVIEW.md` | `docs/ARCHITECTURE.md` |
| `SYSTEM_COMPONENTS.md` | `docs/ARCHITECTURE.md`, `docs/COMPILER.md`, `docs/AUDITOR.md` |
| `DATA_FLOW_AND_SEQUENCE.md` | `docs/DATAFLOW.md`, `docs/PERFORMANCE.md` |
| `MEMORY_LAYOUT_AND_DAG_SPEC.md` | `docs/SDFDAG_RUNTIME.md`, `docs/COMPILER.md` |
| `TENSOR_AND_API_CONTRACTS.md` | `docs/CONTRACTS.md`, `docs/SDFDAG_RUNTIME.md` |
| `TESTING_AND_VERIFICATION.md` | `docs/VERIFICATION.md` |
| `VOXEL_COMPILER.md` | `docs/COMPILER.md` |
| `OMNISENSE_ARCHITECTURE.md` | `docs/ARCHITECTURE.md`, `docs/SIMULATION.md` |
| `DASHBOARD_DESIGN.md` | `docs/AUDITOR.md`, `docs/DATAFLOW.md` |

## 4. Current Best-Of-Both-Worlds Direction

The strongest merged architecture is:

1. keep the current in-process canonical trainer
2. document and enforce strict low-level CUDA and tensor contracts
3. preserve service-mode ZMQ and passive auditing without letting them back into the hot path
4. keep storage-layout redesign benchmark-gated until actor dataflow and PPO bottlenecks are addressed
5. prioritize actor-side tensor flow and PPO update cost over more environment micro-optimization

## 5. Current Bottleneck Thesis

The current codebase and docs now support the following planning thesis:

- the environment path is already sufficiently accelerated that it is no longer
	the only dominant problem
- the main remaining architecture-significant throughput problems are in actor
	dataflow, selective materialization, host extraction, and PPO update flow
- planning should therefore start with the trainer hot path, not with a second
	round of speculative runtime-format redesign

## 6. Roadmap Structure

This roadmap is organized into four phases so planning can advance without
losing the imported architecture context.

### Phase 0 - Documentation And Decision Lock

Status: complete for this pass

Completed outcomes:

- imported architecture ideas mapped into in-repo docs
- canonical-vs-experimental boundaries documented
- dedicated docs added for compiler, dataflow, and auditor architecture
- actor, training, and contract docs upgraded to the same style level

Exit criterion:

- implementation work can proceed without needing `imported/` as the sole source of architecture detail

### Phase 1 - Canonical Hot-Path Cleanup

Status: active implementation priority

Objective:

- remove avoidable Python and host-side staging from the canonical training loop

Work packages:

1. selective observation materialization only for actors that must be published
2. reduced action materialization on the direct trainer path
3. batched host extraction only where unavoidable
4. tighter perf accounting around `host_extract_ms` and `telemetry_publish_ms`

Success criteria:

- fewer unconditional `DistanceMatrix` and `Action` object builds in the rollout loop
- lower host extraction time without regressions in passive observability
- no regression in contract-correct dashboard publication

### Phase 2 - PPO Update Optimization

Status: queued immediately after or partially overlapping Phase 1

Objective:

- reduce wall-clock PPO update cost on the canonical learner path

Work packages:

1. verify rollout tensor caching and reuse across PPO epochs
2. reduce optimizer-side device copies and allocator churn
3. keep minibatch shuffle and indexing fully on-device where practical
4. validate perf-only telemetry and ablation combinations remain safe

Current Phase 2 progress:

- [x] remove Python-per-sequence hidden-state reconstruction from `MultiTrajectoryBuffer.sample_minibatches(...)` on the canonical BPTT path
- [x] cache sequence-start hidden states and aux sequence views once per PPO update so minibatch sampling reuses tensor-native indexing instead of iterating CUDA permutation indices in Python
- [x] cache per-update sequence reshapes inside `MultiTrajectoryBuffer` so repeated PPO epochs reuse BPTT tensor views instead of rebuilding them every sampling pass
- [x] extend `TrajectoryBuffer.MiniBatch` with tensor-native `hidden_batch` and teach `PpoLearner.train_ppo_epoch(...)` to consume it before falling back to legacy `hidden_states`
- [x] accumulate PPO minibatch metrics on-device inside `PpoLearner.train_ppo_epoch(...)` so loss, KL, clip fraction, and RND accounting no longer call `.item()` on every minibatch
- [x] pass cached sequence observations/actions/aux tensors through `TrajectoryBuffer.MiniBatch` so `PpoLearner.train_ppo_epoch(...)` can call `policy.evaluate_sequence(...)` without rebuilding BPTT minibatches from flattened tensors
- [x] validate the new sequence hidden-batch path with focused actor checks: `22 passed`, focused `ruff`, focused `mypy --strict`

Success criteria:

- reduced `ppo_update_ms`
- stable or improved training correctness metrics
- no reintroduction of alternate trainer modes

### Phase 3 - Verification And Stability Deepening

Status: planning started in this pass

Objective:

- strengthen proof around the current canonical path before deeper runtime redesign

Work packages:

1. long-run allocation-stability tests around repeated CUDA stepping
2. explicit dtype and contiguity tests at the extension boundary
3. deeper `.gmdag` artifact invariant tests tied to the actual in-repo format
4. optional passive dataset-auditor validation if that surface is implemented

Success criteria:

- stronger protection against silent CUDA-boundary regressions
- stronger confidence in long-running training stability
- clearer separation between runtime faults and trainer faults

### Phase 4 - Benchmark-Gated Runtime Experiments

Status: deferred until Phases 1 through 3 produce clear results

Objective:

- evaluate whether lower-level runtime redesign is still warranted after hot-path cleanup

Candidate experiments:

1. truncation-aware storage or metadata
2. alternate payload precision policies
3. layout redesign such as Morton-style ordering
4. measured stream-overlap experiments that do not create a second production runtime

Success criteria:

- real trainer win, not just microbenchmark win
- preserved actor contract
- no widening of the production architecture surface

## 7. Immediate Planning Output

To make planning progress concrete, the next implementation planning unit is:

### Sprint A - Trainer Materialization Cleanup

Scope:

- canonical trainer only
- no format redesign
- no alternate runtime modes

Primary tasks:

1. audit every remaining unconditional `DistanceMatrix` materialization on the trainer path
2. audit every remaining per-actor `Action` materialization on the trainer path
3. classify each occurrence as:
	 - required for passive publication
	 - required for telemetry
	 - removable
	 - replaceable with batched tensor handling
4. implement the removable cases first

Current audited checklist:

- [x] confirm tensor-native stepping is already the preferred trainer path
- [x] confirm selected-actor publish-time `DistanceMatrix` materialization is already selective
- [x] identify fallback Python `Action` construction in `ppo_trainer.py`
- [x] identify fallback observation rebuild via `_obs_to_tensor(...)` in `ppo_trainer.py`
- [x] identify per-step host extraction for loop similarity and reward accounting in `ppo_trainer.py`
- [x] identify unconditional tensor-path `depth_cpu` extraction in `SdfDagBackend._consume_actor_observation()`
- [x] identify tensor-path MJX speed-throttling seam still reading `actor.prev_depth` through NumPy
- [x] remove canonical-trainer fallback Python `Action` construction in `ppo_trainer.py`
- [x] remove canonical-trainer fallback observation rebuild via `_obs_to_tensor(...)`
- [x] require tensor-native `reset_tensor()` and `batch_step_tensor_actions()` on the canonical PPO runtime seam
- [x] remove the tensor-path MJX speed-throttling host round-trip so previous depth can stay tensor-native until publish-time materialization is requested
- [x] re-measure `host_extract_ms`, `telemetry_publish_ms`, and `sps` after the first seam cleanup
- [x] batch intrinsic-reward and loop-similarity host extraction behind telemetry-only need in `ppo_trainer.py`
- [x] re-measure the canonical trainer after the host-extraction reduction pass
- [x] move canonical reward/done/truncated rollout tensors onto the `SdfDagTensorStepBatch` seam so the trainer stops rebuilding them from `StepResult` objects each tick
- [x] re-measure rollout-side `sps`, `env_ms`, `trans_ms`, and `host_extract_ms` after the tensor-result seam change

Measurement plan:

- compare before and after trainer runs on the same profile
- track `sps`, `host_extract_ms`, `telemetry_publish_ms`, and `ppo_update_ms`
- confirm dashboard and telemetry still function for the selected published actor set

Current post-cleanup measurement:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048.log`
- normalized summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048.log.summary.json`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, default telemetry-on profile
- current mean metrics: `sps=126.15`, `host_extract_ms=0.7125`, `telemetry_publish_ms=0.0`, `ppo_update_ms=19596.59`
- summary source: `ppo_update_source=artifact-log`, `final_checkpoint_path=checkpoints\policy_final.pt`
- previous comparable attribution baseline from `artifacts/benchmarks/attribution-matrix/20260309-095734/summary.csv` row `baseline`: `sps=120.3375`, `host_ms=0.75`, `tele_ms=0.0`, `ppo_update_ms=20967.29`
- initial interpretation: Sprint A seam cleanup improved steady-state SPS by about `5.8`, slightly reduced host extraction cost, and reduced mean PPO update wall time on this bounded profile

Current post-host-extraction measurement:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_hostcut.log`
- normalized summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_hostcut.log.summary.json`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, default telemetry-on profile
- current mean metrics: `sps=131.19`, `host_extract_ms=0.3625`, `telemetry_publish_ms=0.0`, `ppo_update_ms=19030.26`
- summary source: `ppo_update_source=artifact-log`, `final_checkpoint_path=checkpoints\policy_final.pt`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048.log`: `sps=+5.04`, `host_extract_ms=-0.35`, `telemetry_publish_ms=+0.0`, `ppo_update_ms=-566.33`
- delta versus `artifacts/benchmarks/attribution-matrix/20260309-095734/summary.csv` row `baseline`: `sps=+10.85`, `host_ms=-0.3875`, `tele_ms=+0.0`, `ppo_update_ms=-1937.03`
- interpretation: the host-extraction reduction produced another measurable Sprint A win, roughly halving mean host extraction cost while improving bounded-profile SPS and slightly reducing mean inline PPO update wall time

Current post-result-tensor rollout measurement:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, default telemetry-on profile
- recovered summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log.summary.json`
- mean metrics from `8` logged step samples: `sps=132.25`, `env_ms=9.9625`, `mem_ms=1.0375`, `trans_ms=3.725`, `host_extract_ms=0.4375`, `telemetry_publish_ms=0.0`, `ppo_update_ms=23649.38`
- summary source: `ppo_update_source=artifact-log`, `final_checkpoint_path=checkpoints\policy_final.pt`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_hostcut.log`: `sps=+1.06`, `env_ms=-0.65`, `mem_ms=+0.11`, `trans_ms=+0.40`, `host_extract_ms=+0.075`, `telemetry_publish_ms=+0.0`, `ppo_update_ms=+4619.12`
- note: bounded-run summaries should use the artifact log when it contains the optimizer tail and fall back to `logs/navi_actor_train.log` only when redirected capture omits the final completion window; the fallback path was validated with a deliberately truncated temporary log
- interpretation: moving reward/done/truncated tensors onto the canonical step batch improved bounded rollout SPS again and kept host extraction low, while bounded `ppo_update_ms` remains noisy enough that optimizer-wall-time claims still need a larger repeat set than this single recovered run

Current post-Phase-2 PPO update measurement:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_rerun.log`
- normalized summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_rerun.log.summary.json`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, `apartment_1.gmdag`, fresh actor PUB port `tcp://*:5742`
- mean metrics from `8` logged step samples: `sps=125.7875`, `env_ms=11.9125`, `mem_ms=1.075`, `trans_ms=3.8625`, `host_extract_ms=0.4`, `telemetry_publish_ms=0.0`, `ppo_update_ms=22105.53`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log.summary.json`: `sps=-6.4625`, `env_ms=+1.95`, `mem_ms=+0.0375`, `trans_ms=+0.1375`, `host_extract_ms=-0.0375`, `telemetry_publish_ms=+0.0`, `ppo_update_ms=-1543.85`
- interpretation: the first rerun suggested a real optimizer-side win but was too noisy to trust on its own; the repeat-set aggregate below is the stronger signal

Current post-Phase-2 PPO update repeat set:

- aggregate artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat_set.summary.json`
- member summaries:
	- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat1_rerun.log.summary.json`
	- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat2_rerun.log.summary.json`
	- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat3_rerun.log.summary.json`
- repeat-set mean metrics across `3` bounded runs: `sps=123.4625`, `env_ms=11.4833`, `mem_ms=0.95`, `trans_ms=3.9667`, `host_extract_ms=0.3833`, `telemetry_publish_ms=0.0667`, `ppo_update_ms=20456.99`
- repeat-set spread: `sps` ranged from `113.175` to `129.1125`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log.summary.json`: `sps=-8.7875`, `env_ms=+1.5208`, `mem_ms=-0.0875`, `trans_ms=+0.2417`, `host_extract_ms=-0.0542`, `telemetry_publish_ms=+0.0667`, `ppo_update_ms=-3192.39`
- interpretation: the hidden-batch plus sequence-view PPO cleanup now shows a stronger average optimizer-wall-time reduction of about `3.19s` on this bounded profile, but it does not improve end-to-end rollout throughput on the same run set; the next Phase 2 pass should target learner-side overhead that can preserve the PPO win without giving back SPS

Current post-Phase-2 metric-accumulation rerun:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_metricaccum_rerun2.log`
- normalized summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_metricaccum_rerun2.log.summary.json`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, `apartment_1.gmdag`, fresh actor PUB port `tcp://*:5748`
- mean metrics from `20` logged step samples: `sps=118.815`, `env_ms=11.155`, `mem_ms=1.105`, `trans_ms=4.42`, `host_extract_ms=0.315`, `telemetry_publish_ms=0.165`, `ppo_update_ms=19746.43`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log.summary.json`: `sps=-13.435`, `env_ms=+1.1925`, `mem_ms=+0.0675`, `trans_ms=+0.695`, `host_extract_ms=-0.1225`, `telemetry_publish_ms=+0.165`, `ppo_update_ms=-3902.95`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat_set.summary.json`: `sps=-4.6475`, `env_ms=-0.3283`, `mem_ms=+0.155`, `trans_ms=+0.4533`, `host_extract_ms=-0.0683`, `telemetry_publish_ms=+0.0983`, `ppo_update_ms=-710.56`
- interpretation: moving minibatch metric accounting fully onto device cut another `~0.71s` off `ppo_update_ms` relative to the current Phase 2 repeat-set mean and lowered `host_extract_ms` again, but this single rerun still underperformed on end-to-end SPS; the next learner pass should focus on sequence-branch reshape/remainder churn in `learner_ppo.py` before trusting optimizer wins as throughput wins

Current post-Phase-2 direct-sequence rerun:

- `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_directseq_rerun2.log`
- normalized summary artifact: `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_directseq_rerun2.log.summary.json`
- bounded canonical run: `4` actors, `128x24`, `2048` total steps, `apartment_1.gmdag`, fresh actor PUB port `tcp://*:5750`
- mean metrics from `20` logged step samples: `sps=125.74`, `env_ms=11.665`, `mem_ms=1.025`, `trans_ms=3.76`, `host_extract_ms=0.355`, `telemetry_publish_ms=0.0`, `ppo_update_ms=19141.32`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_result_tensors.log.summary.json`: `sps=-6.51`, `env_ms=+1.7025`, `mem_ms=-0.0125`, `trans_ms=+0.035`, `host_extract_ms=-0.0825`, `telemetry_publish_ms=+0.0`, `ppo_update_ms=-4508.06`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_seqcache_repeat_set.summary.json`: `sps=+2.2775`, `env_ms=+0.1817`, `mem_ms=+0.075`, `trans_ms=-0.2067`, `host_extract_ms=-0.0283`, `telemetry_publish_ms=-0.0667`, `ppo_update_ms=-1315.67`
- delta versus `artifacts/benchmarks/post-sprintA-trainer/canonical_2048_phase2_metricaccum_rerun2.log.summary.json`: `sps=+6.925`, `env_ms=+0.51`, `mem_ms=-0.08`, `trans_ms=-0.66`, `host_extract_ms=+0.04`, `telemetry_publish_ms=-0.165`, `ppo_update_ms=-605.11`
- interpretation: handing cached sequence tensors straight to `policy.evaluate_sequence(...)` preserved the Phase 2 optimizer-wall-time gains, recovered bounded SPS relative to the prior metric-accumulation rerun, and removed the remaining learner-side reshape/remainder churn on the canonical BPTT path; the next learner pass should target any residual remainder fallback and optimizer-side indexing/allocation churn that still sits above the `~19.1s` inline PPO floor on this profile

## 8. Proposed Order Of Execution

The planning order from here should be:

1. Phase 1 / Sprint A: trainer materialization cleanup
2. Phase 2: PPO update optimization
3. Phase 3: runtime-boundary and stability validation expansion
4. Phase 4 only if the previous phases do not sufficiently move throughput

## 9. User Decisions Currently Needed

None are blocking right now.

The architecture-significant contradictions from `imported/` are already
resolved by current repo policy and current docs.

If you want to reopen any of these later, they should be explicit decisions:

- distributed hot-loop stepping vs in-process trainer
- aggressive runtime-format redesign vs trainer-first optimization
- swarm-style coordination vs current single-policy parallel training
