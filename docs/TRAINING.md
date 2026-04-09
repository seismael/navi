# TRAINING.md — Canonical Corpus Training Operations

## 1. Canonical Path

Canonical training runs on one production surface only:

1. Discover the full source-scene corpus under `data/scenes/` unless explicitly narrowed.
2. Prepare or refresh compiled `.gmdag` assets.
3. Launch `navi-actor train` on the in-process `sdfdag` backend.
4. Attach the auditor dashboard only if live inspection is needed.

Default behavior:

- full discovered corpus
- default bootstrap download set: Habitat test scenes plus public ReplicaCAD stages
- continuous training until stopped
- canonical observation contract `256x48`
- staged overwrite-first corpus refresh when explicitly requested
- compiled-corpus reuse by default after refresh, even if raw source downloads were cleaned

Canonical training is valid with no dashboard attached. When the auditor is
attached, it remains a passive consumer of low-rate published observations and
must not alter trainer cadence, environment semantics, or observation math.
Detached wrapper launches now disable the actor observation stream entirely so
no publish-row materialization or update-heartbeat observation callbacks run
unless passive viewing is explicitly requested with `-WithDashboard`.
When the dashboard is attached, it passively displays actor 0 observations
and shows the active actor count plus observation freshness (`Obs=XXms`).
The dashboard uses a `zmq.CONFLATE` observation socket so the displayed frame
is always the latest published observation, preventing stale buffered frames
from replaying after transient UI pauses. No control endpoint or selector
mechanism is required.

Canonical wrappers now create one structured run root per launch. Unless you
override output directories explicitly, the default operational outputs are
written under `artifacts/runs/<run_id>/` with subdirectories for `logs/`,
`metrics/`, `manifests/`, `captures/`, and `checkpoints/`. Stable top-level
logs still exist for quick tailing, but run review should start from the run
root because it carries the shared `run_id` across the whole training flow.

Canonical training measurement is now split into coarse machine-readable streams
under that same run root so bottleneck review does not depend on ad hoc log
scraping alone:

- `metrics/actor_training.command.jsonl` records command-surface phases such as
	corpus preparation, trainer construction, trainer start, bounded training
	execution, and final checkpoint save.
- `metrics/actor_training.jsonl` records trainer lifecycle, heartbeat perf,
	runtime perf, PPO update summaries, checkpoint operations, and final training
	summary records.
- wrapper-driven launches such as `run-ghost-stack.ps1 -Train` also emit
	wrapper-side phase metrics under the active run root so process startup,
	readiness wait, and observed child exit latency can be correlated with the
	actor-side metrics.

Those records intentionally stay coarse. They sample at operation boundaries and
the existing `log_every` cadence instead of every rollout tick, so measurement
remains useful without becoming a new source of training stall.

Measurement control stays on the canonical training surface itself. Use the
existing training command or wrapper plus environment-backed options to decide
which internal stats are active:

- `NAVI_ACTOR_EMIT_INTERNAL_STATS=true|false` controls whether run-scoped
	machine-readable training metrics are written at all.
- `NAVI_ACTOR_ATTACH_RESOURCE_SNAPSHOTS=true|false` controls whether those
	internal metric records include coarse process and CUDA memory snapshots.
- `NAVI_ACTOR_PRINT_PERFORMANCE_SUMMARY=true|false` controls whether the train
	command prints a concise human-readable performance summary and metrics paths.

Equivalent CLI flags exist on `navi-actor train` so one training command can
be used both for normal runs and for deeper bottleneck attribution without
introducing a second launcher surface.

## 2. Prerequisites

- Python `3.12`
- CUDA-capable machine with working PyTorch CUDA runtime
- free port `5557` for actor telemetry

Repository training wrappers launch the actor via `python -m navi_actor.cli`
inside the actor project environment so canonical training does not depend on
console-script resolution state.

Canonical training currently assumes the actor temporal core runs through the
pure-PyTorch Mamba-2 SSD path by default. The canonical train and serve
surfaces expose one explicit temporal-core selector contract on the same
trainer surface: `mamba2` (default), `gru`, and `mambapy`.

The active benchmark machine is an MX150 (`sm_61`, `2 GB` VRAM). That matters
for training guidance:

- `256x48` remains the production default
- `512x96` is a valid bounded comparison surface but carries a much heavier PPO
	update cost on this machine
- `768x144` currently exceeds full-trainer memory during actor-side attention
	even though the environment runtime itself can still be benchmarked at that
	profile

Canonical rollout storage on that surface remains CUDA-resident. Host-staged or
pinned-CPU rollout buffers are not part of the canonical trainer design and
should be treated as non-canonical diagnostics unless the architecture standard
is updated in the same change.

Canonical actor reward shaping also requests `torch.compile` by default when
the shaping path stays tensor-only on the rollout hot path. On unsupported
GPU/compiler stacks, Navi reports that compile was requested but inactive and
continues on the eager tensor path for honest attribution.

Resource snapshots collected on the canonical trainer surface should be read as
coarse attribution aids, not a replacement for deeper profilers. The production
metrics surface captures:

- elapsed wall time for major operations and phases
- process CPU memory and thread counts
- CUDA allocator state such as allocated, reserved, peak allocated, peak
	reserved, and free/total device memory when CUDA is active
- existing trainer and runtime timing summaries such as rollout-step, PPO
	update, and `SdfDagBackend` perf snapshots

That combination is intended to answer the first-order questions quickly:
which phase is slow, whether pressure is mostly CPU-side or CUDA-side, and
whether checkpointing, corpus work, rollout, or PPO update is dominating a run.

## 3. Refresh And Preflight

```powershell
./scripts/refresh-scene-corpus.ps1

uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag --actors 4 --steps 200
```

When compiler packages or compiled-runtime dependencies change, refresh the
entire promoted corpus before using benchmark results to judge the upgrade:

```powershell
./scripts/refresh-scene-corpus.ps1
```

## 4. Canonical Commands

```powershell
# Standard continuous training
./scripts/train.ps1

# Standard continuous training with the alternate Mambapy backend
./scripts/train.ps1 -TemporalCore mambapy

# Disable internal metrics for a lighter run
$env:NAVI_ACTOR_EMIT_INTERNAL_STATS = "false"
./scripts/train.ps1

# Keep internal metrics but omit coarse process/CUDA resource snapshots
$env:NAVI_ACTOR_ATTACH_RESOURCE_SNAPSHOTS = "false"
./scripts/train.ps1

# Standard continuous training on an alternate actor telemetry port
./scripts/train.ps1 -ActorTelemetryPort 5565

# Long-duration wrapper
./scripts/train-all-night.ps1

# Canonical overnight validation wrapper
./scripts/run-nightly-validation.ps1

# Full stack training launcher
./scripts/run-ghost-stack.ps1 -Train

# Full stack training launcher with explicit passive dashboard attach
./scripts/run-ghost-stack.ps1 -Train -WithDashboard

# Full stack training launcher with Mambapy selected explicitly
./scripts/run-ghost-stack.ps1 -Train -TemporalCore mambapy

# Full stack training launcher on an alternate actor telemetry port
./scripts/run-ghost-stack.ps1 -Train -ActorTelemetryPort 5565

# First end-to-end canonical qualification surface
./scripts/qualify-canonical-stack.ps1

# One-command bounded temporal comparison on the canonical trainer surface
./scripts/run-temporal-compare.ps1
./scripts/run-temporal-compare.ps1 -TemporalCores @('mamba2','gru','mambapy')

# One-command bounded observation-resolution comparison on the canonical trainer surface
./scripts/run-resolution-compare.ps1
./scripts/run-resolution-compare.ps1 -Profiles @('256x48','512x96','768x144') -Repeats 2 -TotalSteps 2048
```

The resolution comparison wrapper runs repeated bounded trainer passes on the
same canonical `sdfdag` path and emits per-run logs plus summary JSON/CSV under
`artifacts/benchmarks/resolution-compare/`. Profiles are specified as
`AzimuthxElevation`. Remember that doubling both dimensions is a `4x` ray-count
increase and tripling both dimensions is a `9x` ray-count increase.

Interpret those runs carefully:

- trainer results reflect both the environment and the actor
- larger profiles stress RayViT patch-token attention and PPO update memory,
  not just ray casting
- use `bench-sdfdag` separately when you need environment-only attribution

For durable long-running training with explicit fleet size control, use the
direct actor CLI instead of the wrapper:

```powershell
uv run --project .\projects\actor navi-actor train --actors 4
uv run --project .\projects\actor navi-actor train --actors 8 --total-steps 0
uv run --project .\projects\actor navi-actor train --actors 8 --temporal-core gru --actor-pub tcp://localhost:5565
uv run --project .\projects\auditor navi-auditor dashboard --actor-sub tcp://localhost:5565 --passive
```

That surface is the canonical way to vary `--actors` directly for long runs.
Attach the auditor separately when needed. If you need live observation frames,
launch a surface that keeps observation streaming enabled instead of a detached
wrapper run with `--no-emit-observation-stream`. The dashboard always displays
actor 0 observations with zero selector overhead.

If you want one command that launches both canonical training and the passive
dashboard together on the full discovered corpus, use the wrapper with an
explicit actor count:

```powershell
./scripts/run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard
```

The qualification script now proves the bounded canonical flow end to end:

- dataset preflight
- optional sandboxed corpus refresh into a fresh compiled corpus root
- bounded canonical training
- passive live dashboard attach
- passive training-stream recording
- checkpoint resume from a produced periodic checkpoint
- replay plus passive replay attach

The nightly wrapper builds on those same surfaces and adds:

- focused preflight regression suites
- bounded shared-model checkpoint and resume proof
- repeated environment drift benchmarks
- an overnight soak with checkpoint and passive attach monitoring
- one machine-readable morning summary artifact root

The main training and qualification wrappers also perform explicit cleanup of
stale transient generated outputs before launching. That cleanup is limited to
aged captures, benchmark scratch outputs, temporary directories, and detached
nightly or process-capture leftovers. It must not delete the active run root.

For refresh-to-training qualification without mutating the live promoted corpus:

```powershell
./scripts/qualify-canonical-stack.ps1 -EnableCorpusRefreshQualification -RefreshSourceRoot .\data\scenes -RefreshManifest .\data\scenes\scene_manifest_all.json -RefreshScene .\artifacts\gmdag\corpus\ai-habitat_ReplicaCAD_baked_lighting\Baked_sc0_staging_00.gmdag
```

Explicit narrowing remains available when requested:

```powershell
./scripts/train.ps1 -Scene .\artifacts\gmdag\corpus\ai-habitat_ReplicaCAD_baked_lighting\Baked_sc0_staging_00.gmdag
./scripts/train.ps1 -TotalSteps 500000
./scripts/run-ghost-stack.ps1 -Train -TotalSteps 500000
./scripts/train.ps1 -TemporalCore mambapy -TotalSteps 500000
```

For a bounded like-for-like backend comparison on the canonical trainer surface:

```powershell
./scripts/run-temporal-compare.ps1
./scripts/run-temporal-compare.ps1 -TemporalCores @('mamba2','gru')
./scripts/run-temporal-compare.ps1 -TemporalCores @('mamba2','gru','mambapy') -TotalSteps 256
```

The comparison wrapper runs each backend sequentially on the direct actor
trainer surface with isolated per-repeat logs, checkpoints,
and summary JSON files under one timestamped artifact root. The wrapper now
defaults to the canonical `mamba2` runtime; pass
`-TemporalCores @('mamba2','gru')` for an explicit like-for-like comparison run. The
wrapper records both mean and median aggregate metrics so local Windows
variance does not decide the temporal-core conclusion.
Use `-ProfileCudaEvents` only for diagnostic comparison runs when you need
median learner substage timings such as `backward_ms` or `gpu_backward_ms`.
Compare runs by changing only the temporal-core selector unless a diagnostic
experiment explicitly calls for more changes.

## 5. Checkpoints

- periodic checkpoints use the configured checkpoint directory
- when the default checkpoint root is left unchanged on wrapper-driven runs, checkpoints are written under the current run root `checkpoints/`
- final checkpoints are written as `policy_final.pt` when enabled
- trainer checkpoints now persist the active `run_id` so review tooling can correlate saved state with the originating run
- resume with `-ResumeCheckpoint` on wrappers or `--checkpoint` on the actor CLI

Example:

```powershell
./scripts/train-all-night.ps1 -ResumeCheckpoint .\checkpoints\all_night\policy_step_0025000.pt
```

## 6. Model Lifecycle & Registry

### Checkpoint Format (v3)

All training sources (RL, BC, inference) now emit **v3 checkpoints** with enriched metadata:

| Field | Description |
|-------|-------------|
| `version` | Always `3` |
| `step_id` | Total training steps completed |
| `episode_count` | Total episodes completed |
| `reward_ema` | Exponential moving average of episode reward |
| `wall_time_hours` | Cumulative training wall time |
| `parent_checkpoint` | Path to the checkpoint this was resumed from |
| `training_source` | `"rl"`, `"bc"`, or `"inference"` |
| `temporal_core` | Which temporal core was active (`mamba2`, `gru`, `mambapy`) |
| `corpus_summary` | Description of training data |
| `created_at` | ISO timestamp |

Only v3 checkpoints are accepted. Loading v2 or older checkpoints will fail fast with a clear error.

### Model Registry

Promoted models live in `artifacts/models/` with a JSON catalog:

```
artifacts/models/
  registry.json       # version catalog with metadata
  latest.pt           # always points to the best promoted model
  v001.pt             # versioned copies
  v002.pt
  ...
```

### Promoting a Checkpoint

```bash
# Manually promote a checkpoint
uv run brain promote ./artifacts/runs/<run_id>/checkpoints/policy_step_0050000.pt --notes "50K step RL run" --tags rl,mamba2

# List all promoted models
uv run brain models
```

### Auto-Continue from Latest

When no `--checkpoint` is specified, both the `train` CLI and `run-ghost-stack.ps1 -Train` automatically resume from `artifacts/models/latest.pt` if it exists. This enables seamless accumulation across RL sessions, BC pre-training, and nightly runs.

### Auto-Promote After Training

After training completes, the trainer automatically promotes the final checkpoint to the registry if its `reward_ema` exceeds the current latest. This ensures the best model is always discoverable.

### Evaluate & Compare

```bash
# Evaluate a single checkpoint (bounded inference with metrics)
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000 --json

# Compare two checkpoints side-by-side
uv run brain compare ./artifacts/models/v001.pt ./artifacts/models/v002.pt --steps 2000
```

### Nightly Integration

Successful nightly validation runs (`run-nightly-validation.ps1`) automatically promote their best checkpoint to the registry with a `nightly` tag, so the next training session continues from the nightly's progress.

## 7. Dashboard Attach

`run-ghost-stack.ps1 -Train` now keeps the dashboard detached by default so the
canonical training wrapper does not imply a viewer dependency. Use
`-WithDashboard` only when a live passive observer is explicitly needed.

Detached wrapper launches also pass `--no-emit-observation-stream`, so later
manual dashboard attachment to that same run is telemetry-only. Relaunch with
`-WithDashboard` when live observation frames are required.

```powershell
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

The canonical trainer guarantees actor telemetry on `5557`. Environment sockets may be idle during direct in-process training.

For scripted passive proof without launching the GUI, use:

```powershell
uv run --project .\projects\auditor navi-auditor dashboard-attach-check --actor-sub tcp://localhost:5557 --json
```

If you override the training telemetry port on the wrapper, use the same port
in dashboard attach commands.

## 8. Recovery

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*navi-environment*' -or $_.CommandLine -like '*navi-actor*' -or $_.CommandLine -like '*navi-auditor*') } | ForEach-Object { try { & taskkill /PID $_.ProcessId /T /F *> $null } catch {} }
```

## 9. Operational Notes

- production training advertises only the canonical `sdfdag` path
- production training defaults the temporal core to `mamba2`, with `gru` and `mambapy` available as explicit selectors on the same surface when comparing runs
- wrappers default to the canonical `256x48` observation contract with minibatch `64`, BPTT `8`, and rollout `512`
- corpus compilation defaults to `512`, aligned with the canonical `256x48` environment observation contract
- benchmarks, wrappers, and tests must use real dataset scenes or compiled dataset `.gmdag` assets; generated or sample scenes are not part of the canonical path
- the default bootstrap downloader now pulls the 3 public Habitat test scenes plus 5 public ReplicaCAD stage assets unless `-Datasets` is explicitly narrowed
- `refresh-scene-corpus.ps1` stages downloads and removes transient source assets after a successful corpus promotion
- the promoted live `gmdag_manifest.json` is compiled-only metadata: `source_path` is rewritten to the live `.gmdag` asset path after transient source cleanup
- canonical scene-pool training keeps actors on each scene for multiple completed episodes before rotation; the default budget is `16` completed episodes per scene across the fleet
- collision remains non-terminal in canonical training; collision penalty is
  velocity-scaled (`penalty * (1 + speed)`) so fast crashes are punished more
  severely than gentle grazing, plus positive clearance-delta reward for
  increasing obstacle clearance after near-contact
- progress reward is proximity-discounted: forward displacement is scaled by
  `(1 - proximity_ratio)` so approaching walls yields diminishing credit
- exploration rewards are clearance-gated: multiplied by
  `clamp(clearance, 0, 1)` so exploring into tight spaces yields less credit
- actor-side forward velocity bonus is disabled by default (`velocity_weight=0.0`)
  to remove approach bias near obstacles
- canonical environment shaping also penalizes starvation-heavy views and
  persistent near-field wall-hugging using ratios derived from the current
  spherical observation
- canonical geometry-foraging shaping positively values mid-range structure
  visibility, forward-sector structure reacquisition, and controlled inspection
  turns that reveal more geometry instead of less
- canonical drone max speed defaults to `5.0 m/s` with velocity smoothing `0.15`
  for responsive avoidance; proximity speed limiter engages within `2.0 m` and
  clearance escape incentive extends to `3.0 m`
- use explicit overrides only for deliberate experiments; temporal-core comparisons should change only `TemporalCore` or `--temporal-core` while holding all other canonical settings fixed
- resolution-scaling comparisons should change only `AzimuthxElevation` while holding actor count, temporal core, rollout length, minibatch size, and PPO epochs fixed
- dashboard mode detection stays on low-volume telemetry to avoid rollout stalls
- the actor CLI still defaults the dashboard observation stream to a passive `10 Hz` frame for the selected telemetry actor, but `run-ghost-stack.ps1 -Train` disables that stream when launched detached and re-enables it only with `-WithDashboard`
- widening observation streaming or full training telemetry to all actors remains a deliberate high-overhead diagnostic mode and should be enabled only when the operator explicitly requests all-actor fan-out
- those observation frames reuse the canonical spherical contract; any half-sphere or other display transform belongs in auditor code only
- the current grouped rollout surface may launch actor subsets on per-group CUDA streams, but any future ping-pong rewrite must preserve actor-local `observation -> action -> next observation` ordering and prove its value with bounded canonical measurements
- PPO update-loss scalar materialization is diagnostic-only on the canonical hot path: the trainer still emits coarse `actor.training.ppo.update` events for mode/status detection by default, but full PPO loss fields are populated only when explicit update-loss telemetry is enabled
- when a future temporal-core candidate proves better, re-promotion must be proven by rerunning a controlled head-to-head training comparison and bounded canonical training surfaces before the active docs and scripts are switched away from Mamba2 SSD
- stronger hardware may move the high-resolution ceiling outward, but documentation and scripts must not imply that higher observation profiles are already production-safe on the active MX150 machine
- `rollout_overlap_groups` (`ActorConfig`, env var `NAVI_ACTOR_ROLLOUT_OVERLAP_GROUPS`) controls multi-group pipelined rollout; default is `1` (optimal for MX150's 3 SMs); `2` is available for larger GPUs with enough SMs for concurrent kernel execution
- on `sm_61`, GPU compute utilization during training is limited by eager PyTorch dispatcher overhead (~72-90 kernel launches per rollout tick, ~165-376 per PPO minibatch) and cannot be improved without `torch.compile` (`sm_70+` required), `mamba-ssm` fused Triton kernels (not available on Windows), or a PPO/rollout double-buffer overlap architecture; see `docs/PERFORMANCE.md` §4.0 for the full analysis
- canonical `SdfDagBackend` hot-path `cast_rays()` calls use `skip_direction_validation=True` because yaw-rotated unit vectors are mathematically guaranteed normalized, eliminating four GPU→CPU synchronization barriers per call

---

## Behavioral Cloning Pre-Training

Behavioral cloning (BC) provides a supervised pre-training path that bootstraps
the policy from human navigation demonstrations before RL fine-tuning.

### Overview

The BC pipeline trains the **same** `CognitiveMambaPolicy` architecture that PPO
trains.  No separate model or surrogate is involved.  The result is a standard
v3 checkpoint loadable by `navi-actor train --checkpoint <path>`.

### Demonstration Capture

Demonstrations are recorded during manual dashboard navigation using the auditor
`explore` command:

```powershell
# Launch explorer with automatic recording
uv run --project projects/auditor explore --record --gmdag-file <scene.gmdag>
```

Recording starts automatically when the dashboard opens.  Navigate the scene
with WASD/arrow keys.  Every teleop step captures the current observation and
the normalised action.  Demonstrations are saved as `.npz` archives under
`artifacts/demonstrations/` when the dashboard is closed (ESC/Q) or when the
user presses **B** to pause recording.

Each `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `observations` | `(N, 3, Az, El)` | Stacked depth/semantic/valid channels |
| `actions` | `(N, 4)` | Normalised `[fwd, vert, lat, yaw]` in `[-1, 1]` |
| `format_version` | scalar | Archive format version (currently `1`) |
| `scene` | string | Scene identifier |
| `azimuth_bins` | scalar | Observation azimuth resolution |
| `elevation_bins` | scalar | Observation elevation resolution |

Action normalisation uses the drone kinematic limits:

| Channel | Raw Unit | Normaliser | Default |
|---------|----------|------------|---------|
| forward | m/s | `drone_max_speed` | 5.0 |
| vertical | m/s | `drone_climb_rate` | 2.0 |
| lateral | m/s | `drone_strafe_speed` | 3.0 |
| yaw | rad/s | `drone_yaw_rate` | 3.0 |

### Single-Scene Training

```powershell
# Train from scratch on accumulated demonstrations
uv run --project projects/actor brain bc-pretrain

# Custom hyperparameters
uv run --project projects/actor brain bc-pretrain --epochs 100 --learning-rate 5e-4 --bptt-len 16
```

### Incremental Multi-Scene Training

The `--checkpoint` flag loads an existing model before training, enabling
incremental improvement across scenes:

```powershell
# Scene 1: explore and record
uv run --project projects/auditor explore --record --gmdag-file scene1.gmdag

# Train from scratch
uv run --project projects/actor brain bc-pretrain

# Scene 2: explore and record
uv run --project projects/auditor explore --record --gmdag-file scene2.gmdag

# Update the existing model with new data
uv run --project projects/actor brain bc-pretrain \
    --checkpoint artifacts/checkpoints/bc_base_model.pt
```

Or use the two-step workflow that separates navigation from training:

```powershell
# Step 1: Fly through all scenes (demos accumulate, no training waits)
./scripts/run-explore-scenes.ps1

# Step 2: Train on all accumulated demos
./scripts/run-bc-pretrain.ps1
./scripts/run-bc-pretrain.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt
```

### Training Algorithm

1. Load all `.npz` files from `artifacts/demonstrations/`.
2. Chunk into BPTT sequences of configurable length (default 8).
3. Shuffle sequences and iterate in minibatches.
4. Forward through `evaluate_sequence()` — the same pipeline PPO uses.
5. Optimise negative log-likelihood loss with entropy regularisation.
6. Save a v3 checkpoint with fresh RND module weights.

Policy `log_std` is frozen during BC by default (`--freeze-log-std`) to
preserve exploration capacity for subsequent PPO training.

### Transition to RL Fine-Tuning

The BC checkpoint is a standard v3 file:

```powershell
uv run --project projects/actor navi-actor train \
    --checkpoint artifacts/checkpoints/bc_base_model.pt
```

The RL trainer unfreezes `log_std` and continues optimising the same policy
weights with PPO.  This gives the agent a human-informed starting point
instead of learning from zero.

### Default Paths

| Artifact | Default Location |
|----------|------------------|
| Demonstrations | `artifacts/demonstrations/*.npz` |
| BC checkpoint | `artifacts/checkpoints/bc_base_model.pt` |