# navi-actor

Brain Layer for the Navi ecosystem. Houses the sacred cognitive engine, PPO
trainer, and policy checkpoint management.

**Full specification:** [docs/ACTOR.md](../../docs/ACTOR.md)  
**Training guide:** [docs/TRAINING.md](../../docs/TRAINING.md)  
**Inference guide:** [docs/INFERENCE.md](../../docs/INFERENCE.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Install

```bash
cd projects/actor
uv sync
```

CUDA wheel setup (pins `torch==2.5.1+cu121` for `sm_61+`):

```powershell
./scripts/setup-actor-cuda.ps1
```

---

## Engine Isolation Principle

**The training engine is sacred.** The cognitive pipeline is never modified to
accommodate a new data source. External data always connects through a
`DatasetAdapter` in `environment/backends/` that transforms raw observations
into the canonical `(1, Az, El)` `DistanceMatrix` format.

---

## Cognitive Architecture

`CognitiveMambaPolicy` implements a 5-stage pipeline:

1. **RayViTEncoder** — ViT `(B, 3, Az, El)` → `(B, 128)` spatial embedding
2. **RND Curiosity** — intrinsic exploration reward from embedding novelty
3. **EpisodicMemory** — tensor-native cosine-similarity loop-avoidance
4. **TemporalCore** — sequence engine (`mamba2` default, `gru` and `mambapy` comparison)
5. **ActorCriticHeads** — 4-DOF action distribution + value estimation

Input contract: only `depth` and `semantic` from `DistanceMatrix` are consumed.

---

## Usage

### Inference (Service Mode)

```bash
uv run navi-actor serve \
    --sub tcp://localhost:5559 --pub tcp://*:5557 \
    --mode step --step-endpoint tcp://localhost:5560

# Shortcut
uv run brain

# With a trained checkpoint
uv run navi-actor serve --policy-checkpoint checkpoints/policy_final.pt \
    --mode step --step-endpoint tcp://localhost:5560
```

### Training

```bash
# Canonical continuous training on the full corpus (mamba2 default)
uv run navi-actor train --actors 4

# Continuous (no step limit)
uv run navi-actor train --actors 8 --total-steps 0
```

Repository wrappers:

```powershell
./scripts/train.ps1
./scripts/train.ps1 -TemporalCore gru
./scripts/train-all-night.ps1
./scripts/run-ghost-stack.ps1 -Train -WithDashboard
```

### In-Process Inference

Evaluate a trained checkpoint with direct in-process CUDA stepping (same
backend as training, without PPO or rollout buffers):

```bash
# Full corpus inference with dashboard
uv run navi-actor infer --checkpoint artifacts/checkpoints/bc_base_model.pt

# Deterministic evaluation with step limit
uv run navi-actor infer --checkpoint model.pt --deterministic --total-steps 10000

# Via unified entry point
uv run brain infer --checkpoint model.pt --actors 4
```

Repository wrappers:

```powershell
./scripts/run-ghost-stack.ps1 -Infer -Checkpoint .\model.pt
./scripts/run-inference.ps1 -Checkpoint .\model.pt -Deterministic
```

### Behavioral Cloning Pre-Training

```bash
# Train from accumulated human demonstrations
uv run brain bc-pretrain

# Resume from existing checkpoint (incremental improvement)
uv run brain bc-pretrain --checkpoint artifacts/checkpoints/bc_base_model.pt

# Custom hyperparameters
uv run brain bc-pretrain --epochs 100 --learning-rate 5e-4 --bptt-len 16
```

Demonstrations are recorded in the auditor project:

```bash
# Navigate scenes continuously (demos accumulate, no training wait)
./scripts/run-explore-scenes.ps1

# Record a single scene
uv run --project projects/auditor explore --record --gmdag-file <scene.gmdag>
```

Train on accumulated demonstrations:

```powershell
./scripts/run-bc-pretrain.ps1
./scripts/run-bc-pretrain.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt
```

The BC checkpoint is a standard v3 file loadable by `navi-actor train --checkpoint <path>`
for RL fine-tuning.

### Model Management

```bash
# Promote a checkpoint to the model registry
uv run brain promote ./checkpoints/policy_step_0050000.pt --notes "50K RL run" --tags rl,mamba2

# List all promoted models
uv run brain models

# Evaluate a checkpoint (bounded inference with quality metrics)
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000

# Compare two checkpoints side-by-side
uv run brain compare ./artifacts/models/v001.pt ./artifacts/models/v002.pt --steps 2000

# Evaluate with JSON output for scripting
uv run brain evaluate ./artifacts/models/latest.pt --steps 2000 --json
```

Training auto-continues from `artifacts/models/latest.pt` when no `--checkpoint` is specified.
After training, the final checkpoint is auto-promoted if its `reward_ema` exceeds the current latest.

### Windows Wrapper

```powershell
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 \
    --mode step --step-endpoint tcp://localhost:5560
```

---

## Temporal Core

The canonical default is **pure-PyTorch Mamba-2 SSD**, proven by a 25K-step
training comparison to deliver significantly better learning quality
(reward_ema −0.88 vs GRU's −1.48) with a modest throughput trade-off
(~72 SPS vs ~100 SPS).

Available backends on the same trainer surface:

| Backend | Status |
|---------|--------|
| `mamba2` | **Default** — pure-PyTorch SSD, no build step |
| `gru` | Comparison — cuDNN fused, highest throughput |
| `mambapy` | Comparison — reference Mamba implementation |

Future hardware-fused `mamba-ssm` remains deferred until a supported
environment exists and end-to-end training proves the upgrade.

### Temporal Benchmarking

```powershell
# Microbenchmark (writes JSON artifact)
./scripts/run-temporal-bakeoff.ps1

# Bounded end-to-end training comparison
./scripts/run-temporal-compare.ps1
./scripts/run-temporal-compare.ps1 -TemporalCores @('gru','mambapy')
```

---

## Checkpoint Format (v3)

| Key | Description |
|-----|-------------|
| `version` | Format version (3) |
| `step_id` | Total training steps completed |
| `episode_count` | Total episodes completed |
| `reward_ema` | Exponential moving average of episode reward |
| `wall_time_hours` | Cumulative training wall time |
| `parent_checkpoint` | Path to the checkpoint this was resumed from |
| `training_source` | `"rl"`, `"bc"`, or `"inference"` |
| `temporal_core` | Active temporal core (`mamba2`, `gru`, `mambapy`) |
| `corpus_summary` | Description of training data |
| `created_at` | ISO timestamp |
| `policy_state_dict` | All policy network weights |
| `rnd_state_dict` | RND predictor + running statistics |
| `optimizer_state_dict` | Adam optimizer state (if available) |
| `rnd_optimizer_state_dict` | RND optimizer state (if available) |
| `reward_shaper_step` | Reward shaper annealing step counter |

Only v3 checkpoints are accepted. The `--checkpoint` flag resumes training
from any previous checkpoint, preserving all learned knowledge across sessions.

---

## Resolution Scaling

Token count scales as `(Az / 8) × (El / 8)` with `patch_size=8`:

| Profile | Tokens | Notes |
|---------|--------|-------|
| `256×48` | 192 | Canonical production default |
| `384×72` | 432 | Viable, moderate PPO cost increase |
| `512×96` | 768 | Viable, significant PPO cost |
| `768×144` | 1728 | OOM during self-attention on MX150 |

See [docs/RESOLUTION_BENCHMARKS.md](../../docs/RESOLUTION_BENCHMARKS.md).

---

## Default Port

| Port | Topics |
|------|--------|
| `5557` | `action_v2`, `telemetry_event_v2` (PUB) |

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
