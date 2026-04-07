# Inference

In-process canonical inference evaluates a trained policy checkpoint on the SDF/DAG
environment runtime. The architecture mirrors the unified training path (direct
`SdfDagBackend`, tensor-native throughout, async ZMQ telemetry worker) but without
PPO, rollout buffers, reward shaping, RND, or episodic memory.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              InferenceRunner (CUDA)              │
│                                                  │
│  SdfDagBackend ──► CognitiveMambaPolicy ──► Env │
│  (batched rays)    (eval, no_grad)      (step)  │
│                                                  │
│  ZMQ PUB telemetry ──► Dashboard (passive)       │
└─────────────────────────────────────────────────┘
```

- **Single process**: environment + policy in one CUDA process (like training).
- **Pure evaluation**: `torch.no_grad()`, policy in `.eval()` mode.
- **Deterministic mode**: optional `--deterministic` uses `policy.heads.mean(features)`
  instead of sampling from the Gaussian distribution.
- **Dashboard**: passive observer via ZMQ PUB/SUB (enabled by default in ghost stack).

## Telemetry Events

| Event | Description | Payload |
|-------|-------------|---------|
| `actor.inference.features` | Spherical features (triggers INFERENCE mode in dashboard) | 17+ float32 |
| `actor.step_result` | Per-step reward/done for dashboard charts | `[reward, episode_return, done, truncated]` |
| `actor.action_published` | Action vector for forward/yaw charts | `[fwd, lateral, vertical, yaw]` |
| `actor.inference.perf` | Performance metrics | `[sps, fwd_ms, step_ms, tick_ms, n_actors]` |
| `actor.inference.episode` | Episode completion | `[episode_return, episode_length]` |
| `distance_matrix_v2` | Live observation for dashboard rendering | Serialized DistanceMatrix |

## CLI

```bash
# Direct CLI
uv run --project projects/actor python -m navi_actor.cli infer \
    --checkpoint path/to/model.pt \
    --actors 4 \
    --deterministic \
    --total-steps 10000

# Entry point shortcut
uv run --project projects/actor inference --checkpoint path/to/model.pt

# Full parameter list
uv run --project projects/actor python -m navi_actor.cli infer --help
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Path to trained model checkpoint |
| `--actors` | 4 | Number of parallel actors |
| `--deterministic` | false | Use action mean instead of sampling |
| `--total-steps` | 0 (unlimited) | Stop after N environment steps |
| `--total-episodes` | 0 (unlimited) | Stop after N completed episodes |
| `--temporal-core` | mamba2 | Temporal core backend |
| `--log-every` | 100 | Log summary every N steps |
| `--emit-observation-stream / --no-emit-observation-stream` | true | Dashboard observation publishing |
| `--dashboard-observation-hz` | 10 | Dashboard frame rate |

Scene/corpus selection uses the same parameters as training:
`--scene`, `--manifest`, `--corpus-root`, `--gmdag-root`, `--gmdag-file`,
`--datasets`, `--exclude-datasets`.

## Scripts

### Ghost Stack

```powershell
# In-process inference with dashboard (default)
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt"

# Deterministic, specific dataset, no dashboard
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -Deterministic -Datasets "ai-habitat_ReplicaCAD_baked_lighting" -NoDashboard

# Bounded inference
.\scripts\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -TotalSteps 10000
```

### Standalone Wrapper

```powershell
# Direct inference wrapper
.\scripts\run-inference.ps1 -Checkpoint ".\my_model.pt"

# With all options
.\scripts\run-inference.ps1 -Checkpoint ".\my_model.pt" -Deterministic -Actors 2 -TotalSteps 5000
```

## Dashboard Integration

The dashboard automatically detects inference mode when it receives
`actor.inference.features` telemetry events. No configuration is needed.

- **SPS display**: fed by `actor.inference.perf` events
- **Reward chart**: fed by `actor.step_result` events
- **Observation view**: actor 0 half-sphere from `distance_matrix_v2`
- **Action charts**: forward/yaw from `actor.action_published`
- **Mode indicator**: shows INFERENCE when features are received

## Checkpoint Compatibility

The inference runner accepts both:
1. **Raw state dict**: `torch.save(policy.state_dict(), path)`
2. **Training snapshot**: files with `policy_state_dict` key (produced by `PpoTrainer`)

The runner automatically detects the format and loads appropriately.
