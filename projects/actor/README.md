# navi-actor

Brain Layer for Ghost-Matrix runtime.

**Full specification:** [docs/ACTOR.md](../../docs/ACTOR.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Engine Isolation Principle

**The training engine is sacred.** The cognitive pipeline
(RayViTEncoder → Mamba2 → EpisodicMemory → ActorCriticHeads → PPO) is never
modified to accommodate a new data source. External data always connects through
a `DatasetAdapter` in `environment/backends/` that transforms raw
observations *to* the engine's canonical `(1, Az, El)` DistanceMatrix format.

## Cognitive Architecture

`CognitiveMambaPolicy` implements a 5-stage pipeline:
1. **RayViTEncoder** — ViT `(B, 3, Az, El)` → `(B, 128)` spatial embedding
2. **RND Curiosity** — intrinsic exploration reward from embedding novelty
3. **EpisodicMemory** — FAISS KNN loop-closure detection and context retrieval
4. **Mamba2TemporalCore** — O(n) selective state-space temporal integration
5. **ActorCriticHeads** — 4-DOF action distribution + value estimation

Input contract: only `depth` and `semantic` from `DistanceMatrix` are consumed.
No RGB frames, camera images, or non-canonical fields enter this engine.

## Responsibilities

- Subscribes to `distance_matrix_v2`.
- Produces `action_v2` (4-DOF: linear xyz + yaw rate).
- Uses step-mode REQ/REP with Simulation Layer when enabled.
- Publishes `telemetry_event_v2` with 13 PPO training metrics.

## Usage

```bash
cd projects/actor
uv sync
uv run navi-actor run --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560
```

## Online Training (full spherical view)

Run the trainer against a step-mode Environment stream:

```bash
cd projects/actor
uv run navi-actor train --sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560 --steps 300
```

Save a learned checkpoint for runtime use:

```bash
uv run navi-actor train --sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560 --steps 2000 --save-checkpoint artifacts/policy_spherical.npz
```

Long-run training with periodic checkpoints and evaluation episodes:

```bash
uv run navi-actor train --sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560 --steps 10000 --checkpoint-every 1000 --checkpoint-dir artifacts/checkpoints --checkpoint-prefix policy_spherical --eval-every 500 --eval-episodes 3 --eval-horizon 120
```

Save evaluation progress as CSV + plot image:

```bash
uv run navi-actor train --sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560 --steps 10000 --eval-every 500 --eval-episodes 3 --eval-horizon 120 --eval-csv artifacts/eval_progress.csv --eval-plot artifacts/eval_progress.png
```

Evaluation outputs include exploration metrics:
- `eval_novelty_rate`: fraction of evaluation steps entering new cells.
- `eval_coverage_mean`: average unique-cells-per-step coverage per episode.

Conservative low-collision profile:

```bash
uv run navi-actor train --sub tcp://localhost:5559 --step-endpoint tcp://localhost:5560 --steps 200 --max-forward 0.25 --sigma-forward 0.03
```

Run Actor with learned policy checkpoint:

```bash
uv run navi-actor run --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560 --policy learned --policy-checkpoint artifacts/policy_spherical.npz
```

## Checkpoint Format (v2)

Knowledge accumulation checkpoints store the full training state:

| Key | Description |
|-----|-------------|
| `version` | Format version (2) |
| `policy_state_dict` | All policy network weights |
| `rnd_state_dict` | RND predictor + running statistics |
| `optimizer_state` | Adam optimizer state |
| `rnd_optimizer_state` | RND optimizer state |
| `reward_shaper_step` | Reward shaper annealing step counter |

Checkpoints are saved as compressed `.npz` files. The `--checkpoint` flag
resumes training from any previous checkpoint, preserving all learned knowledge
across scenes.

## Runtime Ports

- `5557` PUB: `action_v2`, `telemetry_event_v2`

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
