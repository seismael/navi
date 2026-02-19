# navi-actor

Brain Layer for Ghost-Matrix runtime.

## Responsibilities

- Subscribes to `distance_matrix_v2`.
- Produces `action_v2`.
- Uses step-mode REQ/REP with Simulation Layer when enabled.

## Usage

```bash
cd projects/actor
uv sync
uv run navi-actor run --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560
```

## Online Training (full spherical view)

Run the trainer against a step-mode Section Manager stream:

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
