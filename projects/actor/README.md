# navi-actor

Brain Layer for Ghost-Matrix runtime.

**Full specification:** [docs/ACTOR.md](../../docs/ACTOR.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Engine Isolation Principle

**The training engine is sacred.** The cognitive pipeline
(RayViTEncoder → TemporalCore → EpisodicMemory → ActorCriticHeads → PPO) is never
modified to accommodate a new data source. External data always connects through
a `DatasetAdapter` in `environment/backends/` that transforms raw
observations *to* the engine's canonical `(1, Az, El)` DistanceMatrix format.

## Cognitive Architecture

`CognitiveMambaPolicy` implements a 5-stage pipeline:
1. **RayViTEncoder** — ViT `(B, 3, Az, El)` → `(B, 128)` spatial embedding
2. **RND Curiosity** — intrinsic exploration reward from embedding novelty
3. **EpisodicMemory** — FAISS KNN loop-closure detection and context retrieval
4. **TemporalCore** — canonical sequence engine (benchmark-selected policy)
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
uv run navi-actor serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Shortcut command (equivalent to: navi-actor serve)
uv run brain
```

## Canonical Training

Run the only production training surface directly on the compiled sdfdag runtime:

```bash
cd projects/actor
uv run navi-actor train --gmdag-file ../../artifacts/gmdag/sample_apartment.gmdag --actors 4 --total-steps 100000 --checkpoint-every 25000 --checkpoint-dir ../../checkpoints
```

Repository wrappers keep the same canonical path:

```powershell
./scripts/train.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag
./scripts/train-all-night.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag
```

Run Actor with learned policy checkpoint:

```bash
uv run navi-actor serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560 --policy-checkpoint checkpoints/policy_final.pt
```

## Windows Wrapper Script

```powershell
# From repository root
./scripts/run-brain.ps1 --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560
./scripts/run-brain.ps1 --policy-checkpoint checkpoints/policy_final.pt --mode step --step-endpoint tcp://localhost:5560
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

Checkpoints are saved as `.pt` files. The `--checkpoint` flag
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

## Temporal Backend Bake-Off

Use the migration harness to compare temporal-core candidates under actor shape parity.

```bash
uv run --project projects/actor python projects/actor/scripts/bench_temporal_backends.py --candidates mamba2,gru,lstm --batch 16 --seq-len 128 --d-model 128 --repeats 40 --warmup 10
```

Repository wrapper (writes timestamped JSON artifact):

```powershell
./scripts/run-temporal-bakeoff.ps1
./scripts/run-temporal-bakeoff.ps1 -Candidates "mamba2,gru,lstm" -Device cuda
```

Canonical backend selection must use CUDA benchmarks on native Windows and native Linux.
CPU runs are allowed only as explicit diagnostics:

```powershell
./scripts/run-temporal-bakeoff.ps1 -SkipMamba -Candidates "gru,lstm" -Device cpu -AllowCpuDiagnostic
```

## CUDA Setup (Windows First)

Install CUDA-enabled PyTorch wheels into the actor virtual environment.
Default profile is pinned for wider GPU architecture support (including `sm_61`):
`torch==2.5.1+cu121` on Python 3.12.

Actor now uses canonical `mambapy` temporal core on Windows/Linux CUDA; `mamba-ssm`
is not part of the runtime path.

```powershell
./scripts/setup-actor-cuda.ps1
./projects/actor/.venv/Scripts/python.exe ./scripts/check_gpu.py
```

Linux/WSL2 setup command (run later when available):

```bash
bash ./scripts/setup-actor-cuda.sh
./projects/actor/.venv/bin/python ./scripts/check_gpu.py
```

Example pinned install override:

```powershell
./scripts/setup-actor-cuda.ps1 -CudaTag cu121 -TorchVersion 2.5.1
```
