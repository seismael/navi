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
3. **EpisodicMemory** — tensor-native cosine-similarity loop-avoidance and context retrieval
4. **TemporalCore** — canonical sequence engine (`gru` by default, `mambapy` by explicit comparison)
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

Run the only production training surface directly on the canonical corpus runtime:

```bash
cd projects/actor
uv run python -m navi_actor.cli train
```

If the default telemetry port is already occupied, override it explicitly:

```bash
uv run python -m navi_actor.cli train --actor-pub tcp://localhost:5565
```

Repository wrappers keep the same canonical path:

```powershell
./scripts/train.ps1
./scripts/train-all-night.ps1
./scripts/run-resolution-compare.ps1
```

## Resolution Scaling Reality

The actor does not scale linearly with observation resolution.

Current `RayViTEncoder` implementation in `perception.py` uses:

- strided `Conv2d` patch projection with `patch_size=8`
- fixed spherical positional encodings
- `nn.TransformerEncoder` over the resulting patch tokens

That means token count scales with `(Az / 8) * (El / 8)` and self-attention
cost grows roughly with the square of that token count.

Concrete token counts on the current profiles are:

- `256x48` -> `32 * 6 = 192` tokens
- `384x72` -> `48 * 9 = 432` tokens
- `512x96` -> `64 * 12 = 768` tokens
- `768x144` -> `96 * 18 = 1728` tokens

The canonical March 2026 resolution sweep therefore showed a split result:

- the environment path remained viable at higher ray counts
- the actor encoder plus PPO update became the dominant scaling limit
- on the active MX150 machine, the full trainer OOMs at `768x144` inside
	transformer self-attention during PPO update

Run Actor with learned policy checkpoint:

```bash
uv run python -m navi_actor.cli serve --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560 --policy-checkpoint checkpoints/policy_final.pt
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

## Canonical Temporal Profile

Use the canonical profiling harness to measure the supported temporal
candidates under actor shape parity.

```bash
uv run --project projects/actor python projects/actor/scripts/bench_temporal_backends.py --batch 16 --seq-len 128 --d-model 128 --repeats 40 --warmup 10 --device cuda
```

Repository wrapper (writes a timestamped JSON artifact):

```powershell
./scripts/run-temporal-bakeoff.ps1
```

The bakeoff now benchmarks `gru` by default on the active Windows machine. Use
`-Candidates gru mambapy` when you want an explicit comparison run.

For bounded end-to-end canonical trainer comparisons instead of microbenchmarks,
use:

```powershell
./scripts/run-temporal-compare.ps1
./scripts/run-temporal-compare.ps1 -TemporalCores @('gru','mambapy')
```

CPU runs remain explicit diagnostics only:

```powershell
./scripts/run-temporal-bakeoff.ps1 -Device cpu -AllowCpuDiagnostic
```

## CUDA Setup (Windows First)

Install CUDA-enabled PyTorch wheels into the actor virtual environment.
Default profile is pinned for wider GPU architecture support (including `sm_61`):
`torch==2.5.1+cu121` on Python 3.12.

Actor canonical runtime now defaults to the native cuDNN GRU path, with `mambapy` available on the same train and serve surfaces for explicit comparisons.

The active repository decision is to keep the profiled cuDNN GRU runtime as the
default production temporal path after repeated bounded trainer runs showed an
end-to-end throughput win over `mambapy` on the same selector surface.

Future fused Mamba-2 work is still supported, but it is deferred until a better
environment and hardware surface are available.

Optional future fused install from vendored sources:

```powershell
./scripts/setup-actor-cuda.ps1 -InstallFusedTemporal
```

Optional future fused install from a prebuilt compatible wheel:

```powershell
./scripts/setup-actor-cuda.ps1 -InstallFusedTemporal -FusedWheelPath C:\path\to\mamba_ssm-<version>-cp312-<tag>-win_amd64.whl
```

The wheel must match the actor environment exactly. On this repository that
means Python `3.12`, Windows `win_amd64`, and a fused package stack compatible
with the pinned CUDA PyTorch actor environment.

Temporal bakeoff remains diagnostic-only. The current canonical backend is the
native cuDNN GRU path. Future fused `mamba-ssm` work remains available only as
a later promotion target once a supported environment exists and the real
bounded trainer proves the win.

### Switching Back Later

When the proper environment is ready later, the switch-back path is:

1. install the fused temporal dependencies from vendored sources or a compatible wheel
2. verify that `mamba2` imports and runs on the actual machine
3. benchmark the active GRU path against any fused candidates with `run-temporal-bakeoff.ps1`
4. rerun the bounded canonical 4-actor training surface
5. only then promote the fused selector into the canonical docs, wrappers, and defaults

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
