# Performance Architecture Notes

## Verified Status

The repository is aligned around one canonical training path:

- scene compilation through `projects/voxel-dag` into `.gmdag`
- batched CUDA ray execution through `projects/torch-sdf`
- direct in-process actor training over `SdfDagBackend`
- GRU as the active temporal-core default on the Windows training machine
- passive auditor attachment only when explicitly requested

Legacy helper code removed during this work stays removed. Legacy public server
methods still exist only where they are actively used by the service-mode ZMQ
surface.

## Implemented And Active

### 1. Tensor-Native Canonical Trainer

- the canonical PPO trainer runs environment stepping in process instead of over ZMQ
- rollout storage stays on the active device in `MultiTrajectoryBuffer`
- environment tensor stepping accepts `(actors, 4)` action tensors directly
- hot-path observation, reward, and rollout bookkeeping remain tensor-first

### 2. SDF/DAG Runtime

- compiled assets are produced at the canonical resolution `512`
- `torch_sdf.cast_rays()` consumes contiguous CUDA tensors shaped `[batch, rays, 3]`
- observation output remains the canonical spherical contract with actor batches shaped `N x 3 x 256 x 48`
- DAG buffers and ray buffers stay GPU-resident across steps on the canonical runtime

### 3. Hot-Path Fusion And Fallbacks

- environment stateless tensor helpers use `torch.compile(fullgraph=True)` when supported
- actor reward shaping uses the same compile strategy with `torch.jit.script` and eager fallback
- on SM `< 7.0`, Navi now skips `torch.compile` up front instead of failing lazily on first execution
- on unsupported stacks, the runtime logs the skip reason and continues on scripted or eager tensor code

### 4. Encoder And Temporal Core

- `RayViTEncoder` has compile wrapping with eager fallback and existing positional-encoding caching
- encoder compile remains disabled automatically on stacks that cannot support it cleanly
- canonical temporal execution is `torch.nn.GRU(batch_first=True)` on the active Windows machine
- `mambapy` remains an explicit comparison backend, not a second canonical path

### 5. Config Alignment

The actor defaults in config, CLI, and runtime constructors now agree on the
production settings used by the canonical trainer:

- `minibatch_size=64`
- `bptt_len=8`
- `value_coeff=0.5`
- `existential_tax=-0.02`
- `velocity_weight=0.1`
- `intrinsic_coeff_init=1.0`
- `loop_penalty_coeff=2.0`

## Runtime Notes

### Current Windows Training Machine

- GPU class: MX150 / SM 6.1
- practical outcome: `torch.compile` cannot be the active canonical fast path there
- active fallback: scripted or eager tensor helpers, plus cuDNN GRU for the temporal core

### Dashboard Behavior

- `run-ghost-stack.ps1 -Train` is valid with no dashboard attached
- detached wrapper launches disable observation streaming unless `-WithDashboard` is requested
- passive dashboard attach works from actor telemetry alone, but live observation frames require the observation stream to be enabled at launch
- selector mode now still starts focused on actor `0` instead of opening on a blank actor target
- actor discovery now comes from a one-shot trainer roster query and selector changes update the trainer-side selected actor over a dedicated control endpoint
- canonical selected-actor viewing no longer requires widening rich dashboard telemetry to every actor in the fleet

### Durable Long-Run Surface

For long uninterrupted training with variable actor count, the direct actor CLI
is the durable surface because it exposes `--actors` directly and does not tie
viewer lifecycle to the wrapper process.

## Current Bottlenecks

The main remaining bottleneck on the active Windows machine is still the
environment side, not the PPO learner:

- rollout throughput is below the long-term `>= 60 SPS` target on MX150
- `env_ms` remains the dominant timing bucket
- GRU forward time is materially lower than the prior Python-heavy temporal path

## Remaining Optional Work

These are performance opportunities, not open correctness or cleanup gaps:

1. CUDA macro-cell or void-distance caching in `projects/torch-sdf/cpp_src/kernel.cu`
2. larger-fleet overlap or ping-pong execution only after benchmark proof on stronger hardware

## Canonical Data Flow

```text
Scene mesh
	-> voxel-dag compiler
	-> .gmdag asset
	-> torch-sdf batched ray casting
	-> SdfDagBackend tensor postprocess and reward
	-> RayViTEncoder
	-> GRU temporal core
	-> PPO rollout/update loop
```