← [Navi Overview](../../README.md)

# torch-sdf

CUDA sphere-tracing runtime for the Navi ecosystem. Executes bounded batched
sphere tracing against compiled `.gmdag` assets and writes results directly into
preallocated PyTorch CUDA tensors through zero-copy PyBind11/LibTorch bindings.

**Format specification:** [docs/GMDAG.md](../../docs/GMDAG.md)  
**Runtime details:** [docs/SDFDAG_RUNTIME.md](../../docs/SDFDAG_RUNTIME.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Key Features

- **Bounded stackless traversal** — iterative DAG descent with explicit
  `max_steps`, horizon, and hit-epsilon semantics
- **Macro-cell void cache** — reuses empty child-cell bounds so repeated
  samples in the same void region skip redundant root traversals
- **Zero-copy execution** — reads/writes PyTorch CUDA tensors via raw pointers
  with no CPU staging
- **Strict CUDA backend** — fail-fast validation for device, dtype, shape,
  contiguity, and runtime parameters

---

## Installation

```bash
cd projects/torch-sdf
pip install -e .
```

Requires CUDA Toolkit 11.8+ and PyTorch 2.0+ with CUDA support.

**Windows:** set `CUDA_HOME` / `CUDA_PATH` before building so the extension
can locate CUDA libraries at runtime.

```powershell
$env:CUDA_HOME = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
pip install -e .
```

---

## Runtime Contract

### Inputs

| Tensor | dtype | Shape | Requirements |
|--------|-------|-------|-------------|
| `origins` | `float32` | `[batch, rays, 3]` | Contiguous CUDA |
| `dirs` | `float32` | `[batch, rays, 3]` | Contiguous CUDA, unit-length |
| `out_distances` | `float32` | `[batch, rays]` | Contiguous CUDA, preallocated |
| `out_semantics` | `int32` | `[batch, rays]` | Contiguous CUDA, preallocated |

### Parameters

- `max_steps` — positive integer (iteration limit)
- `max_distance` — finite positive float (horizon)
- `resolution` — positive integer (matches compiled asset)
- `bbox_min`, `bbox_max` — three finite floats each, strictly ordered

### Semantics

- Hit: local clearance < internal hit epsilon
- Miss: ray exits domain or exceeds horizon → semantic `0`
- Cached void: rays in same empty child cell advance to cell exit without fresh
  DAG descent

---

## Testing

```bash
# Full pipeline: hallway integrity + throughput benchmark
python tests/test_full_pipeline.py

# Focused wrapper validation
python tests/test_sphere_tracing.py
```

---

## Architectural Compliance

Any changes to bit packing, hit semantics, or tensor-boundary validation must
be mirrored in the `voxel-dag` compiler, the environment integration layer, and
their tests in the same change.
