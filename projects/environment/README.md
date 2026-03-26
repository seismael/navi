# navi-environment

Simulation Layer for the Navi ecosystem. Executes headless batched environment
stepping against compiled `.gmdag` assets using the CUDA `sdfdag` backend and
publishes `DistanceMatrix v2` observations.

**Full specification:** [docs/SIMULATION.md](../../docs/SIMULATION.md)  
**Runtime details:** [docs/SDFDAG_RUNTIME.md](../../docs/SDFDAG_RUNTIME.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Install

```bash
cd projects/environment
uv sync
```

---

## Responsibilities

- Batched `sdfdag` stepping against compiled `.gmdag` assets on the GPU
- Publishing `DistanceMatrix v2` observations in `(n_envs, Az, El)` shape
- Corpus preparation: discovery, compilation, and validation of `.gmdag` assets
- Runtime preflight checks and throughput benchmarking

---

## Commands

```bash
# Environment service (shortcut)
uv run environment

# Explicit service launch
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560 \
    --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag

# Prepare the full discovered corpus
uv run navi-environment prepare-corpus --force-recompile

# Compile a single asset
uv run navi-environment compile-gmdag \
    --source ../../data/scenes/hssd/102343992.glb \
    --output ../../artifacts/gmdag/corpus/hssd/102343992.gmdag \
    --resolution 512

# Runtime validation
uv run navi-environment check-sdfdag \
    --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag

# Throughput benchmark
uv run navi-environment bench-sdfdag \
    --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag \
    --actors 4 --steps 200
```

### Windows Wrapper

```powershell
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560 \
    --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
```

---

## Default Ports

| Port | Topic | Transport |
|------|-------|-----------|
| `5559` | `distance_matrix_v2` | PUB |
| `5560` | `step_request_v2` / `step_result_v2` | REP |

---

## Torch Compile

Environment hot-path fusion is enabled by default (`NAVI_SDFDAG_TORCH_COMPILE=1`).
On GPUs where the inductor/triton stack cannot compile (e.g. `sm_61`), the
runtime logs a warning and continues on the eager tensor path. Use
`--no-torch-compile` or `NAVI_SDFDAG_TORCH_COMPILE=0` for A/B attribution runs.

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
