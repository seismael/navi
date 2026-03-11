# navi-environment

Canonical environment runtime for Ghost-Matrix.

## Responsibilities

- execute headless `sdfdag` stepping against compiled `.gmdag` assets
- publish `DistanceMatrix v2` observations in `(1, Az, El)` shape
- expose corpus preparation, runtime preflight, and throughput benchmarking

## Canonical Commands

```bash
cd projects/environment
uv sync

# Environment service shortcut
uv run environment

# Explicit service launch
uv run navi-environment serve --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag

# Prepare the full discovered corpus
uv run navi-environment prepare-corpus --force-recompile

# Compile one explicit asset
uv run navi-environment compile-gmdag --source ../../data/scenes/hssd/102343992.glb --output ../../artifacts/gmdag/corpus/hssd/102343992.gmdag --resolution 512

# Runtime checks
uv run navi-environment check-sdfdag --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag
uv run navi-environment bench-sdfdag --gmdag-file ../../artifacts/gmdag/corpus/apartment_1.gmdag --actors 4 --steps 200
```

## Windows Wrapper

```powershell
./scripts/run-environment.ps1 --mode step --pub tcp://*:5559 --rep tcp://*:5560 --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
```

## Runtime Ports

- `5559` PUB: `distance_matrix_v2`
- `5560` REP: `step_request_v2` / `step_result_v2`

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
