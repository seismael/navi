# Navi — Ghost-Matrix Throughput RL System

Navi is a headless-first, message-driven RL stack with three layers:
- Simulation Layer (`section-manager`)
- Brain Layer (`actor`)
- Gallery Layer (`auditor`)

Shared wire contracts live in `projects/contracts`.

## Active Projects

- `projects/contracts`
- `projects/section-manager`
- `projects/actor`
- `projects/auditor`

## Canonical Topics

- `distance_matrix_v2`
- `action_v2`
- `step_request_v2`
- `step_result_v2`
- `telemetry_event_v2`

## Quick Start

```bash
# Terminal 1 — Simulation Layer
cd projects/section-manager
uv sync

# Optional: compile PLY world assets to canonical sparse zarr
uv run navi-section-manager compile-world --source ../../assets/world.ply --output ../../assets/world.zarr
# Optional: OBJ/STL sources are also supported
# uv run navi-section-manager compile-world --source ../../assets/world.obj --source-format obj --output ../../assets/world.zarr

uv run navi-section-manager serve --mode step --pub tcp://*:5559 --rep tcp://*:5560

# Terminal 2 — Brain Layer
cd ../actor
uv sync
uv run navi-actor run --sub tcp://localhost:5559 --pub tcp://*:5557 --mode step --step-endpoint tcp://localhost:5560

# Terminal 3 — Gallery Layer
cd ../auditor
uv sync
uv run navi-auditor record --sub tcp://localhost:5559,tcp://localhost:5557 --out session.zarr
```

Windows one-command run:

```powershell
./scripts/run-ghost-stack.ps1
```

Windows one-command run with learned policy checkpoint:

```powershell
./scripts/run-ghost-stack.ps1 -ActorPolicy learned -ActorPolicyCheckpoint "projects/actor/artifacts/policy_spherical.npz"
```

## Repository Commands

```bash
make sync-all
make test-all
make check-all
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [AGENTS.md](AGENTS.md)
- [NEW_APPROACH.md](NEW_APPROACH.md)
