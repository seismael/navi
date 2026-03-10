# TRAINING.md - Canonical Corpus Training Operations

## 1. Executive Summary

Canonical training in Navi runs on one production surface only:

1. discover the full source-scene corpus unless explicitly narrowed
2. prepare or refresh compiled `.gmdag` assets
3. launch `navi-actor train` on the in-process `sdfdag` backend
4. attach the auditor only if live inspection is needed

The operational defaults are deliberately biased toward production throughput:

- full discovered corpus
- continuous training until stopped
- staged overwrite-first corpus refresh when explicitly requested
- compiled-corpus reuse after refresh even if transient raw downloads were removed

## 2. Preconditions

Training requires:

- Python `3.12`
- a CUDA-capable machine with working PyTorch CUDA runtime
- the compiled runtime dependencies needed by `torch-sdf`
- free actor telemetry port `5557`
- access to a valid compiled corpus or ability to prepare one

## 3. Canonical Operational Sequence

The expected operator sequence is:

1. refresh or validate the corpus when needed
2. verify runtime readiness with `check-sdfdag`
3. attribute environment performance with `bench-sdfdag` if needed
4. launch canonical training
5. optionally attach the dashboard in passive mode
6. checkpoint and recover through the canonical wrappers when needed

## 4. Refresh And Preflight

```powershell
./scripts/refresh-scene-corpus.ps1

uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag --actors 4 --steps 200
```

Purpose of these commands:

- `refresh-scene-corpus.ps1` updates staged source data and promotes a clean compiled corpus
- `check-sdfdag` proves compiler/runtime readiness and validates one compiled asset
- `bench-sdfdag` attributes environment-layer throughput before blaming the actor

## 5. Canonical Launch Commands

```powershell
# Standard continuous training
./scripts/train.ps1

# Long-duration wrapper
./scripts/train-all-night.ps1

# Full stack training launcher
./scripts/run-ghost-stack.ps1 -Train
```

Explicit narrowing remains available when deliberately requested:

```powershell
./scripts/train.ps1 -Scene .\data\scenes\replicacad\frl_apartment_stage.glb -AutoCompileGmDag
./scripts/train.ps1 -TotalSteps 500000
./scripts/run-ghost-stack.ps1 -Train -TotalSteps 500000
```

## 6. Canonical Defaults

Production-safe defaults emphasize throughput and stable operation:

- `sdfdag` runtime only
- full discovered corpus unless overridden
- compile resolution `512`
- throughput-safe training profile in wrappers
- continuous training until stopped unless explicitly bounded

The canonical repository stance is that single-scene or short-run examples are
explicit overrides, not default training narratives.

## 7. Checkpointing And Resume

Checkpoint behavior:

- periodic checkpoints use the configured checkpoint directory
- final checkpoints are written as `policy_final.pt` when enabled
- resume uses `-ResumeCheckpoint` on wrappers or `--checkpoint` on the actor CLI

Example:

```powershell
./scripts/train-all-night.ps1 -ResumeCheckpoint .\checkpoints\all_night\policy_step_0025000.pt
```

## 8. Dashboard Attach

```powershell
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

Training-time notes:

- the canonical trainer guarantees actor telemetry on `5557`
- environment sockets may be idle during direct in-process training
- dashboard mode detection must rely on actor telemetry and remain passive

## 9. Recovery And Clean Restart

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*navi-environment*' -or $_.CommandLine -like '*navi-actor*' -or $_.CommandLine -like '*navi-auditor*') } | ForEach-Object { try { & taskkill /PID $_.ProcessId /T /F *> $null } catch {} }
```

Operational rule:

- prefer robust cleanup over partial restart
- verify fresh socket binding after restart
- do not assume stale ports are safe to reuse without validation

## 10. Operational Policies

### 10.1 Corpus Policy

- canonical training advertises only the compiled-corpus path
- promoted manifest entries must point at live `.gmdag` assets
- transient source downloads may be removed after successful promotion

### 10.2 Scene Residency Policy

- scene-pool training keeps actors on a scene for multiple completed episodes
- default budget is `16` completed episodes per scene across the fleet
- this reduces scene-swap overhead and improves local recovery learning

### 10.3 Collision Policy

- collision remains non-terminal in canonical training
- collision penalty and obstacle-clearance recovery reward remain active
- starvation-heavy views and persistent near-field wall-hugging are penalized

### 10.4 Observability Policy

- dashboard mode detection stays on low-volume telemetry
- dashboards remain optional and passive
- coarse heartbeat publication is allowed only as a training-safe convenience

## 11. Training Profiles And Interpretation

The current production workflow should be interpreted through the live bottleneck
map:

- environment acceleration has already shifted a large share of the remaining
  work to actor dataflow and PPO update cost
- `bench-sdfdag` is useful for proving environment readiness but does not settle
  trainer bottlenecks
- full-training regressions must be interpreted with actor perf telemetry,
  especially host extraction, telemetry publication, and PPO update timing

## 12. Experimentation Boundaries

Allowed experiments:

- explicit scene narrowing
- explicit total-step bounds
- explicit benchmarking and ablation runs
- explicit diagnostic toggles on the canonical trainer

Disallowed as documentation default:

- advertising alternate backends as production training paths
- treating service mode as equal to the direct trainer for throughput work
- importing external performance claims as current repo truth

## 13. Immediate Planning Context

The next production-significant implementation work implied by the current docs
is:

1. remove remaining unconditional observation materialization on the canonical path
2. reduce remaining action and scalar host extraction inside rollout steps
3. optimize PPO minibatch and optimizer flow before deeper environment redesign
4. add stronger runtime-boundary and long-run allocation validation

## 14. Related Docs

- `docs/ARCHITECTURE.md`
- `docs/SIMULATION.md`
- `docs/ACTOR.md`
- `docs/PERFORMANCE.md`
- `PLAN.md`
