# TRAINING.md — Canonical Corpus Training Operations

## 1. Canonical Path

Canonical training runs on one production surface only:

1. Discover the full source-scene corpus under `data/scenes/` unless explicitly narrowed.
2. Prepare or refresh compiled `.gmdag` assets.
3. Launch `navi-actor train` on the in-process `sdfdag` backend.
4. Attach the auditor dashboard only if live inspection is needed.

Default behavior:

- full discovered corpus
- default bootstrap download set: Habitat test scenes plus public ReplicaCAD stages
- continuous training until stopped
- canonical observation contract `256x48`
- staged overwrite-first corpus refresh when explicitly requested
- compiled-corpus reuse by default after refresh, even if raw source downloads were cleaned

## 2. Prerequisites

- Python `3.12`
- CUDA-capable machine with working PyTorch CUDA runtime
- free port `5557` for actor telemetry

## 3. Refresh And Preflight

```powershell
./scripts/refresh-scene-corpus.ps1

uv run --project .\projects\environment navi-environment check-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag
uv run --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\corpus\apartment_1.gmdag --actors 4 --steps 200
```

When compiler packages or compiled-runtime dependencies change, refresh the
entire promoted corpus before using benchmark results to judge the upgrade:

```powershell
./scripts/refresh-scene-corpus.ps1
```

## 4. Canonical Commands

```powershell
# Standard continuous training
./scripts/train.ps1

# Long-duration wrapper
./scripts/train-all-night.ps1

# Canonical overnight validation wrapper
./scripts/run-nightly-validation.ps1

# Full stack training launcher
./scripts/run-ghost-stack.ps1 -Train

# First end-to-end canonical qualification surface
./scripts/qualify-canonical-stack.ps1
```

The qualification script now proves the bounded canonical flow end to end:

- dataset preflight
- optional sandboxed corpus refresh into a fresh compiled corpus root
- bounded canonical training
- passive live dashboard attach
- passive training-stream recording
- checkpoint resume from a produced periodic checkpoint
- replay plus passive replay attach

The nightly wrapper builds on those same surfaces and adds:

- focused preflight regression suites
- bounded shared-model checkpoint and resume proof
- repeated environment drift benchmarks
- an overnight soak with checkpoint and passive attach monitoring
- one machine-readable morning summary artifact root

For refresh-to-training qualification without mutating the live promoted corpus:

```powershell
./scripts/qualify-canonical-stack.ps1 -EnableCorpusRefreshQualification -RefreshSourceRoot .\data\scenes -RefreshManifest .\data\scenes\scene_manifest_all.json -RefreshScene .\data\scenes\hssd\102343992.glb
```

Explicit narrowing remains available when requested:

```powershell
./scripts/train.ps1 -Scene .\data\scenes\hssd\102343992.glb -AutoCompileGmDag
./scripts/train.ps1 -TotalSteps 500000
./scripts/run-ghost-stack.ps1 -Train -TotalSteps 500000
```

## 5. Checkpoints

- periodic checkpoints use the configured checkpoint directory
- final checkpoints are written as `policy_final.pt` when enabled
- resume with `-ResumeCheckpoint` on wrappers or `--checkpoint` on the actor CLI

Example:

```powershell
./scripts/train-all-night.ps1 -ResumeCheckpoint .\checkpoints\all_night\policy_step_0025000.pt
```

## 6. Dashboard Attach

```powershell
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

The canonical trainer guarantees actor telemetry on `5557`. Environment sockets may be idle during direct in-process training.

For scripted passive proof without launching the GUI, use:

```powershell
uv run --project .\projects\auditor navi-auditor dashboard-attach-check --actor-sub tcp://localhost:5557 --json
```

## 7. Recovery

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*navi-environment*' -or $_.CommandLine -like '*navi-actor*' -or $_.CommandLine -like '*navi-auditor*') } | ForEach-Object { try { & taskkill /PID $_.ProcessId /T /F *> $null } catch {} }
```

## 8. Operational Notes

- production training advertises only the canonical `sdfdag` path
- wrappers default to the canonical `256x48` observation contract with minibatch `64`, BPTT `8`, and rollout `512`
- corpus compilation defaults to `512`, aligned with the canonical `256x48` environment observation contract
- benchmarks, wrappers, and tests must use real dataset scenes or compiled dataset `.gmdag` assets; generated or sample scenes are not part of the canonical path
- the default bootstrap downloader now pulls the 3 public Habitat test scenes plus 5 public ReplicaCAD stage assets unless `-Datasets` is explicitly narrowed
- `refresh-scene-corpus.ps1` stages downloads and removes transient source assets after a successful corpus promotion
- the promoted live `gmdag_manifest.json` is compiled-only metadata: `source_path` is rewritten to the live `.gmdag` asset path after transient source cleanup
- canonical scene-pool training keeps actors on each scene for multiple completed episodes before rotation; the default budget is `16` completed episodes per scene across the fleet
- collision remains non-terminal in canonical training, with negative collision reward plus positive reward for increasing obstacle clearance after near-contact
- canonical environment shaping also penalizes starvation-heavy views and persistent near-field wall-hugging using ratios derived from the current spherical observation
- canonical geometry-foraging shaping positively values mid-range structure visibility, forward-sector structure reacquisition, and controlled inspection turns that reveal more geometry instead of less
- use explicit overrides only for deliberate experiments
- dashboard mode detection stays on low-volume telemetry to avoid rollout stalls
- the selected-actor dashboard observation stream defaults to a passive `10 Hz`
	cadence so live inspection remains visibly current without reintroducing
	per-step publication