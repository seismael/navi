# TRAINING.md — Canonical SDF/DAG Training Operations

**Subsystem:** End-to-end runtime operations  
**Status:** Active canonical training guide  
**Policy:** See [AGENTS.md](../AGENTS.md) for canonical-only runtime rules

---

## 1. Canonical Path

Canonical actor training runs on one path only:

1. Discover the canonical source-scene corpus unless an explicit scene override is supplied.
2. Compile or refresh the required `.gmdag` assets.
3. Launch `train` on backend `sdfdag`.
3. Attach the auditor dashboard over ZMQ if live inspection is needed.

`train` is the only production training command and runs with direct in-process
`sdfdag` stepping.

By default, canonical training uses the full discovered dataset corpus and runs
continuously until the user explicitly supplies a stop condition.

Mesh, voxel, and habitat backends are diagnostic-only and must not be used for production training.

---

## 2. Prerequisites

- Python `3.12`
- CUDA-capable machine with working PyTorch CUDA runtime
- A populated dataset corpus under `data/scenes/` or an explicit compiled `.gmdag` override
- Free ZMQ port `5557` for actor telemetry
- Free ZMQ ports `5559` and `5560` only if you also attach diagnostic tools that still listen on environment sockets

If you need to build one explicit asset manually:

```powershell
uv run --python 3.12 --project .\projects\environment navi-environment compile-gmdag --source .\data\scenes\sample_apartment.glb --output .\artifacts\gmdag\sample_apartment.gmdag --resolution 2048
```

---

## 3. Preflight Check

Validate the canonical runtime before long runs:

```powershell
uv run --python 3.12 --project .\projects\environment navi-environment bench-sdfdag --gmdag-file .\artifacts\gmdag\sample_apartment.gmdag --actors 4 --steps 200
```

This confirms the `.gmdag` asset loads, CUDA is available, and the environment can sustain batched `torch-sdf` stepping.

---

## 4. Training Commands

### 4.1. Canonical Training

```powershell
./scripts/train.ps1
```

This default canonical launch should discover all available dataset scenes,
prepare the compiled `.gmdag` corpus, and continue training until stopped.

### 4.2. Long-Duration Training

```powershell
./scripts/train-all-night.ps1
```

### 4.3. Unified Stack Launcher

```powershell
./scripts/run-ghost-stack.ps1 -Train
```

### 4.4. Explicit Override Examples

```powershell
./scripts/train.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag
./scripts/train.ps1 -TotalSteps 500000
./scripts/run-ghost-stack.ps1 -Train -TotalSteps 500000
```

Use explicit scene or duration overrides only when narrowing the canonical
default behavior on purpose.

---

## 5. Dashboard Attach

Attach the dashboard to a running training session. In the canonical direct trainer,
actor telemetry on `5557` is guaranteed while environment sockets may remain idle:

```powershell
./scripts/run-dashboard.ps1 --matrix-sub tcp://localhost:5559 --actor-sub tcp://localhost:5557 --step-endpoint tcp://localhost:5560
```

Expected window title:

```text
Ghost-Matrix RL Auditor
```

Expected log pattern:

```text
auditor.stream poll total=... topics={dm:..., action:0, telem:...}
```

---

## 6. Checkpoints

- Periodic checkpoints are written under the configured checkpoint directory.
- Final checkpoint is typically written as `policy_final.pt`.
- Resume by passing `-ResumeCheckpoint` to the wrapper or `--checkpoint` to the actor CLI.

Example:

```powershell
./scripts/train-all-night.ps1 -GmDagFile ./artifacts/gmdag/sample_apartment.gmdag -ResumeCheckpoint .\checkpoints\all_night\policy_step_0025000.pt
```

---

## 7. Recovery

If ports are stuck after a crash:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*navi-environment*' -or $_.CommandLine -like '*navi-actor*' -or $_.CommandLine -like '*navi-auditor*') } | ForEach-Object { try { & taskkill /PID $_.ProcessId /T /F *> $null } catch {} }
```

Then confirm that `5557` is no longer held before restarting training. If you also run
diagnostic environment tools, clear `5559` and `5560` as well.

---

## 8. Operational Notes

- Canonical launchers should default to the full discovered training corpus rather than a single sample scene.
- Canonical corpus tooling should refresh source data and compiled `.gmdag` assets with overwrite semantics when explicitly requested.
- Canonical training wrappers now default to the throughput-tuned compiled-path profile (`128x24`, minibatch `64`, BPTT `8`, rollout `512`).
- Override individual knobs only for explicit experiments; the repository no longer advertises alternate training profiles.
- Dashboard mode detection depends on low-volume telemetry to avoid rollout stalls.
- The canonical trainer keeps the actor contract intact while removing the
	environment step socket from the rollout hot loop; this is the primary path
	for closing the remaining Navi vs Topo-nav architecture gap.