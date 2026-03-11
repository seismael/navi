# NIGHTLY_VALIDATION.md — Canonical Overnight Validation

## 1. Goal

The nightly pipeline exists to answer four questions with one run:

1. can the canonical stack start cleanly on the promoted corpus
2. do the bounded training, checkpoint, resume, and passive observer surfaces still work end to end
3. does long-running canonical training remain numerically stable and operationally healthy
4. is there evidence that the shared model is improving over time rather than silently degrading

This is an operator-confidence workflow, not a formal proof system.

## 2. Default Policy

- default duration: `8` hours
- default fleet: `4` actors
- hard-gate policy: fail fast before the soak if preflight or bounded qualification fails
- soft-warning policy: collect warnings for drift or attach instability without aborting a healthy run
- artifact root: `artifacts/nightly/<timestamp>/`

## 3. Canonical Command

```powershell
./scripts/run-nightly-validation.ps1
```

Optional duration override:

```powershell
./scripts/run-nightly-validation.ps1 -SoakHours 12
```

## 4. Phase Layout

1. bootstrap artifact root and run manifest
2. reclaim ports and stop stale Navi processes
3. runtime preflight via `check-sdfdag` and `dataset-audit`
4. focused actor, environment, and auditor regression suites
5. bounded canonical qualification via `qualify-canonical-stack.ps1`
6. bounded shared-model training plus checkpoint summary
7. bounded checkpoint resume proof
8. repeated environment benchmark drift checks
9. overnight canonical soak with periodic checkpoint and attach monitoring
10. morning-facing summary and baseline diff artifacts

## 5. Hard Gates

The nightly must stop immediately when any of the following occurs:

- runtime preflight reports `ok: false`
- focused regression suites fail
- bounded canonical qualification fails
- bounded shared-model training does not emit checkpoints or produces non-finite metrics
- bounded resume proof cannot produce a fresh checkpoint
- overnight trainer crashes
- overnight checkpoints stop advancing beyond the configured stall threshold
- repeated non-finite metrics are detected in the soak logs

## 6. Soft Warnings

The nightly should record warnings, but not stop the run, for:

- repeated dashboard attach failures on the actor PUB stream
- environment benchmark drift relative to the last accepted baseline
- rollout throughput degradation without an accompanying crash
- unusually slow PPO update windows that still recover

## 7. Morning Review

The top-level artifacts to inspect are:

- `reports/nightly_summary.json`
- `reports/nightly_summary.md`
- `reports/nightly_diff_vs_baseline.json`

Those reports should be enough to decide whether the night was healthy before opening raw logs.