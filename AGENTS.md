# AGENTS.md — Ghost-Matrix Implementation Blueprint

This file is the implementation policy for `ARCHITECTURE.md`.

## 1) Scope

Navi is a Ghost-Matrix system focused on throughput RL with strict separation of:
- Simulation Layer (headless stepping + sensing)
- Brain Layer (policy + training)
- Gallery Layer (record/replay/visualization)

## 2) Non-Negotiables

1. Canonical wire contracts are v2 only:
   - `RobotPose`
   - `DistanceMatrix`
   - `Action`
   - `StepRequest`
   - `StepResult`
   - `TelemetryEvent`
2. Legacy wire contracts/topics are not allowed in new code.
3. Inter-process communication is ZMQ only (PUB/SUB + REQ/REP).
4. Training runtime is headless; rendering is optional and asynchronous.
5. Services are sovereign packages; no service imports another service package.
6. Code quality gates remain mandatory: `ruff`, `mypy --strict`, `pytest`.

## 3) Repository Structure (Target)

```text
navi/
├── ARCHITECTURE.md
├── AGENTS.md
├── NEW_APPROACH.md
├── README.md
├── Makefile
├── scripts/
│   └── run-ghost-stack.ps1
└── projects/
    ├── contracts/
    │   └── src/navi_contracts/
    │       ├── models.py
    │       ├── topics.py
    │       ├── serialization.py
    │       └── types.py
    ├── section-manager/
    │   └── src/navi_section_manager/
    │       ├── server.py
    │       ├── mjx_env.py
    │       ├── raycast.py
    │       └── distance_matrix_v2.py
    ├── actor/
    │   └── src/navi_actor/
    │       ├── server.py
    │       ├── policy.py
    │       └── training/
    └── auditor/
        └── src/navi_auditor/
            ├── recorder.py
            ├── rewinder.py
            └── matrix_viewer.py
```

## 4) Active Runtime Topics

- `distance_matrix_v2`
- `action_v2`
- `step_request_v2`
- `step_result_v2`
- `telemetry_event_v2`

## 5) Implementation Rules

- Every module must use `from __future__ import annotations`.
- Public modules and all package `__init__.py` must define `__all__`.
- All functions/methods require full type annotations.
- Keep module sizes focused and single-responsibility.
- Remove dead code during migration; do not leave deprecated branches.

## 6) Migration Policy

- This repository is in hard-cut migration mode.
- Allowed projects for final architecture: `contracts`, `section-manager`, `actor`, `auditor`.
- `ingress` and `cartographer` are removed from active architecture.
- Any new code must target v2 contracts and Ghost-Matrix flow only.

## 7) Validation Commands

Per project:

```bash
uv sync
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```

Repository goal:

```bash
make check-all
```
