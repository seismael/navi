# Contributing to Navi

Thank you for your interest in contributing to Navi! We welcome contributions
from the community to help make Navi a more powerful and robust reinforcement
learning system for autonomous navigation.

Navi is an open-source reinforcement learning system built around compiled
signed-distance-field geometry and a high-throughput in-process PPO trainer.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Security Policy](#security-policy)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Quality Gates](#quality-gates)
- [Project Structure](#project-structure)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Licensing of Contributions](#licensing-of-contributions)

---

## Code of Conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold
its terms. Please report unacceptable behavior to the contact channel
listed in the Code of Conduct.

---

## Security Policy

If you discover a security vulnerability, please follow our
[Security Policy](SECURITY.md) to report it responsibly.

---

## How to Contribute

There are many ways to contribute:

- **Report bugs** with reproducible steps, expected vs. actual behavior, and
  relevant log output.
- **Propose features or architectural changes** by opening a discussion issue
  before sending a pull request.
- **Improve documentation** in [README.md](README.md), the [docs/](docs/)
  index, or per-project READMEs.
- **Submit pull requests** for fixes, performance improvements, new tests, or
  new benchmark surfaces.

All non-trivial changes should align with the standards in
[AGENTS.md](AGENTS.md), which serves as the authoritative implementation policy
for the project (configuration, logging, artifact governance, and architectural
principles).

---

## Development Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | Managed via `uv` |
| [uv](https://docs.astral.sh/uv/) | Latest | Package manager for all sub-projects |
| CUDA Toolkit | 12.1+ | Required for GPU training and sphere tracing |
| C++ Compiler | MSVC 2022 / GCC 12+ | For `voxel-dag` and `torch-sdf` native builds |
| PowerShell | 5.1+ | Orchestration scripts |

### Install

```powershell
# Install dependencies in every sub-project
make sync-all

# Install GPU-enabled PyTorch into the actor environment
.\scripts\setup-actor-cuda.ps1

# Verify GPU availability
python scripts\check_gpu.py
```

See the [Quick Start](README.md#quick-start) in the README for the full
end-to-end bootstrap, including corpus preparation and a first training run.

---

## Quality Gates

Per [AGENTS.md](AGENTS.md) section 2.5, the following are mandatory for all
changes:

| Gate | Command | Scope |
|------|---------|-------|
| Lint | `make lint-all` | `ruff check` + `ruff format --check` across all projects |
| Format | `make format-all` | `ruff format` across all projects |
| Types | `make typecheck-all` | `mypy --strict` across all projects |
| Tests | `make test-all` | `pytest` in every sub-project |
| Full CI gate | `make check-all` | lint + typecheck + tests |

A pull request is not ready for review until `make check-all` passes locally.

---

## Project Structure

Navi is composed of six sovereign packages, each with its own virtual
environment and `pyproject.toml`. **No service package imports another** - all
cross-project integration occurs at CLI orchestration boundaries
(`scripts/` PowerShell wrappers, or the canonical training entrypoint).

| Project | Role |
|---------|------|
| [projects/contracts](projects/contracts/) | Wire-format models, ZMQ topics |
| [projects/environment](projects/environment/) | Headless `sdfdag` stepping, corpus tooling |
| [projects/actor](projects/actor/) | Sacred cognitive engine, PPO trainer |
| [projects/auditor](projects/auditor/) | Dashboard, recording, replay, exploration |
| [projects/voxel-dag](projects/voxel-dag/) | Mesh -> `.gmdag` offline compiler |
| [projects/torch-sdf](projects/torch-sdf/) | CUDA sphere-tracing runtime |

When making changes:

- Keep edits scoped to the project that owns the concern.
- Do not introduce service-to-service imports.
- Update producer, consumer, tests, and docs in the same change when a
  contract evolves (no partial dual-path support - see
  [AGENTS.md](AGENTS.md) section 2.6).
- Avoid creating compatibility wrappers, deprecated aliases, or legacy
  fallback paths.

---

## Pull Request Process

1. **Open an issue first** for non-trivial changes so design questions can
   be discussed before implementation.
2. **Branch from `main`** and keep your branch focused on one logical change.
3. **Run the full quality gate** (`make check-all`) before pushing.
4. **Write or update tests** that exercise your change. Performance work
   should reference (or add) a benchmark surface.
5. **Update documentation** in the same PR - README, per-project README,
   or relevant `docs/*.md`.
6. **Describe the change clearly** in the PR description: what changed,
   why, what was tested, and any benchmark numbers if relevant.
7. **Respond to review feedback** promptly and keep the discussion in the
   PR thread.

---

## Reporting Issues

When opening an issue, please include:

- A clear, descriptive title.
- The Navi commit SHA you are running.
- Your platform (OS, Python version, CUDA version, GPU model).
- For bugs: reproduction steps, expected behavior, actual behavior, and the
  relevant log excerpt from `logs/` or the active run root under
  `artifacts/runs/<run_id>/logs/`.
- For performance regressions: benchmark surface used, the measurement, and
  the baseline you are comparing against.
- For feature requests: the use case and why existing surfaces do not cover
  it.

---

## Licensing of Contributions

Navi is licensed under the [Apache License, Version 2.0](LICENSE).

By submitting a contribution (a pull request, patch, or any other code or
documentation change) you agree that your contribution is licensed under the
same Apache 2.0 license that covers the project (commonly known as
"inbound = outbound"). No separate Contributor License Agreement (CLA) and
no Developer Certificate of Origin (DCO) sign-off is required at this time.

If you are contributing on behalf of an employer, please ensure you have
their permission to do so before submitting.
