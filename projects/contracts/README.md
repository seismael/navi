← [Navi Overview](../../README.md)

# navi-contracts

Wire-format models and serialization for the Navi ecosystem. This is the shared
package consumed by all other projects — it defines the canonical data models,
ZMQ topic constants, and MessagePack serialization helpers.

**Full specification:** [docs/CONTRACTS.md](../../docs/CONTRACTS.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

---

## Install

```bash
cd projects/contracts
uv sync
```

This is a library package — no long-running service process.

---

## Canonical Models

| Model | Description |
|-------|-------------|
| `RobotPose` | 6-DOF pose `(x, y, z, roll, pitch, yaw, timestamp)` |
| `DistanceMatrix` | Spherical depth observation `(n_envs, Az, El)` with semantic + delta-depth channels |
| `Action` | 4-DOF velocity command `[forward, vertical, lateral, yaw]` |
| `StepRequest` | Step request from Brain → Environment (REQ/REP) |
| `StepResult` | Step result with reward, done, truncated, episode return |
| `TelemetryEvent` | Keyed numeric telemetry for dashboarding and replay |

## Wire Protocol

All inter-service communication in the Navi ecosystem uses ZMQ with MessagePack
serialization. This package defines the canonical wire formats consumed by all
other projects.

| Topic | Direction | Transport |
|-------|-----------|-----------|
| `distance_matrix_v2` | Environment → Brain, Gallery | PUB/SUB |
| `action_v2` | Brain → Environment, Gallery | PUB/SUB |
| `step_request_v2` | Brain → Environment | REQ/REP |
| `step_result_v2` | Environment → Brain | REQ/REP |
| `telemetry_event_v2` | Any → Gallery | PUB/SUB |

### Default Network Ports

| Port | Service | Role |
|------|---------|------|
| `5559` | Environment | PUB (observation broadcast) |
| `5560` | Environment | REP (step request/response) |
| `5557` | Actor | PUB (action + telemetry broadcast) |

Ports are configurable via the root `.env` file or CLI parameters. The unified
trainer only uses port `5557` for the passive dashboard telemetry stream.

---

## Usage

```python
from navi_contracts import DistanceMatrix, Action, serialize, deserialize

# Serialize any canonical model
payload: bytes = serialize(distance_matrix)

# Deserialize (auto-detects type)
msg = deserialize(payload)
```

---

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
