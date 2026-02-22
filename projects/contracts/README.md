# navi-contracts

Wire-format models and serialization for Ghost-Matrix services.

**Full specification:** [docs/CONTRACTS.md](../../docs/CONTRACTS.md)  
**Implementation policy:** [AGENTS.md](../../AGENTS.md)

## Canonical Models

| Model | Description |
|-------|-------------|
| `RobotPose` | 6-DOF pose `(x, y, z, roll, pitch, yaw, timestamp)` |
| `DistanceMatrix` | Spherical depth observation `(n_envs, Az, El)` with semantic + delta-depth |
| `Action` | 4-DOF velocity command `[forward, vertical, lateral, yaw]` |
| `StepRequest` | Discrete step request from Brain → Simulation (REQ/REP) |
| `StepResult` | Step result with reward, done, truncated flags |
| `TelemetryEvent` | Generic numeric telemetry for dashboarding and replay |

## Canonical Topics

| Constant | Value | Transport |
|----------|-------|-----------|
| `TOPIC_DISTANCE_MATRIX` | `distance_matrix_v2` | PUB/SUB |
| `TOPIC_ACTION` | `action_v2` | PUB/SUB |
| `TOPIC_STEP_REQUEST` | `step_request_v2` | REQ/REP |
| `TOPIC_STEP_RESULT` | `step_result_v2` | REQ/REP |
| `TOPIC_TELEMETRY_EVENT` | `telemetry_event_v2` | PUB/SUB |

## Install

```bash
cd projects/contracts
uv sync
```

## Usage

```python
from navi_contracts import DistanceMatrix, Action, serialize, deserialize

# Serialize any canonical model
payload: bytes = serialize(distance_matrix)

# Deserialize (auto-detects type)
msg = deserialize(payload)
```

## Validation

```bash
uv run ruff check .
uv run mypy src/
uv run pytest tests/
```
