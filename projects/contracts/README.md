# navi-contracts

Wire-format models and serialization for Ghost-Matrix services.

## Canonical Models

- `RobotPose`
- `DistanceMatrix`
- `Action`
- `StepRequest`
- `StepResult`
- `TelemetryEvent`

## Canonical Topics

- `TOPIC_DISTANCE_MATRIX`
- `TOPIC_ACTION`
- `TOPIC_STEP_REQUEST`
- `TOPIC_STEP_RESULT`
- `TOPIC_TELEMETRY_EVENT`

## Install

```bash
cd projects/contracts
uv sync
```

## Usage

```python
from navi_contracts import DistanceMatrix, Action, serialize, deserialize
```
