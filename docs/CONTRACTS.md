# CONTRACTS.md - Canonical Wire And Tensor Contract Specification

**Package:** `navi-contracts`
**Status:** Active canonical specification
**Policy:** See `AGENTS.md` for implementation rules and non-negotiables

## 1. Purpose

This document defines the stable data contracts that connect Navi domains.
It distinguishes between:

- public wire contracts that remain stable across services
- internal tensor seams used to keep the canonical training hot path on CUDA

That distinction matters because the repository now optimizes aggressively below
the wire boundary without reopening service compatibility.

## 2. Contract Philosophy

Navi keeps a narrow contract surface by design.

The public contract must be:

- stable enough for service mode, replay, and diagnostics
- expressive enough for training and inference
- small enough that new runtime optimizations do not require new wire models

The internal tensor seams must be:

- explicit
- performance-oriented
- non-public
- incapable of silently widening the wire protocol

## 3. Canonical Public Models

The following dataclasses are the only models permitted on the inter-process
wire. Visualization payloads such as camera images or rendered RGB panels are
not public contracts.

### 3.1 RobotPose

```python
@dataclass(frozen=True, slots=True)
class RobotPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    timestamp: float
```

Purpose:

- carry full 6-DOF pose metadata with each observation
- support diagnostics, replay, and auditor overlays

### 3.2 DistanceMatrix

```python
@dataclass(frozen=True, slots=True)
class DistanceMatrix:
    episode_id: int
    env_ids: NDArray[int32]
    matrix_shape: tuple[int, int]
    depth: NDArray[float32]
    delta_depth: NDArray[float32]
    semantic: NDArray[int32]
    valid_mask: NDArray[bool_]
    overhead: NDArray[float32]
    robot_pose: RobotPose
    step_id: int
    timestamp: float
```

Purpose:

- canonical observation contract for training, service mode, replay, and passive tooling
- preserve actor-side observation semantics across runtime changes below the boundary

Shape and meaning:

- `matrix_shape == (azimuth_bins, elevation_bins)`
- default resolution is `(256, 48)`
- single-environment backends emit arrays shaped `(1, Az, El)`
- `depth` is normalized into `[0, 1]` by the configured environment horizon
- `delta_depth` is the per-bin temporal difference
- `semantic` uses canonical integer ids
- `valid_mask` is `True` only for rays considered valid hits under the current runtime contract
- `overhead` is a passive-gallery field only and is not part of the training signal

### 3.3 Action

```python
@dataclass(frozen=True, slots=True)
class Action:
    env_ids: NDArray[int32]
    linear_velocity: NDArray[float32]
    angular_velocity: NDArray[float32]
    policy_id: str
    step_id: int
    timestamp: float
```

Purpose:

- carry the actor's motion command across service boundaries
- preserve a stable public mapping even though the actor internally works in 4-DOF form

Current public mapping from internal 4-DOF action:

- `linear_velocity[:, 0]` = forward
- `linear_velocity[:, 1]` = vertical
- `linear_velocity[:, 2]` = lateral
- `angular_velocity[:, 2]` = yaw

### 3.4 StepRequest

```python
@dataclass(frozen=True, slots=True)
class StepRequest:
    action: Action
    step_id: int
    timestamp: float
```

### 3.5 StepResult

```python
@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: int
    env_id: int
    episode_id: int
    done: bool
    truncated: bool
    reward: float
    episode_return: float
    timestamp: float
```

### 3.6 TelemetryEvent

```python
@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event_type: str
    episode_id: int
    env_id: int
    step_id: int
    payload: NDArray[float32]
    timestamp: float
```

Purpose:

- carry coarse numeric telemetry for dashboards, logs, and replay
- keep observability decoupled from the main wire contracts used for stepping

### 3.7 BatchStepRequest

```python
@dataclass(frozen=True, slots=True)
class BatchStepRequest:
    actions: tuple[Action, ...]
    step_id: int
    timestamp: float
```

### 3.8 BatchStepResult

```python
@dataclass(frozen=True, slots=True)
class BatchStepResult:
    results: tuple[StepResult, ...]
    observations: tuple[DistanceMatrix, ...]
```

## 4. Shape And Convention Rules

### 4.1 Spherical Convention

Canonical spherical rules:

- `DistanceMatrix` layout is `(azimuth, elevation)`
- azimuth bin `0` is forward along local `-Z`
- dashboard forward-FOV extraction must center the azimuth seam before cropping
- canonical training datasets come from compiled corpus assets under `artifacts/gmdag/corpus`

### 4.2 Public Array Expectations

| Field | Shape | Meaning |
| --- | --- | --- |
| `depth` | `(B, Az, El)` | normalized distances in `[0, 1]` |
| `delta_depth` | `(B, Az, El)` | temporal per-bin depth change |
| `semantic` | `(B, Az, El)` | semantic ids |
| `valid_mask` | `(B, Az, El)` | validity under current runtime horizon rules |
| `linear_velocity` | `(B, 3)` | forward, vertical, lateral |
| `angular_velocity` | `(B, 3)` | roll, pitch, yaw rates |

## 5. Type Aliases

Defined in `navi_contracts.types` for static analysis and documentation:

| Alias | Underlying Type | Shape | Description |
| --- | --- | --- | --- |
| `MatrixShape` | `tuple[int, int]` | - | `(azimuth_bins, elevation_bins)` |
| `DepthMatrix` | `NDArray[float32]` | `(B, Az, El)` | normalized depth |
| `DeltaDepthMatrix` | `NDArray[float32]` | `(B, Az, El)` | temporal depth delta |
| `SemanticMatrix` | `NDArray[int32]` | `(B, Az, El)` | semantic ids |
| `ValidMask` | `NDArray[bool_]` | `(B, Az, El)` | valid-hit mask |
| `EnvIdVector` | `NDArray[int32]` | `(B,)` | environment ids |
| `VelocityMatrix` | `NDArray[float32]` | `(B, 3)` | linear or angular commands |
| `TelemetryPayload` | `NDArray[float32]` | `(N, M)` | generic numeric telemetry payload |

## 6. ZMQ Topics

All inter-service communication uses versioned topic strings.

| Constant | Value | Transport | Direction |
| --- | --- | --- | --- |
| `TOPIC_DISTANCE_MATRIX` | `distance_matrix_v2` | PUB/SUB | Simulation -> Brain, Gallery |
| `TOPIC_ACTION` | `action_v2` | PUB/SUB | Brain -> Simulation, Gallery |
| `TOPIC_STEP_REQUEST` | `step_request_v2` | REQ/REP | Brain -> Simulation |
| `TOPIC_STEP_RESULT` | `step_result_v2` | REQ/REP | Simulation -> Brain |
| `TOPIC_TELEMETRY_EVENT` | `telemetry_event_v2` | PUB/SUB | Any -> Gallery |

No other public topics may be added without explicit approval.

## 7. Internal Tensor Seams

The imported documentation was strongest when it documented low-level tensor
boundaries explicitly. Navi now does the same while keeping these seams clearly
separate from the public wire surface.

### 7.1 `torch_sdf.cast_rays()` Boundary

Canonical runtime tensors:

| Tensor | Shape | Dtype | Device | Requirement |
| --- | --- | --- | --- | --- |
| `origins` | `[B, R, 3]` | `float32` | CUDA | contiguous |
| `dirs` | `[B, R, 3]` | `float32` | CUDA | contiguous |
| `out_distances` | `[B, R]` | `float32` | CUDA | preallocated |
| `out_semantics` | `[B, R]` | `int32` | CUDA | preallocated |

Rules:

- device placement, rank, and contiguity are validated before kernel launch
- canonical runtime is CUDA-only at this boundary
- long kernel execution should release the Python GIL

### 7.2 Environment-To-Trainer Observation Seam

The canonical trainer may consume tensor-native observations directly when the
runtime provides them.

Current observation tensor:

- shape `(B, 3, Az, El)`
- channel `0`: normalized depth
- channel `1`: semantic ids cast to `float32`
- channel `2`: valid mask cast to `float32`

This seam exists to keep the hot path on CUDA.
It does not replace the public `DistanceMatrix` contract.

### 7.3 Action Tensor Seam

The canonical trainer may also step the environment from batched action tensors
before any public `Action` materialization is needed.

That seam is valid only inside the canonical training runtime.
It does not authorize a new wire contract.

### 7.4 Materialization Rule

`DistanceMatrix` and `Action` remain the canonical public and diagnostic models.
On the production training hot path they may be materialized only when needed
for:

- passive dashboard publication
- coarse telemetry
- service mode
- replay and recorder compatibility
- tests and explicit diagnostics

Unconditional rebuilding of these objects inside the rollout loop is a
performance bug.

## 8. Serialization

### 8.1 Wire Encoding

All public messages are serialized via `msgpack` with custom extension support
for numpy arrays.

Encoding steps:

1. convert dataclass fields into a msgpack-compatible mapping
2. pack numpy arrays while preserving dtype and shape
3. serialize nested pose metadata as structured fields

### 8.2 ZMQ Frame Layout

Messages are sent as multipart frames:

```text
Frame 0: topic bytes
Frame 1: payload bytes
```

### 8.3 API

```python
from navi_contracts import serialize, deserialize

payload = serialize(distance_matrix)
msg = deserialize(payload)
assert isinstance(msg, DistanceMatrix)
```

## 9. Contract Evolution Policy

Public contracts are strict and narrow.

Evolution rules:

- update producers, consumers, tests, and docs in one pass
- do not keep dual-path wire compatibility once migration is complete
- do not add convenience models that duplicate canonical payloads
- keep tensor-native performance seams internal whenever possible

## 10. Non-Negotiables

1. v2-only public wire surface
2. no new public wire models without explicit approval
3. no visualization payloads in canonical public contracts
4. immutable dataclasses for safety and performance
5. service sovereignty across package boundaries
6. internal tensor seams do not widen the public wire
