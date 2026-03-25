## Spherical Observation Convention

- Canonical `DistanceMatrix` layout is `(azimuth, elevation)`.
- Environment azimuth bin `0` is the forward ray along local `-Z`.
- Dashboard and replay tools must roll the azimuth axis before center-cropping a forward FOV; otherwise the forward seam is split across the panorama edges and front-view panels appear distorted.
- Canonical training/runtime datasets come only from the downloaded corpus compiled into `artifacts/gmdag/corpus`; no generated or procedural scene source is allowed in the canonical path.

# CONTRACTS.md — Canonical Wire Format Specification

**Package:** `navi-contracts`  
**Status:** Active canonical specification  
**Policy:** See [AGENTS.md](../AGENTS.md) for implementation rules and non-negotiables

---

## 1. Canonical Models

The following canonical dataclasses are the **only** models permitted on the
inter-process wire. No additional models may be added without explicit approval.
Visualization types (RGB frames, camera images) are never part of these
contracts.

### 1.1. RobotPose

6-DOF robot pose with timestamp.

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

### 1.2. DistanceMatrix

Canonical observation contract for training and inference.

```python
@dataclass(frozen=True, slots=True)
class DistanceMatrix:
    episode_id: int
    env_ids: NDArray[int32]            # (batch,)
    matrix_shape: tuple[int, int]      # (azimuth_bins, elevation_bins)
    depth: NDArray[float32]            # (n_envs, Az, El), normalized [0, 1]
    delta_depth: NDArray[float32]      # (n_envs, Az, El), frame difference
    semantic: NDArray[int32]           # (n_envs, Az, El), class IDs [0, 10]
    valid_mask: NDArray[bool_]         # (n_envs, Az, El), ray hit validity
    overhead: NDArray[float32]         # (H, W, 3), BGR minimap
    robot_pose: RobotPose
    step_id: int
    timestamp: float
```

**Shape convention:**

- `matrix_shape[0]` = azimuth bins (rows), `matrix_shape[1]` = elevation bins
  (columns).
- Default resolution: `(256, 48)`.
- Single-env backends produce `n_envs = 1` → array shapes `(1, Az, El)`.
- `depth` is normalized to `[0, 1]` by dividing by `max_distance`.
- `delta_depth` is the per-bin change since the previous step (temporal
  velocity awareness).
- `semantic` uses integer IDs in `[0, 10]` (see
  [SIMULATION.md §3.3](SIMULATION.md) for the full semantic table).
- `valid_mask` is `True` for bins that received at least one ray hit.
- `overhead` is a 256×256 BGR minimap centered on the robot. This field is
  consumed only by the Gallery Layer — it never enters the training engine.
  It is the sole visualization-adjacent field permitted in the wire contract,
  included for diagnostic convenience. It carries no semantic training data.
- viewer layers are consumers, not contract authors: dashboard layout,
  half-sphere extraction, axis rolling for display, or palette choice must not
  change `DistanceMatrix` shape, normalization, semantic meaning, or wire fields.

### 1.3. Action

Movement command produced by the Brain policy.

```python
@dataclass(frozen=True, slots=True)
class Action:
    env_ids: NDArray[int32]            # (batch,)
    linear_velocity: NDArray[float32]  # (batch, 3) — [forward, vertical, lateral]
    angular_velocity: NDArray[float32] # (batch, 3) — [roll_rate, pitch_rate, yaw_rate]
    policy_id: str                     # identifies which policy produced this action
    step_id: int
    timestamp: float
```

**4-DOF mapping:** The actor internally produces `[fwd, vert, lat, yaw]`.
On the wire this is packed as:

- `linear_velocity[:, 0]` = forward, `[:, 1]` = vertical, `[:, 2]` = lateral.
- `angular_velocity[:, 2]` = yaw rate (roll and pitch rates are typically zero).

### 1.4. StepRequest

Discrete step request from Brain to Simulation Layer (REQ/REP).

```python
@dataclass(frozen=True, slots=True)
class StepRequest:
    action: Action
    step_id: int
    timestamp: float
```

### 1.5. StepResult

Step acknowledgement from Simulation Layer to Brain (REQ/REP).

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

### 1.6. TelemetryEvent

Asynchronous telemetry event for logging, dashboarding, and replay.

```python
@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event_type: str                    # e.g., "ppo.update", "environment.step"
    episode_id: int
    env_id: int
    step_id: int
    payload: NDArray[float32]          # generic numeric payload
    timestamp: float
```

### 1.7. BatchStepRequest

Batched step request from Brain to Simulation Layer (REQ/REP).

```python
@dataclass(frozen=True, slots=True)
class BatchStepRequest:
    actions: tuple[Action, ...]
    step_id: int
    timestamp: float
```

### 1.8. BatchStepResult

Batched step reply from Simulation Layer to Brain (REQ/REP).

```python
@dataclass(frozen=True, slots=True)
class BatchStepResult:
    results: tuple[StepResult, ...]
    observations: tuple[DistanceMatrix, ...]
```

---

## 2. Type Aliases

Defined in `navi_contracts.types` for static analysis and documentation:

| Alias | Underlying Type | Shape | Description |
|-------|----------------|-------|-------------|
| `MatrixShape` | `tuple[int, int]` | — | `(azimuth_bins, elevation_bins)` |
| `DepthMatrix` | `NDArray[float32]` | `(B, Az, El)` | Normalized distance values `[0, 1]` |
| `DeltaDepthMatrix` | `NDArray[float32]` | `(B, Az, El)` | Temporal depth deltas |
| `SemanticMatrix` | `NDArray[int32]` | `(B, Az, El)` | Semantic identifiers per cell |
| `ValidMask` | `NDArray[bool_]` | `(B, Az, El)` | True where a ray hit is valid |
| `EnvIdVector` | `NDArray[int32]` | `(B,)` | Active environment IDs |
| `VelocityMatrix` | `NDArray[float32]` | `(B, 3)` | Linear or angular velocity commands |
| `TelemetryPayload` | `NDArray[float32]` | `(N, M)` | Generic numeric telemetry payload |

---

## 3. ZMQ Topics

All inter-service communication uses versioned topic strings for PUB/SUB
routing and REQ/REP message discrimination.

| Constant | Value | Transport | Direction |
|----------|-------|-----------|-----------|
| `TOPIC_DISTANCE_MATRIX` | `distance_matrix_v2` | PUB/SUB | Simulation → Brain, Gallery |
| `TOPIC_ACTION` | `action_v2` | PUB/SUB | Brain → Simulation, Gallery |
| `TOPIC_STEP_REQUEST` | `step_request_v2` | REQ/REP | Brain → Simulation |
| `TOPIC_STEP_RESULT` | `step_result_v2` | REQ/REP | Simulation → Brain |
| `TOPIC_TELEMETRY_EVENT` | `telemetry_event_v2` | PUB/SUB | Any → Gallery |

No other topics may be added without explicit approval.

---

## 4. Internal Tensor Seams

The imported project documented low-level tensor contracts explicitly. Navi now
does the same, while keeping them clearly separate from the public wire format.

These seams are normative for performance work but are not additional wire
models.

### 4.1 `torch_sdf.cast_rays()` Boundary

Canonical raycasting tensors:

| Tensor | Shape | Dtype | Device | Notes |
| --- | --- | --- | --- | --- |
| `origins` | `[B, R, 3]` | `float32` | CUDA | contiguous |
| `dirs` | `[B, R, 3]` | `float32` | CUDA | contiguous, direction vectors |
| `out_distances` | `[B, R]` | `float32` | CUDA | preallocated output |
| `out_semantics` | `[B, R]` | `int32` | CUDA | preallocated output |

Scalar parameters:

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `sdf_max_steps` | `int` | config | max sphere-tracing iterations |
| `max_distance` | `float` | config | horizon clamp distance |
| `bbox_min` / `bbox_max` | `float` | asset | DAG world bounds |
| `resolution` | `int` | asset | DAG voxel resolution |
| `skip_direction_validation` | `bool` | `False` | bypass direction-norm validation |

Boundary rules:

- device placement, rank, and contiguity are validated before kernel launch
- direction vectors must be normalized within an explicit tolerance; exact
  floating-point equality with `1.0` is not a valid contract rule
- when `skip_direction_validation=True`, the caller guarantees normalization
  and the four GPU→CPU synchronization barriers for norm checking are skipped;
  canonical `SdfDagBackend` hot-path calls use this mode because yaw-rotated
  unit vectors are mathematically guaranteed normalized
- probe, inspection, and diagnostic calls should keep `skip_direction_validation=False`
- canonical runtime is CUDA-only; CPU fallback is not part of this seam
- long CUDA execution should release the Python GIL

### 4.2 Environment-To-Trainer Tensor Seam

The canonical trainer may consume tensor-native observations directly when the
runtime provides them.

Current canonical observation tensor:

- shape: `(B, 3, Az, El)`
- channel `0`: normalized depth
- channel `1`: semantic ids as `float32`
- channel `2`: valid mask as `float32`

This seam exists to keep the hot path on CUDA.
It does not replace the external `DistanceMatrix` contract.

Observer tools may materialize, crop, or colorize tensors after publication,
but they do not change this seam's channel meaning, normalization, or axis
ordering.

### 4.3 Materialization Rule

`DistanceMatrix` and `Action` remain the canonical service and diagnostic
contracts. On the production training hot path, they may be materialized only
when needed for:

- passive dashboard publication
- coarse telemetry
- service mode
- tests and explicit diagnostics

Rebuilding them unconditionally in the rollout loop is a performance bug.

### 4.4 Observer Proxy Rule

If browser-facing transport is introduced for dashboards or dataset QA, it is an
observer-side proxy layered on top of the canonical ZMQ surfaces.

That proxy does not widen the public wire contract and must never be used in
hot-path throughput reasoning.

---

## 5. Serialization

### 4.1. Wire Encoding

All messages are serialized via **msgpack** with custom extension types for
numpy arrays. The encoding pipeline:

1. Dataclass fields are converted to a msgpack-compatible dict.
2. numpy arrays are packed as msgpack ext types preserving dtype and shape.
3. `RobotPose` is serialized as a nested dict via `dataclasses.asdict()`.

### 4.2. ZMQ Frame Layout

Messages are sent as **multipart ZMQ frames**:

```text
Frame 0: topic_bytes    (UTF-8 encoded topic string)
Frame 1: payload_bytes  (msgpack-serialized message)
```

### 4.3. API

```python
from navi_contracts import serialize, deserialize

# Encode
payload: bytes = serialize(distance_matrix)

# Decode (auto-detects type)
msg = deserialize(payload)
assert isinstance(msg, DistanceMatrix)
```

---

## 6. Non-Negotiables

1. **v2 only.** Legacy wire contracts and topics are not permitted in new code.
2. **No new models** may be added to this package without explicit approval.
3. **No visualization types.** RGB frames, camera images, and rendered outputs
   are never canonical contracts. They belong exclusively to the Gallery Layer.
4. **Immutable dataclasses.** All models use `frozen=True, slots=True` for
   safety and performance.
5. **Service sovereignty.** No service may import another service's package.
   All integration is via serialized messages over ZMQ.
6. **Internal seams do not widen the wire.** Tensor-native training paths may
  exist internally, but they do not authorize new public wire models.
