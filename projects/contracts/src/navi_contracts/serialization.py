"""Msgpack-based serialization for wire-format models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

from navi_contracts.models import (
    Action,
    ActorControlRequest,
    ActorControlResponse,
    BatchStepRequest,
    BatchStepResult,
    DistanceMatrix,
    RobotPose,
    StepRequest,
    StepResult,
    TelemetryEvent,
    _robot_pose_from_dict,
    _robot_pose_to_dict,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = [
    "deserialize",
    "serialize",
]

# --- Custom ext type codes for msgpack ---
_EXT_NDARRAY: int = 1


def _encode_ndarray(arr: NDArray[Any]) -> bytes:
    """Encode a numpy array to bytes: dtype + shape + raw data."""
    dtype_bytes = str(arr.dtype).encode("utf-8")
    shape_bytes = np.array(arr.shape, dtype=np.int64).tobytes()
    header = len(dtype_bytes).to_bytes(2, "little") + dtype_bytes
    header += len(arr.shape).to_bytes(2, "little") + shape_bytes
    return header + arr.tobytes()


def _decode_ndarray(data: bytes) -> NDArray[Any]:
    """Decode a numpy array from bytes."""
    offset = 0
    dtype_len = int.from_bytes(data[offset : offset + 2], "little")
    offset += 2
    dtype_str = data[offset : offset + dtype_len].decode("utf-8")
    offset += dtype_len
    ndim = int.from_bytes(data[offset : offset + 2], "little")
    offset += 2
    shape = tuple(np.frombuffer(data[offset : offset + ndim * 8], dtype=np.int64))
    offset += ndim * 8
    arr: NDArray[Any] = np.frombuffer(data[offset:], dtype=np.dtype(dtype_str)).reshape(shape)
    return arr


def _ext_hook(code: int, data: bytes) -> Any:
    """Msgpack ext type decoder."""
    if code == _EXT_NDARRAY:
        return _decode_ndarray(data)
    msg = f"Unknown ext type: {code}"
    raise ValueError(msg)


def _default(obj: Any) -> Any:
    """Msgpack default encoder for custom types."""
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(_EXT_NDARRAY, _encode_ndarray(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    msg = f"Object of type {type(obj)} is not serializable"
    raise TypeError(msg)


def _serialize_action(action: Action) -> dict[str, Any]:
    """Serialize an Action to a plain dict (shared helper)."""
    return {
        "env_ids": action.env_ids,
        "linear_velocity": action.linear_velocity,
        "angular_velocity": action.angular_velocity,
        "policy_id": action.policy_id,
        "step_id": action.step_id,
        "timestamp": action.timestamp,
    }


def _serialize_step_result(result: StepResult) -> dict[str, Any]:
    """Serialize a StepResult to a plain dict (shared helper)."""
    return {
        "step_id": result.step_id,
        "env_id": result.env_id,
        "episode_id": result.episode_id,
        "done": result.done,
        "truncated": result.truncated,
        "reward": result.reward,
        "episode_return": result.episode_return,
        "timestamp": result.timestamp,
    }


def _serialize_distance_matrix(dm: DistanceMatrix) -> dict[str, Any]:
    """Serialize a DistanceMatrix to a plain dict (shared helper)."""
    return {
        "episode_id": dm.episode_id,
        "env_ids": dm.env_ids,
        "matrix_shape": list(dm.matrix_shape),
        "depth": dm.depth,
        "delta_depth": dm.delta_depth,
        "semantic": dm.semantic,
        "valid_mask": dm.valid_mask,
        "overhead": dm.overhead,
        "robot_pose": _robot_pose_to_dict(dm.robot_pose),
        "step_id": dm.step_id,
        "timestamp": dm.timestamp,
    }


def serialize(
    model: (
        DistanceMatrix
        | Action
        | ActorControlRequest
        | ActorControlResponse
        | RobotPose
        | StepRequest
        | StepResult
        | TelemetryEvent
        | BatchStepRequest
        | BatchStepResult
    ),
) -> bytes:
    """Serialize a wire-format model to msgpack bytes."""
    if isinstance(model, RobotPose):
        payload: dict[str, Any] = {"_type": "RobotPose", **_robot_pose_to_dict(model)}
    elif isinstance(model, DistanceMatrix):
        payload = {
            "_type": "DistanceMatrix",
            **_serialize_distance_matrix(model),
        }
    elif isinstance(model, StepRequest):
        payload = {
            "_type": "StepRequest",
            "action": _serialize_action(model.action),
            "step_id": model.step_id,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, StepResult):
        payload = {
            "_type": "StepResult",
            **_serialize_step_result(model),
        }
    elif isinstance(model, Action):
        payload = {
            "_type": "Action",
            **_serialize_action(model),
        }
    elif isinstance(model, ActorControlRequest):
        payload = {
            "_type": "ActorControlRequest",
            "command": model.command,
            "actor_id": model.actor_id,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, ActorControlResponse):
        payload = {
            "_type": "ActorControlResponse",
            "ok": model.ok,
            "actor_id": model.actor_id,
            "actor_ids": model.actor_ids,
            "message": model.message,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, TelemetryEvent):
        payload = {
            "_type": "TelemetryEvent",
            "event_type": model.event_type,
            "episode_id": model.episode_id,
            "env_id": model.env_id,
            "step_id": model.step_id,
            "payload": model.payload,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, BatchStepRequest):
        payload = {
            "_type": "BatchStepRequest",
            "actions": [_serialize_action(a) for a in model.actions],
            "step_id": model.step_id,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, BatchStepResult):
        payload = {
            "_type": "BatchStepResult",
            "results": [_serialize_step_result(r) for r in model.results],
            "observations": [_serialize_distance_matrix(o) for o in model.observations],
        }
    else:
        msg = f"Unsupported model type: {type(model)}"
        raise TypeError(msg)

    return msgpack.packb(payload, default=_default)


def deserialize(
    data: bytes,
) -> (
    DistanceMatrix
    | Action
    | ActorControlRequest
    | ActorControlResponse
    | RobotPose
    | StepRequest
    | StepResult
    | TelemetryEvent
    | BatchStepRequest
    | BatchStepResult
):
    """Deserialize msgpack bytes to a wire-format model."""
    raw: dict[str, Any] = msgpack.unpackb(data, ext_hook=_ext_hook)
    type_tag: str = raw.pop("_type")

    if type_tag == "RobotPose":
        return RobotPose(**raw)
    if type_tag == "DistanceMatrix":
        raw["matrix_shape"] = tuple(raw["matrix_shape"])
        raw["robot_pose"] = _robot_pose_from_dict(raw["robot_pose"])
        return DistanceMatrix(**raw)
    if type_tag == "StepRequest":
        action_data = raw["action"]
        raw["action"] = Action(**action_data)
        return StepRequest(**raw)
    if type_tag == "StepResult":
        return StepResult(**raw)
    if type_tag == "Action":
        return Action(**raw)
    if type_tag == "ActorControlRequest":
        return ActorControlRequest(**raw)
    if type_tag == "ActorControlResponse":
        return ActorControlResponse(**raw)
    if type_tag == "TelemetryEvent":
        return TelemetryEvent(**raw)
    if type_tag == "BatchStepRequest":
        raw["actions"] = tuple(Action(**a) for a in raw["actions"])
        return BatchStepRequest(**raw)
    if type_tag == "BatchStepResult":
        results = tuple(StepResult(**r) for r in raw["results"])
        observations: list[DistanceMatrix] = []
        for obs_data in raw["observations"]:
            obs_data["matrix_shape"] = tuple(obs_data["matrix_shape"])
            obs_data["robot_pose"] = _robot_pose_from_dict(obs_data["robot_pose"])
            observations.append(DistanceMatrix(**obs_data))
        return BatchStepResult(results=results, observations=tuple(observations))

    msg = f"Unknown model type tag: {type_tag}"
    raise ValueError(msg)
