"""Msgpack-based serialization for wire-format models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgpack
import numpy as np

from navi_contracts.models import (
    Action,
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
    msg = f"Object of type {type(obj)} is not serializable"
    raise TypeError(msg)


def serialize(
    model: DistanceMatrix | Action | RobotPose | StepRequest | StepResult | TelemetryEvent,
) -> bytes:
    """Serialize a wire-format model to msgpack bytes."""
    if isinstance(model, RobotPose):
        payload: dict[str, Any] = {"_type": "RobotPose", **_robot_pose_to_dict(model)}
    elif isinstance(model, DistanceMatrix):
        payload = {
            "_type": "DistanceMatrix",
            "episode_id": model.episode_id,
            "env_ids": model.env_ids,
            "matrix_shape": list(model.matrix_shape),
            "depth": model.depth,
            "delta_depth": model.delta_depth,
            "semantic": model.semantic,
            "valid_mask": model.valid_mask,
            "overhead": model.overhead,
            "robot_pose": _robot_pose_to_dict(model.robot_pose),
            "step_id": model.step_id,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, StepRequest):
        payload = {
            "_type": "StepRequest",
            "action": {
                "env_ids": model.action.env_ids,
                "linear_velocity": model.action.linear_velocity,
                "angular_velocity": model.action.angular_velocity,
                "policy_id": model.action.policy_id,
                "step_id": model.action.step_id,
                "timestamp": model.action.timestamp,
            },
            "step_id": model.step_id,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, StepResult):
        payload = {
            "_type": "StepResult",
            "step_id": model.step_id,
            "done": model.done,
            "truncated": model.truncated,
            "reward": model.reward,
            "episode_return": model.episode_return,
            "timestamp": model.timestamp,
        }
    elif isinstance(model, Action):
        payload = {
            "_type": "Action",
            "env_ids": model.env_ids,
            "linear_velocity": model.linear_velocity,
            "angular_velocity": model.angular_velocity,
            "policy_id": model.policy_id,
            "step_id": model.step_id,
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
    else:
        msg = f"Unsupported model type: {type(model)}"
        raise TypeError(msg)

    return msgpack.packb(payload, default=_default)


def deserialize(
    data: bytes,
) -> DistanceMatrix | Action | RobotPose | StepRequest | StepResult | TelemetryEvent:
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
    if type_tag == "TelemetryEvent":
        return TelemetryEvent(**raw)

    msg = f"Unknown model type tag: {type_tag}"
    raise ValueError(msg)
