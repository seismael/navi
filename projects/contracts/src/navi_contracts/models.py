"""Wire-format data models for Ghost-Matrix inter-service communication."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__: list[str] = [
    "Action",
    "BatchStepRequest",
    "BatchStepResult",
    "DistanceMatrix",
    "RobotPose",
    "StepRequest",
    "StepResult",
    "TelemetryEvent",
]


@dataclass(frozen=True, slots=True)
class RobotPose:
    """6-DOF robot pose with timestamp."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class DistanceMatrix:
    """Canonical observation contract for Ghost-Matrix training and inference."""

    episode_id: int
    env_ids: NDArray[np.int32]
    matrix_shape: tuple[int, int]
    depth: NDArray[np.float32]
    delta_depth: NDArray[np.float32]
    semantic: NDArray[np.int32]
    valid_mask: NDArray[np.bool_]
    overhead: NDArray[np.float32]  # (H, W, 3) BGR minimap
    robot_pose: RobotPose
    step_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class Action:
    """Movement command produced by the Brain policy."""

    env_ids: NDArray[np.int32]
    linear_velocity: NDArray[np.float32]  # (batch, 3) — x, y, z
    angular_velocity: NDArray[np.float32]  # (batch, 3) — roll, pitch, yaw rates
    policy_id: str
    step_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class StepRequest:
    """Discrete step request from Brain to Simulation Layer."""

    action: Action
    step_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class StepResult:
    """Step acknowledgement from Simulation Layer to Brain."""

    step_id: int
    env_id: int
    done: bool
    truncated: bool
    reward: float
    episode_return: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class BatchStepRequest:
    """Batched step request: all actor actions in a single REQ message.

    Replaces N sequential ``StepRequest`` round-trips with one batched
    request containing actions for every actor.
    """

    actions: tuple[Action, ...]
    step_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    """Batched step result: all actor observations + results in one REP reply.

    Returned by the environment server in response to a ``BatchStepRequest``.
    Contains one ``StepResult`` and one ``DistanceMatrix`` per actor, in the
    same order as the actions in the request.
    """

    results: tuple[StepResult, ...]
    observations: tuple[DistanceMatrix, ...]


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Asynchronous telemetry event for logging and replay."""

    event_type: str
    episode_id: int
    env_id: int
    step_id: int
    payload: NDArray[np.float32]
    timestamp: float


def _robot_pose_to_dict(pose: RobotPose) -> dict[str, float]:
    """Convert a RobotPose to a plain dict for serialization."""
    return dataclasses.asdict(pose)  # type: ignore[return-value]


def _robot_pose_from_dict(data: dict[str, float]) -> RobotPose:
    """Reconstruct a RobotPose from a plain dict."""
    return RobotPose(**data)
