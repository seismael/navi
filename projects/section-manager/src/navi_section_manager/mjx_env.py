"""MJX-compatible pose stepping wrapper for Ghost-Matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from navi_contracts import Action, RobotPose

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["MjxBackendInfo", "MjxEnvironment"]


@dataclass(frozen=True)
class MjxBackendInfo:
    """Runtime backend details for simulation stepping."""

    has_jax: bool
    has_mujoco: bool
    backend_name: str


class MjxEnvironment:
    """Environment adapter that exposes MJX-aware stepping semantics.

    Current implementation provides deterministic kinematic stepping while
    validating JAX/MuJoCo availability for runtime capability reporting.
    """

    def __init__(self, dt: float = 1.0) -> None:
        self._dt = dt
        self._backend_info = self._probe_backend()

    @property
    def backend(self) -> MjxBackendInfo:
        """Return detected backend details."""
        return self._backend_info

    def step_pose(self, pose: RobotPose, action: Action, timestamp: float) -> RobotPose:
        """Step the robot pose by one simulation tick.

        Linear velocity is applied in *body frame*: ``linear[0]`` is
        forward along the robot's heading (yaw), ``linear[2]`` is
        lateral.  The body-frame vector is rotated by ``pose.yaw``
        before being added to the world-frame position.
        """
        linear: NDArray[np.float32]
        angular: NDArray[np.float32]

        linear = (
            action.linear_velocity[0]
            if action.linear_velocity.ndim == 2
            else action.linear_velocity
        )
        angular = (
            action.angular_velocity[0]
            if action.angular_velocity.ndim == 2
            else action.angular_velocity
        )

        # Rotate body-frame XZ velocity into world frame using yaw
        cos_yaw = float(np.cos(pose.yaw))
        sin_yaw = float(np.sin(pose.yaw))
        fwd = float(linear[0])
        lat = float(linear[2])
        dx = fwd * cos_yaw - lat * sin_yaw
        dz = fwd * sin_yaw + lat * cos_yaw

        return RobotPose(
            x=pose.x + dx * self._dt,
            y=pose.y + float(linear[1]) * self._dt,
            z=pose.z + dz * self._dt,
            roll=pose.roll + float(angular[0]) * self._dt,
            pitch=pose.pitch + float(angular[1]) * self._dt,
            yaw=pose.yaw + float(angular[2]) * self._dt,
            timestamp=timestamp,
        )

    def _probe_backend(self) -> MjxBackendInfo:
        has_jax = False
        has_mujoco = False
        backend_name = "numpy"

        try:
            import_module("jax")
            has_jax = True
            backend_name = "jax"
        except ModuleNotFoundError:
            has_jax = False

        try:
            import_module("mujoco")
            has_mujoco = True
            if has_jax:
                backend_name = "mjx"
        except ModuleNotFoundError:
            has_mujoco = False

        return MjxBackendInfo(has_jax=has_jax, has_mujoco=has_mujoco, backend_name=backend_name)
