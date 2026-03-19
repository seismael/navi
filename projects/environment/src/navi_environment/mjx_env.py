"""MJX-compatible pose stepping wrapper for Ghost-Matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from navi_contracts import Action, RobotPose

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["MjxBackendInfo", "MjxEnvironment"]

# Dynamic speed scaling constants used by the canonical kinematics layer.
# Threshold is in **metres** — depth is un-normalised before comparison.
_SAFE_DEPTH_THRESHOLD: float = 1.5
_MIN_SPEED_FACTOR: float = 0.05


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

    def __init__(
        self,
        dt: float = 0.02,
        *,
        speed_scales: tuple[float, float, float, float] = (10.0, 3.0, 5.0, 3.0),
    ) -> None:
        self._dt = dt
        # Drone speed scales: (fwd m/s, vert m/s, lat m/s, yaw rad/s)
        self._speed_fwd = speed_scales[0]
        self._speed_vert = speed_scales[1]
        self._speed_lat = speed_scales[2]
        self._speed_yaw = speed_scales[3]
        self._prev_linear = np.zeros(3, dtype=np.float32)
        self._prev_angular = np.zeros(3, dtype=np.float32)
        self._smoothing: float = 0.3  # momentum factor: 0=instant, 1=no change
        self._backend_info = self._probe_backend()

    @property
    def backend(self) -> MjxBackendInfo:
        """Return detected backend details."""
        return self._backend_info

    @property
    def dt(self) -> float:
        """Return the physics timestep (seconds)."""
        return self._dt

    def set_smoothing(self, alpha: float) -> None:
        """Set the velocity smoothing factor.

        Args:
            alpha: Momentum factor in [0, 1].  0 = instant response
                   (no smoothing), values near 1 = heavy damping.
                   Default is 0.3 which simulates light drone inertia.
        """
        self._smoothing = max(0.0, min(1.0, alpha))

    def reset_velocity(self) -> None:
        """Zero the internal velocity state (call on episode reset)."""
        self._prev_linear[:] = 0.0
        self._prev_angular[:] = 0.0

    def step_pose(
        self,
        pose: RobotPose,
        action: Action,
        timestamp: float,
        *,
        prev_depth: NDArray[np.float32] | Any | None = None,
        max_distance: float = 30.0,
    ) -> RobotPose:
        """Step the robot pose by one simulation tick.

        The action carries **normalised steering** in [-1, 1].
        This method scales by ``speed_scales`` to get velocity
        (m/s / rad/s), applies momentum smoothing, rotates to
        world frame, and integrates by ``dt``.

        If *prev_depth* is provided (normalised, shape ``(Az, El)``),
        translational speeds are scaled by a proximity factor derived
        from the front hemisphere.  Yaw is never throttled.

        A first-order exponential smoothing filter simulates
        drone inertia::

            actual = (1 - alpha) * commanded + alpha * previous

        where ``alpha`` is the smoothing factor (default 0.3).
        """
        raw_lin = (
            action.linear_velocity[0]
            if action.linear_velocity.ndim == 2
            else action.linear_velocity
        )
        raw_ang = (
            action.angular_velocity[0]
            if action.angular_velocity.ndim == 2
            else action.angular_velocity
        )

        return self.step_pose_commands(
            pose,
            raw_lin,
            raw_ang,
            timestamp,
            prev_depth=prev_depth,
            max_distance=max_distance,
        )

    def step_pose_commands(
        self,
        pose: RobotPose,
        linear_velocity: NDArray[np.float32],
        angular_velocity: NDArray[np.float32],
        timestamp: float,
        *,
        prev_depth: NDArray[np.float32] | Any | None = None,
        max_distance: float = 30.0,
    ) -> RobotPose:
        """Step the robot pose from normalized command vectors directly."""
        raw_lin = np.asarray(linear_velocity, dtype=np.float32).reshape(-1)
        raw_ang = np.asarray(angular_velocity, dtype=np.float32).reshape(-1)

        # Scale normalised steering → physical velocity
        speed_factor = self._compute_speed_factor(prev_depth, max_distance)
        linear = np.array(
            [
                float(raw_lin[0]) * self._speed_fwd * speed_factor,
                float(raw_lin[1]) * self._speed_vert * speed_factor if len(raw_lin) > 1 else 0.0,
                float(raw_lin[2]) * self._speed_lat * speed_factor if len(raw_lin) > 2 else 0.0,
            ],
            dtype=np.float32,
        )
        angular = np.array(
            [
                float(raw_ang[0]),
                float(raw_ang[1]) if len(raw_ang) > 1 else 0.0,
                float(raw_ang[2]) * self._speed_yaw if len(raw_ang) > 2 else 0.0,  # yaw unscaled
            ],
            dtype=np.float32,
        )

        # Apply first-order exponential smoothing (momentum)
        a = self._smoothing
        smooth_lin = (1.0 - a) * linear + a * self._prev_linear
        smooth_ang = (1.0 - a) * angular + a * self._prev_angular
        self._prev_linear[:] = smooth_lin
        self._prev_angular[:] = smooth_ang

        # Rotate body-frame XZ velocity into world frame using yaw
        cos_yaw = float(np.cos(pose.yaw))
        sin_yaw = float(np.sin(pose.yaw))
        fwd = float(smooth_lin[0])
        lat = float(smooth_lin[2])
        dx = fwd * cos_yaw - lat * sin_yaw
        dz = fwd * sin_yaw + lat * cos_yaw

        return RobotPose(
            x=pose.x + dx * self._dt,
            y=pose.y + float(smooth_lin[1]) * self._dt,
            z=pose.z + dz * self._dt,
            roll=pose.roll + float(smooth_ang[0]) * self._dt,
            pitch=pose.pitch + float(smooth_ang[1]) * self._dt,
            yaw=pose.yaw + float(smooth_ang[2]) * self._dt,
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

    @staticmethod
    def _compute_speed_factor(
        prev_depth: NDArray[np.float32] | Any | None,
        max_distance: float = 30.0,
    ) -> float:
        """Proximity-based speed factor from front hemisphere depth.

        The speed factor is derived from the most recent front-facing depth.

        Args:
            prev_depth: (Az, El) normalised depth from the last step,
                or ``None`` on the first step of an episode. The canonical
                tensor-native environment path may provide a tensor here.
            max_distance: Maximum ray distance (meters) used to
                un-normalise the depth readings.
        """
        if prev_depth is None:
            return _MIN_SPEED_FACTOR
        az_bins = int(prev_depth.shape[0])
        span = max(1, az_bins // 8)
        min_front = min(
            MjxEnvironment._min_depth_value(prev_depth[:span]),
            MjxEnvironment._min_depth_value(prev_depth[-span:]),
        )
        # Un-normalise to physical metres before comparing
        min_front_metres = min_front * max_distance
        factor = min_front_metres / _SAFE_DEPTH_THRESHOLD
        return float(np.clip(factor, _MIN_SPEED_FACTOR, 1.0))

    @staticmethod
    def _min_depth_value(depth_slice: NDArray[np.float32] | Any) -> float:
        """Return the minimum depth from either a numpy array or tensor slice."""
        if isinstance(depth_slice, np.ndarray):
            return float(np.min(depth_slice)) if depth_slice.size > 0 else 1.0
        if hasattr(depth_slice, "numel") and int(depth_slice.numel()) == 0:
            return 1.0
        if hasattr(depth_slice, "amin"):
            min_value = depth_slice.amin()
            if hasattr(min_value, "item"):
                return float(min_value.item())
            return float(min_value)
        array_view = np.asarray(depth_slice, dtype=np.float32)
        return float(np.min(array_view)) if array_view.size > 0 else 1.0
