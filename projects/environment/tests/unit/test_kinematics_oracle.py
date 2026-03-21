"""Phase 8 — Kinematics Oracle Tests.

Validates the ``_step_kinematics_tensor`` function against analytically
predictable trajectories.

No CUDA needed — CPU tensors only.
"""

from __future__ import annotations

import numpy as np
import torch


def _step_kinematics_tensor(
    previous_depths: torch.Tensor,
    actions_linear: torch.Tensor,
    actions_angular: torch.Tensor,
    previous_linear: torch.Tensor,
    previous_angular: torch.Tensor,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    *,
    max_distance: float,
    speed_fwd: float,
    speed_vert: float,
    speed_lat: float,
    speed_yaw: float,
    smoothing: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-implementation of the backend kinematics for isolated CPU testing.

    Matches sdfdag_backend._step_kinematics_tensor exactly.
    """
    actor_count = int(previous_depths.shape[0])
    span = max(1, int(previous_depths.shape[1]) // 8)
    front_left = previous_depths[:, :span, :].reshape(actor_count, -1).amin(dim=1)
    front_right = previous_depths[:, -span:, :].reshape(actor_count, -1).amin(dim=1)
    min_front = torch.minimum(front_left, front_right)
    speed_factors = (min_front * max_distance / 1.5).clamp(min=0.05, max=1.0).unsqueeze(1)

    linear_cmd = actions_linear.clone()
    linear_cmd[:, 0] *= speed_fwd
    linear_cmd[:, 1] *= speed_vert
    linear_cmd[:, 2] *= speed_lat
    linear_cmd *= speed_factors

    angular_cmd = actions_angular.clone()
    angular_cmd[:, 2] *= speed_yaw

    smooth_linear = (1.0 - smoothing) * linear_cmd + smoothing * previous_linear
    smooth_angular = (1.0 - smoothing) * angular_cmd + smoothing * previous_angular

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    fwd = smooth_linear[:, 0]
    lat = smooth_linear[:, 2]
    dx = fwd * cos_yaw - lat * sin_yaw
    dz = fwd * sin_yaw + lat * cos_yaw

    updated_positions = positions.clone()
    updated_positions[:, 0] += dx * dt
    updated_positions[:, 1] += smooth_linear[:, 1] * dt
    updated_positions[:, 2] += dz * dt
    updated_yaws = yaws + (smooth_angular[:, 2] * dt)
    return updated_positions, updated_yaws, smooth_linear, smooth_angular


# ── Defaults ─────────────────────────────────────────────────────────

_SPEED_FWD = 1.0
_SPEED_VERT = 0.5
_SPEED_LAT = 0.5
_SPEED_YAW = 1.0
_SMOOTHING = 0.0  # No smoothing for oracle tests — makes expectations exact
_DT = 1.0 / 30.0
_MAX_DISTANCE = 30.0

AZ, EL = 8, 3  # Small dims for unit tests


def _far_depth() -> torch.Tensor:
    """Depth tensor that gives speed_factor = 1.0 (everything far away)."""
    return torch.full((1, AZ, EL), 1.0)  # normalized depth = 1.0 → real dist = max_distance → factor clamped to 1.0


def _near_depth(normalized: float = 0.02) -> torch.Tensor:
    """Depth tensor that gives low speed_factor (geometry very close)."""
    return torch.full((1, AZ, EL), normalized)


# ── Tests ────────────────────────────────────────────────────────────

class TestForwardMotionAtYawZero:
    """Pure forward translation at yaw=0.

    Kinematics convention (from sdfdag_backend._step_kinematics_tensor):
        dx = fwd * cos(yaw) - lat * sin(yaw)
        dz = fwd * sin(yaw) + lat * cos(yaw)

    At yaw=0: cos(0)=1, sin(0)=0 → dx = fwd, dz = 0.
    Forward motion at yaw=0 increases position.x.
    """

    def test_position_changes_in_x(self) -> None:
        pos = torch.tensor([[0.0, 1.0, 0.0]])
        yaw = torch.tensor([0.0])

        new_pos, _new_yaw, _, _ = _step_kinematics_tensor(
            _far_depth(),
            actions_linear=torch.tensor([[1.0, 0.0, 0.0]]),
            actions_angular=torch.tensor([[0.0, 0.0, 0.0]]),
            previous_linear=torch.zeros(1, 3),
            previous_angular=torch.zeros(1, 3),
            positions=pos,
            yaws=yaw,
            max_distance=_MAX_DISTANCE,
            speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT,
            speed_lat=_SPEED_LAT,
            speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING,
            dt=_DT,
        )
        # Expected: pos.x += fwd * speed_fwd * speed_factor * dt
        expected_dx = 1.0 * _SPEED_FWD * 1.0 * _DT
        assert abs(float(new_pos[0, 0]) - expected_dx) < 1e-6
        assert abs(float(new_pos[0, 1]) - 1.0) < 1e-6, "Y should not change"
        assert abs(float(new_pos[0, 2])) < 1e-6, "Z should not change"

    def test_yaw_unchanged(self) -> None:
        pos = torch.tensor([[0.0, 1.0, 0.0]])
        yaw = torch.tensor([0.0])
        _, new_yaw, _, _ = _step_kinematics_tensor(
            _far_depth(),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING, dt=_DT,
        )
        assert abs(float(new_yaw[0])) < 1e-7


class TestPureYawRotation:
    """Angular yaw command with no linear motion."""

    def test_yaw_changes(self) -> None:
        pos = torch.tensor([[0.0, 1.0, 0.0]])
        yaw = torch.tensor([0.0])
        _, new_yaw, _, _ = _step_kinematics_tensor(
            _far_depth(),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING, dt=_DT,
        )
        expected_yaw = _SPEED_YAW * _DT
        assert abs(float(new_yaw[0]) - expected_yaw) < 1e-6

    def test_position_unchanged_during_rotation(self) -> None:
        pos = torch.tensor([[5.0, 2.0, 3.0]])
        yaw = torch.tensor([1.0])
        new_pos, _, _, _ = _step_kinematics_tensor(
            _far_depth(),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING, dt=_DT,
        )
        np.testing.assert_allclose(new_pos.numpy(), pos.numpy(), atol=1e-6)


class TestSpeedScalingNearGeometry:
    """When depth is very low (geometry close), speed_factor < 1.0."""

    def test_speed_factor_reduces_movement(self) -> None:
        pos = torch.tensor([[0.0, 1.0, 0.0]])
        yaw = torch.tensor([0.0])

        # With far depth → full speed
        new_far, _, _, _ = _step_kinematics_tensor(
            _far_depth(),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING, dt=_DT,
        )

        # With near depth → reduced speed
        new_near, _, _, _ = _step_kinematics_tensor(
            _near_depth(0.02),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=_SMOOTHING, dt=_DT,
        )

        dx_far = float(new_far[0, 0])
        dx_near = float(new_near[0, 0])
        assert dx_near < dx_far, (
            f"Near-geometry movement ({dx_near:.6f}) should be less than far ({dx_far:.6f})"
        )
        assert dx_near > 0.0, "Movement should still be positive (clamped ≥ 0.05)"


class TestSmoothingEffect:
    """With smoothing > 0, output blends current command with previous."""

    def test_smoothing_blends(self) -> None:
        pos = torch.tensor([[0.0, 1.0, 0.0]])
        yaw = torch.tensor([0.0])
        smoothing = 0.5

        prev_lin = torch.tensor([[0.0, 0.0, 0.0]])

        _new_pos, _, smooth_lin, _ = _step_kinematics_tensor(
            _far_depth(),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            prev_lin,
            torch.zeros(1, 3),
            pos, yaw,
            max_distance=_MAX_DISTANCE, speed_fwd=_SPEED_FWD,
            speed_vert=_SPEED_VERT, speed_lat=_SPEED_LAT, speed_yaw=_SPEED_YAW,
            smoothing=smoothing, dt=_DT,
        )

        # With smoothing=0.5 and zero previous, smooth forward = half of raw
        expected_smooth_fwd = 0.5 * _SPEED_FWD * 1.0
        assert abs(float(smooth_lin[0, 0]) - expected_smooth_fwd) < 1e-6
