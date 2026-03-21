"""Phase 6 — Yaw Rotation Oracle Tests.

Validates that the Y-axis rotation applied to local ray directions
produces mathematically correct world-space directions, using the
same rotation matrix as ``_cast_actor_batch_tensors``.

No CUDA needed — uses pure torch CPU tensors.

Rotation matrix (Y-up, yaw around Y):

    ┌ cos(yaw)   0   sin(yaw) ┐
    │    0        1      0     │
    └ -sin(yaw)  0   cos(yaw) ┘
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _rotate_yaw(local_dirs: torch.Tensor, yaw: float) -> torch.Tensor:
    """Apply the same Y-axis rotation as sdfdag_backend._cast_actor_batch_tensors.

    local_dirs: [N, 3]  yaw: scalar radians
    Returns: [N, 3] world dirs
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    world = torch.zeros_like(local_dirs)
    world[:, 0] = local_dirs[:, 0] * cos_y + local_dirs[:, 2] * sin_y
    world[:, 1] = local_dirs[:, 1]
    world[:, 2] = -local_dirs[:, 0] * sin_y + local_dirs[:, 2] * cos_y
    return world


# ── Test Directions ──────────────────────────────────────────────────

FORWARD = torch.tensor([[0.0, 0.0, -1.0]])
RIGHT   = torch.tensor([[1.0, 0.0, 0.0]])
UP      = torch.tensor([[0.0, 1.0, 0.0]])
MULTI   = torch.tensor([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


class TestIdentityRotation:
    """yaw=0 → world == local."""

    def test_forward_unchanged(self) -> None:
        world = _rotate_yaw(FORWARD, 0.0)
        np.testing.assert_allclose(world.numpy(), FORWARD.numpy(), atol=1e-7)

    def test_multi_unchanged(self) -> None:
        world = _rotate_yaw(MULTI, 0.0)
        np.testing.assert_allclose(world.numpy(), MULTI.numpy(), atol=1e-7)


class TestNinetyDegreeRotation:
    """yaw=π/2 → forward [0,0,-1] becomes [-1,0,0] (rotated left in world)."""

    def test_forward_rotates_to_negative_x(self) -> None:
        world = _rotate_yaw(FORWARD, math.pi / 2)
        np.testing.assert_allclose(world.numpy(), [[-1.0, 0.0, 0.0]], atol=1e-6)

    def test_right_rotates_to_negative_z(self) -> None:
        # [1,0,0] at yaw=π/2 → [cos(π/2)·1 + sin(π/2)·0, 0, -sin(π/2)·1 + cos(π/2)·0]
        # = [0, 0, -1]
        world = _rotate_yaw(RIGHT, math.pi / 2)
        np.testing.assert_allclose(world.numpy(), [[0.0, 0.0, -1.0]], atol=1e-6)


class TestOneEightyRotation:
    """yaw=π → forward [0,0,-1] becomes [0,0,1] (reversed)."""

    def test_forward_reverses(self) -> None:
        world = _rotate_yaw(FORWARD, math.pi)
        np.testing.assert_allclose(world.numpy(), [[0.0, 0.0, 1.0]], atol=1e-6)


class TestArbitraryRotation:
    """yaw=π/6 (30°) — manual rotation matrix applied."""

    def test_three_directions(self) -> None:
        yaw = math.pi / 6
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        for i in range(MULTI.shape[0]):
            local = MULTI[i].numpy()
            expected = rot @ local
            actual = _rotate_yaw(MULTI[i : i + 1], yaw).numpy()[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5, err_msg=f"dir {i}")


class TestYComponentPreserved:
    """For any yaw, the Y component of every direction must be unchanged."""

    @pytest.mark.parametrize("yaw", [0.0, math.pi / 6, math.pi / 2, math.pi, 3 * math.pi / 2])
    def test_y_preserved(self, yaw: float) -> None:
        world = _rotate_yaw(MULTI, yaw)
        np.testing.assert_allclose(
            world[:, 1].numpy(),
            MULTI[:, 1].numpy(),
            atol=1e-7,
            err_msg=f"Y-component changed at yaw={yaw:.4f}",
        )
