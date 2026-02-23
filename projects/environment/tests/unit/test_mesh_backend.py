"""Tests for MeshSceneBackend (trimesh-based raycasting)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from navi_contracts import Action, DistanceMatrix, StepResult
from navi_environment.backends.mesh_backend import MeshSceneBackend
from navi_environment.config import EnvironmentConfig

_SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "data", "scenes", "sample_apartment.glb",
)
_SCENE_EXISTS = os.path.isfile(_SCENE_PATH)

pytestmark = pytest.mark.skipif(
    not _SCENE_EXISTS,
    reason="sample_apartment.glb not found — run scripts/generate_sample_scene.py first",
)


def _make_backend(az: int = 32, el: int = 16) -> MeshSceneBackend:
    """Build a backend with small ray counts for speed."""
    cfg = EnvironmentConfig(
        backend="mesh",
        habitat_scene=_SCENE_PATH,
        azimuth_bins=az,
        elevation_bins=el,
        max_distance=15.0,
    )
    return MeshSceneBackend(cfg)


def _make_action(step_id: int = 1, fwd: float = 0.3, yaw: float = 0.1) -> Action:
    return Action(
        env_ids=np.array([0], dtype=np.int32),
        linear_velocity=np.array([[fwd, 0, 0]], dtype=np.float32),
        angular_velocity=np.array([[0, 0, yaw]], dtype=np.float32),
        policy_id="test",
        step_id=step_id,
        timestamp=time.time(),
    )


class TestMeshSceneBackend:
    """Verify MeshSceneBackend integrates correctly."""

    def test_reset_returns_distance_matrix(self) -> None:
        b = _make_backend()
        obs = b.reset(0)
        assert isinstance(obs, DistanceMatrix)
        assert obs.episode_id == 0
        assert obs.step_id == 0

    def test_observation_shapes(self) -> None:
        b = _make_backend(az=32, el=16)
        obs = b.reset(0)
        assert obs.depth.shape == (1, 32, 16)
        assert obs.delta_depth.shape == (1, 32, 16)
        assert obs.semantic.shape == (1, 32, 16)
        assert obs.valid_mask.shape == (1, 32, 16)
        assert obs.matrix_shape == (32, 16)

    def test_full_valid_mask(self) -> None:
        """Inside a closed room, all rays should hit geometry."""
        b = _make_backend(az=32, el=16)
        obs = b.reset(0)
        assert obs.valid_mask.sum() == 32 * 16

    def test_depth_normalised_zero_one(self) -> None:
        b = _make_backend(az=32, el=16)
        obs = b.reset(0)
        assert obs.depth.min() >= 0.0
        assert obs.depth.max() <= 1.0

    def test_step_returns_dm_and_result(self) -> None:
        b = _make_backend()
        b.reset(0)
        obs, result = b.step(_make_action(), 1)
        assert isinstance(obs, DistanceMatrix)
        assert isinstance(result, StepResult)
        assert result.step_id == 1

    def test_agent_moves_on_step(self) -> None:
        b = _make_backend()
        obs0 = b.reset(0)
        x0, z0 = obs0.robot_pose.x, obs0.robot_pose.z

        obs1, _ = b.step(_make_action(fwd=0.5, yaw=0.0), 1)
        x1, z1 = obs1.robot_pose.x, obs1.robot_pose.z

        dist = np.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)
        # dt=0.02 → displacement ≈ velocity × dt × (1-smoothing)
        assert dist > 1e-4, f"Agent should have moved but dist={dist:.6f}"

    def test_yaw_changes_on_step(self) -> None:
        b = _make_backend()
        b.reset(0)
        _, _ = b.step(_make_action(fwd=0.0, yaw=0.5), 1)
        # dt=0.02 → yaw change ≈ 0.5 × 0.02 × 0.7 = 0.007 rad
        assert abs(b.pose.yaw) > 1e-4

    def test_delta_depth_nonzero_after_motion(self) -> None:
        b = _make_backend()
        b.reset(0)
        obs, _ = b.step(_make_action(fwd=0.5), 1)
        # After moving, delta_depth should have some non-zero values
        assert np.any(obs.delta_depth != 0.0)

    def test_overhead_minimap_shape(self) -> None:
        cfg = EnvironmentConfig(
            backend="mesh",
            habitat_scene=_SCENE_PATH,
            azimuth_bins=32,
            elevation_bins=16,
            max_distance=15.0,
            compute_overhead=True,
        )
        b = MeshSceneBackend(cfg)
        obs = b.reset(0)
        assert obs.overhead is not None
        assert obs.overhead.shape == (256, 256, 3)

    def test_episode_reset_clears_state(self) -> None:
        b = _make_backend()
        b.reset(0)
        # Step a few times to accumulate state
        for i in range(5):
            b.step(_make_action(step_id=i + 1), i + 1)

        # Reset should return to spawn position
        obs = b.reset(1)
        assert obs.episode_id == 1
        assert obs.step_id == 0
        # Delta-depth should be zero after reset
        assert np.allclose(obs.delta_depth, 0.0)

    def test_close_releases_mesh(self) -> None:
        b = _make_backend()
        b.reset(0)
        b.close()
        assert b._mesh is None

    def test_multiple_episodes(self) -> None:
        """Run 3 short episodes to verify reset idempotency."""
        b = _make_backend()
        for ep in range(3):
            obs = b.reset(ep)
            assert obs.episode_id == ep
            for step in range(10):
                obs, result = b.step(_make_action(step_id=step + 1), step + 1)
                assert result.step_id == step + 1
                assert obs.valid_mask.sum() > 0

    def test_collision_clamping(self) -> None:
        """Agent near wall should not pass through geometry."""
        b = _make_backend()
        b.reset(0)
        # Push hard into one direction many times
        for i in range(50):
            obs, _ = b.step(_make_action(step_id=i + 1, fwd=2.0, yaw=0.0), i + 1)

        # Agent should still be inside mesh bounds
        bounds = b._mesh.bounds
        assert obs.robot_pose.x >= bounds[0, 0] - 1.0
        assert obs.robot_pose.x <= bounds[1, 0] + 1.0
        assert obs.robot_pose.z >= bounds[0, 2] - 1.0
        assert obs.robot_pose.z <= bounds[1, 2] + 1.0
