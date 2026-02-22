"""Tests for Section Manager collision standoff barrier via VoxelBackend."""

from __future__ import annotations

import time

import numpy as np

from navi_contracts import Action
from navi_section_manager.backends.voxel import VoxelBackend
from navi_section_manager.config import SectionManagerConfig
from navi_section_manager.generators.base import AbstractWorldGenerator


class _SingleObstacleGenerator(AbstractWorldGenerator):
    """Generator with one obstacle voxel in the origin chunk."""

    def __init__(self, chunk_size: int = 8) -> None:
        self._chunk_size = chunk_size

    def generate_chunk(self, cx: int, cy: int, cz: int) -> np.ndarray:
        chunk = np.zeros(
            (self._chunk_size, self._chunk_size, self._chunk_size, 2), dtype=np.float32
        )
        if (cx, cy, cz) == (0, 0, 0):
            chunk[4, 2, 2, 0] = 1.0
            chunk[4, 2, 2, 1] = 1.0
        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        return (2.0, 2.0, 2.0)


class TestCollisionBarrier:
    """Validate standoff distance enforcement around occupied points."""

    def test_barrier_prevents_intersection(self) -> None:
        config = SectionManagerConfig(
            mode="step",
            chunk_size=8,
            window_radius=1,
            barrier_distance=0.75,
            collision_probe_radius=2.0,
        )
        generator = _SingleObstacleGenerator(chunk_size=config.chunk_size)
        backend = VoxelBackend(config, generator)
        backend.reset(episode_id=0)

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[2.5, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            policy_id="test-policy",
            step_id=1,
            timestamp=time.time(),
        )
        backend.step(action, step_id=1)

        obstacle = np.array([4.0, 2.0, 2.0], dtype=np.float32)
        pose = np.array([backend.pose.x, backend.pose.y, backend.pose.z], dtype=np.float32)
        distance = float(np.linalg.norm(pose - obstacle))
        assert distance >= config.barrier_distance - 1e-6

    def test_free_motion_when_no_near_obstacle(self) -> None:
        config = SectionManagerConfig(
            mode="step",
            chunk_size=8,
            window_radius=1,
            barrier_distance=0.75,
            collision_probe_radius=2.0,
        )
        generator = _SingleObstacleGenerator(chunk_size=config.chunk_size)
        backend = VoxelBackend(config, generator)
        backend.reset(episode_id=0)

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            policy_id="test-policy",
            step_id=2,
            timestamp=time.time(),
        )
        backend.step(action, step_id=2)

        # With dt=0.02 displacement per step is ~0.014 (velocity * dt * smoothing)
        assert backend.pose.z > 2.0 + 1e-4
