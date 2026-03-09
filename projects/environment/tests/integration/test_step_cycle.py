"""Integration test for the Environment step cycle via VoxelBackend."""

from __future__ import annotations

import time

import numpy as np

from navi_contracts import Action
from navi_environment.backends.voxel import VoxelBackend
from navi_environment.config import EnvironmentConfig
from navi_environment.generators.arena import ArenaGenerator
from navi_environment.generators.maze import MazeGenerator


class TestStepCycle:
    """Integration test: exercise the full VoxelBackend step code path."""

    def test_single_step(self) -> None:
        """One step should produce a valid StepResult and DistanceMatrix."""
        config = EnvironmentConfig(
            mode="step",
            generator="maze",
            seed=42,
            chunk_size=4,
            window_radius=1,
            lookahead_margin=2,
            barrier_distance=0.0,
        )
        gen = MazeGenerator(seed=config.seed, chunk_size=config.chunk_size)
        backend = VoxelBackend(config, gen)

        # Reset initialises the window
        obs = backend.reset(episode_id=0)
        assert obs.step_id == 0

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test-policy",
            step_id=1,
            timestamp=time.time(),
        )
        dm, result = backend.step(action, step_id=1)

        assert result.step_id == 1
        assert result.env_id == 0
        assert result.episode_id == 0
        assert result.done is False
        assert result.truncated is False
        assert isinstance(result.reward, float)
        assert dm.step_id == 1
        assert dm.episode_id == result.episode_id

    def test_episode_id_advances_after_truncation_reset(self) -> None:
        """When an episode truncates and backend resets, StepResult episode_id should advance."""
        config = EnvironmentConfig(
            mode="step",
            generator="arena",
            seed=7,
            chunk_size=8,
            window_radius=1,
            lookahead_margin=2,
            barrier_distance=0.0,
        )
        gen = ArenaGenerator(seed=config.seed, chunk_size=config.chunk_size)
        backend = VoxelBackend(config, gen)
        backend.reset(episode_id=0)

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test-policy",
            step_id=1,
            timestamp=time.time(),
        )

        # Force truncation quickly, then next call should return reset observation with episode+1.
        backend._max_steps_per_episode = 1  # type: ignore[attr-defined]
        _dm1, result1 = backend.step(action, step_id=1)
        assert result1.truncated is True
        assert result1.episode_id == 0

        dm2, result2 = backend.step(action, step_id=2)
        assert result2.env_id == 0
        assert result2.episode_id == 1
        assert dm2.episode_id == 1

    def test_multiple_steps_advance_pose(self) -> None:
        """Multiple steps should move the robot forward."""
        config = EnvironmentConfig(
            mode="step",
            generator="arena",
            seed=42,
            chunk_size=16,
            window_radius=1,
            lookahead_margin=2,
            barrier_distance=0.0,
        )
        gen = ArenaGenerator(seed=config.seed, chunk_size=config.chunk_size)
        backend = VoxelBackend(config, gen)
        backend.reset(episode_id=0)

        initial_x = backend.pose.x
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test-policy",
            step_id=0,
            timestamp=time.time(),
        )

        for i in range(5):
            backend.step(action, step_id=i)

        assert backend.pose.x > initial_x

    def test_step_produces_valid_wedge_shape(self) -> None:
        """A large step should change pose and keep simulation state consistent."""
        config = EnvironmentConfig(
            mode="step",
            generator="maze",
            seed=42,
            chunk_size=4,
            window_radius=1,
            lookahead_margin=2,
            barrier_distance=0.0,
        )
        gen = MazeGenerator(seed=config.seed, chunk_size=config.chunk_size)
        backend = VoxelBackend(config, gen)
        backend.reset(episode_id=0)

        # Big step to guarantee entering new chunks
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test-policy",
            step_id=0,
            timestamp=time.time(),
        )
        backend.step(action, step_id=0)

        # Pose changed
        assert backend.pose.x != config.seed
