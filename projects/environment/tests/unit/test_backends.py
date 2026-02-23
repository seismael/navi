"""Tests for the SimulatorBackend ABC, VoxelBackend, and DatasetAdapter."""

from __future__ import annotations

import time

import numpy as np
import pytest

from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult
from navi_environment.backends.adapter import AdapterMetadata, DatasetAdapter
from navi_environment.backends.base import SimulatorBackend
from navi_environment.backends.habitat_adapter import HabitatAdapter
from navi_environment.backends.habitat_semantic_lut import HabitatSemanticLUT
from navi_environment.backends.voxel import VoxelBackend
from navi_environment.config import EnvironmentConfig
from navi_environment.generators.maze import MazeGenerator


class TestSimulatorBackendABC:
    """Verify the abstract interface contract."""

    def test_cannot_instantiate_abc(self) -> None:
        """SimulatorBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SimulatorBackend()  # type: ignore[abstract]


class TestVoxelBackend:
    """VoxelBackend wrapping procedural generators."""

    def _make_backend(self) -> VoxelBackend:
        config = EnvironmentConfig(
            chunk_size=8,
            window_radius=1,
            lookahead_margin=2,
            seed=42,
        )
        gen = MazeGenerator(seed=config.seed, chunk_size=config.chunk_size)
        return VoxelBackend(config, gen)

    def test_reset_returns_distance_matrix(self) -> None:
        backend = self._make_backend()
        obs = backend.reset(episode_id=0)
        assert isinstance(obs, DistanceMatrix)
        assert obs.episode_id == 0
        assert obs.step_id == 0

    def test_step_returns_dm_and_result(self) -> None:
        backend = self._make_backend()
        backend.reset(episode_id=0)

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test",
            step_id=1,
            timestamp=time.time(),
        )
        dm, result = backend.step(action, step_id=1)
        assert isinstance(dm, DistanceMatrix)
        assert isinstance(result, StepResult)
        assert result.step_id == 1
        assert isinstance(result.reward, float)

    def test_pose_property(self) -> None:
        backend = self._make_backend()
        backend.reset(episode_id=0)
        pose = backend.pose
        assert isinstance(pose, RobotPose)

    def test_episode_id_property(self) -> None:
        backend = self._make_backend()
        backend.reset(episode_id=42)
        assert backend.episode_id == 42

    def test_reset_shape_convention(self) -> None:
        """Depth/semantic arrays are (1, Az, El), matrix_shape is (Az, El)."""
        backend = self._make_backend()
        obs = backend.reset(episode_id=0)
        az, el = obs.matrix_shape
        assert obs.depth.shape == (1, az, el)
        assert obs.semantic.shape == (1, az, el)
        assert obs.delta_depth.shape == (1, az, el)
        assert obs.valid_mask.shape == (1, az, el)

    def test_step_shape_convention(self) -> None:
        """step() output matches the canonical (1, Az, El) shape."""
        backend = self._make_backend()
        backend.reset(episode_id=0)
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test",
            step_id=1,
            timestamp=time.time(),
        )
        dm, _ = backend.step(action, step_id=1)
        az, el = dm.matrix_shape
        assert dm.depth.shape == (1, az, el)
        assert dm.semantic.shape == (1, az, el)

    def test_close_is_safe(self) -> None:
        backend = self._make_backend()
        backend.reset(episode_id=0)
        backend.close()  # should not raise

    def test_reward_accumulation(self) -> None:
        backend = self._make_backend()
        backend.reset(episode_id=0)

        # Use small velocity to avoid kinematic collision detection
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            policy_id="test",
            step_id=0,
            timestamp=time.time(),
        )
        total = 0.0
        for i in range(3):
            _, result = backend.step(action, step_id=i)
            total += result.reward

        assert abs(result.episode_return - total) < 1e-5


class TestDatasetAdapterProtocol:
    """Verify the DatasetAdapter protocol contract."""

    def test_habitat_adapter_satisfies_protocol(self) -> None:
        """HabitatAdapter is a structural match for DatasetAdapter."""
        lut = HabitatSemanticLUT()
        adapter = HabitatAdapter(
            az_bins=64, el_bins=32, max_distance=10.0, semantic_lut=lut,
        )
        assert isinstance(adapter, DatasetAdapter)

    def test_metadata(self) -> None:
        lut = HabitatSemanticLUT()
        adapter = HabitatAdapter(
            az_bins=64, el_bins=32, max_distance=10.0, semantic_lut=lut,
        )
        meta = adapter.metadata
        assert isinstance(meta, AdapterMetadata)
        assert meta.azimuth_bins == 64
        assert meta.elevation_bins == 32
        assert meta.max_distance == 10.0
        assert meta.semantic_classes == 11


class TestHabitatAdapter:
    """Unit tests for HabitatAdapter observation conversion."""

    def _make_adapter(self, az: int = 64, el: int = 32) -> HabitatAdapter:
        lut = HabitatSemanticLUT()
        return HabitatAdapter(az_bins=az, el_bins=el, max_distance=10.0, semantic_lut=lut)

    def test_adapt_shape(self) -> None:
        """Output arrays have shape (1, Az, El) after transpose + env dim."""
        adapter = self._make_adapter(az=64, el=32)
        raw_obs = {
            "equirect_depth": np.random.rand(32, 64).astype(np.float32) * 10.0,
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        result = adapter.adapt(raw_obs, step_id=0)
        assert result["depth"].shape == (1, 64, 32)
        assert result["delta_depth"].shape == (1, 64, 32)
        assert result["semantic"].shape == (1, 64, 32)
        assert result["valid_mask"].shape == (1, 64, 32)
        assert result["overhead"].shape == (256, 256, 3)

    def test_depth_normalised(self) -> None:
        """Depth is in [0, 1] after normalisation."""
        adapter = self._make_adapter()
        raw_obs = {
            "equirect_depth": np.full((32, 64), 5.0, dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        result = adapter.adapt(raw_obs, step_id=0)
        assert result["depth"].min() >= 0.0
        assert result["depth"].max() <= 1.0
        np.testing.assert_allclose(result["depth"][0, 0, 0], 0.5, atol=1e-5)

    def test_delta_depth_first_frame(self) -> None:
        """First frame delta_depth should be all zeros."""
        adapter = self._make_adapter()
        raw_obs = {
            "equirect_depth": np.ones((32, 64), dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        result = adapter.adapt(raw_obs, step_id=0)
        np.testing.assert_array_equal(result["delta_depth"], 0.0)

    def test_delta_depth_second_frame(self) -> None:
        """Second frame produces non-zero delta_depth."""
        adapter = self._make_adapter()
        raw_obs1 = {
            "equirect_depth": np.ones((32, 64), dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        raw_obs2 = {
            "equirect_depth": np.full((32, 64), 2.0, dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        adapter.adapt(raw_obs1, step_id=0)
        result = adapter.adapt(raw_obs2, step_id=1)
        assert result["delta_depth"].sum() != 0.0

    def test_reset_clears_state(self) -> None:
        """After reset, delta_depth is zero again."""
        adapter = self._make_adapter()
        raw_obs = {
            "equirect_depth": np.ones((32, 64), dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        adapter.adapt(raw_obs, step_id=0)
        adapter.reset()
        result = adapter.adapt(raw_obs, step_id=1)
        np.testing.assert_array_equal(result["delta_depth"], 0.0)

    def test_squeeze_3d_depth(self) -> None:
        """Handles (El, Az, 1) depth from some habitat-sim versions."""
        adapter = self._make_adapter(az=64, el=32)
        raw_obs = {
            "equirect_depth": np.ones((32, 64, 1), dtype=np.float32),
            "equirect_semantic": np.zeros((32, 64), dtype=np.int32),
        }
        result = adapter.adapt(raw_obs, step_id=0)
        assert result["depth"].shape == (1, 64, 32)
