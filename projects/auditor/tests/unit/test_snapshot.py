"""Tests for the F12 diagnostic snapshot feature."""

from __future__ import annotations

import json

import numpy as np
import pytest

from navi_auditor.dashboard.app import _build_actor_state_dump
from navi_auditor.stream_engine import StreamState
from navi_contracts import DistanceMatrix, RobotPose


def _make_distance_matrix(az: int = 256, el: int = 48) -> DistanceMatrix:
    """Create a synthetic DistanceMatrix for testing."""
    depth = np.random.rand(1, az, el).astype(np.float32)
    return DistanceMatrix(
        episode_id=42,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(az, el),
        depth=depth,
        delta_depth=np.zeros_like(depth),
        semantic=np.zeros((1, az, el), dtype=np.int32),
        valid_mask=np.ones((1, az, el), dtype=bool),
        overhead=np.zeros((256, 256, 3), dtype=np.float32),
        robot_pose=RobotPose(
            x=1.0, y=2.0, z=3.0,
            roll=0.0, pitch=0.1, yaw=1.5,
            timestamp=1000.0,
        ),
        step_id=999,
        timestamp=1000.0,
    )


class TestBuildActorStateDump:
    """Tests for _build_actor_state_dump helper."""

    def test_returns_complete_dict(self) -> None:
        dm = _make_distance_matrix()
        state = StreamState()
        state.latest_matrix = dm
        state.current_scene_name = "test_scene"
        state.reward_history.extend([1.0, 2.0, 3.0])

        raw_depth = np.asarray(dm.depth[0], dtype=np.float32)
        raw_valid = np.asarray(dm.valid_mask[0], dtype=bool)

        dump = _build_actor_state_dump(state, dm, raw_depth, raw_valid)

        assert "capture_utc" in dump
        assert dump["scene_name"] == "test_scene"
        assert "depth_statistics" in dump
        assert "histories" in dump
        assert "ppo_histories" in dump
        assert "perf_histories" in dump

    def test_depth_statistics_populated(self) -> None:
        dm = _make_distance_matrix()
        state = StreamState()
        raw_depth = np.asarray(dm.depth[0], dtype=np.float32)
        raw_valid = np.asarray(dm.valid_mask[0], dtype=bool)

        dump = _build_actor_state_dump(state, dm, raw_depth, raw_valid)
        stats = dump["depth_statistics"]
        assert isinstance(stats, dict)
        assert stats["valid_ratio"] == pytest.approx(1.0)
        assert stats["valid_bins"] == 256 * 48
        assert stats["valid_min"] is not None
        assert stats["valid_max"] is not None

    def test_json_serialisable(self) -> None:
        dm = _make_distance_matrix()
        state = StreamState()
        state.reward_history.extend([0.5, 1.5])
        state.perf_sps_history.extend([200.0, 250.0])
        raw_depth = np.asarray(dm.depth[0], dtype=np.float32)
        raw_valid = np.asarray(dm.valid_mask[0], dtype=bool)

        dump = _build_actor_state_dump(state, dm, raw_depth, raw_valid)
        text = json.dumps(dump, indent=2)
        loaded = json.loads(text)
        assert loaded["histories"]["reward"] == [0.5, 1.5]
        assert loaded["perf_histories"]["sps"] == [200.0, 250.0]


class TestImagePanelCache:
    """Tests for ImagePanel BGR caching."""

    def test_cached_image_roundtrip(self) -> None:
        from navi_auditor.dashboard.panels import ImagePanel

        panel = ImagePanel()
        bgr = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        panel.set_image(bgr)

        cached = panel.get_cached_image()
        assert cached is not None
        np.testing.assert_array_equal(cached, bgr)
        assert cached is not bgr  # must be a copy

    def test_cached_image_none_initially(self) -> None:
        from navi_auditor.dashboard.panels import ImagePanel

        panel = ImagePanel()
        assert panel.get_cached_image() is None
