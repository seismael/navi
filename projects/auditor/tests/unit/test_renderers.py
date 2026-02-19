"""Tests for the extracted rendering functions."""

from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.renderers import (
    compute_nav_metrics,
    depth_to_viridis,
    distance_color,
    render_bev_occupancy,
    render_first_person,
    render_forward_polar,
    zoom_overhead,
)


class TestDepthToViridis:
    """Tests for Viridis colourmap conversion."""

    def test_output_shape_matches_input(self) -> None:
        depth = np.random.rand(16, 8).astype(np.float32)
        valid = np.ones((16, 8), dtype=bool)
        result = depth_to_viridis(depth, valid)
        assert result.shape == (16, 8, 3)
        assert result.dtype == np.uint8

    def test_invalid_regions_get_fog_of_war(self) -> None:
        depth = np.ones((16, 8), dtype=np.float32) * 0.5
        valid = np.zeros((16, 8), dtype=bool)
        result = depth_to_viridis(depth, valid)
        assert result.shape == (16, 8, 3)

    def test_no_crash_all_valid(self) -> None:
        depth = np.linspace(0.0, 1.0, 128).reshape(16, 8).astype(np.float32)
        valid = np.ones((16, 8), dtype=bool)
        result = depth_to_viridis(depth, valid)
        assert result.shape == (16, 8, 3)


class TestRenderFirstPerson:
    """Tests for first-person 3D projection."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(32, 16).astype(np.float32)
        semantic = np.ones((32, 16), dtype=np.int32)
        valid = np.ones((32, 16), dtype=bool)
        img, dist = render_first_person(depth, semantic, valid, 320, 240)
        assert img.shape == (240, 320, 3)
        assert isinstance(dist, float)

    def test_all_invalid_returns_background(self) -> None:
        depth = np.zeros((32, 16), dtype=np.float32)
        semantic = np.zeros((32, 16), dtype=np.int32)
        valid = np.zeros((32, 16), dtype=bool)
        img, dist = render_first_person(depth, semantic, valid, 320, 240)
        assert img.shape == (240, 320, 3)
        assert dist >= 9.0  # Should be VIEW_RANGE_M


class TestRenderForwardPolar:
    """Tests for polar scan rendering."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(32, 16).astype(np.float32)
        valid = np.ones((32, 16), dtype=bool)
        result = render_forward_polar(depth, valid, 200, 200)
        assert result.shape == (200, 200, 3)


class TestRenderBevOccupancy:
    """Tests for bird's-eye view rendering."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(64, 16).astype(np.float32)
        valid = np.ones((64, 16), dtype=bool)
        result = render_bev_occupancy(depth, valid, 200, 200)
        assert result.shape == (200, 200, 3)


class TestComputeNavMetrics:
    """Tests for navigation sector metrics."""

    def test_returns_three_floats(self) -> None:
        depth = np.random.rand(32, 16).astype(np.float32)
        valid = np.ones((32, 16), dtype=bool)
        fwd, left, right = compute_nav_metrics(depth, valid)
        assert 0.0 <= fwd <= 1.0
        assert 0.0 <= left <= 1.0
        assert 0.0 <= right <= 1.0

    def test_empty_input_returns_defaults(self) -> None:
        depth = np.zeros((0, 0), dtype=np.float32)
        valid = np.zeros((0, 0), dtype=bool)
        fwd, left, right = compute_nav_metrics(depth, valid)
        assert fwd == 1.0


class TestDistanceColor:
    """Tests for distance-to-colour mapping."""

    def test_close_is_red(self) -> None:
        r, g, b = distance_color(0.3)
        assert r == 0
        assert g == 0
        assert b == 255  # BGR red

    def test_far_is_grey(self) -> None:
        r, g, b = distance_color(15.0)
        assert r == 50
        assert g == 50
        assert b == 50


class TestZoomOverhead:
    """Tests for overhead zoom."""

    def test_zoom_preserves_shape(self) -> None:
        overhead = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = zoom_overhead(overhead, 2.0)
        assert result.shape == (100, 100, 3)

    def test_zoom_1x_no_change(self) -> None:
        overhead = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = zoom_overhead(overhead, 1.0)
        assert result.shape == (100, 100, 3)
