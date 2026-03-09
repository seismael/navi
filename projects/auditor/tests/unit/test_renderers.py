"""Tests for the extracted rendering functions."""

from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.renderers import (
    add_orientation_guides,
    compute_nav_metrics,
    depth_to_viridis,
    distance_color,
    overlay_overhead_annotations,
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
    """Tests for first-person dense projection."""

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

    def test_dense_coverage_with_valid_data(self) -> None:
        """Most pixels should be filled with non-background colours."""
        depth = np.random.rand(85, 128).astype(np.float32) * 0.5 + 0.1
        semantic = np.ones((85, 128), dtype=np.int32)  # WALL
        valid = np.ones((85, 128), dtype=bool)
        img, _dist = render_first_person(depth, semantic, valid, 320, 240)
        # With full valid data, very few pixels should remain pure background
        non_black = np.any(img > 20, axis=-1)
        coverage = float(np.mean(non_black))
        assert coverage > 0.7, f"Expected >70% coverage, got {coverage:.1%}"

    def test_pitch_shifts_horizon(self) -> None:
        """Non-zero pitch should produce a visibly different image."""
        depth = np.random.rand(32, 16).astype(np.float32)
        semantic = np.ones((32, 16), dtype=np.int32)
        valid = np.ones((32, 16), dtype=bool)
        img_flat, _ = render_first_person(depth, semantic, valid, 320, 240, pitch=0.0)
        img_up, _ = render_first_person(depth, semantic, valid, 320, 240, pitch=0.3)
        # Images must differ when pitch changes
        assert not np.array_equal(img_flat, img_up)

    def test_zero_bins_returns_background(self) -> None:
        depth = np.zeros((0, 0), dtype=np.float32)
        semantic = np.zeros((0, 0), dtype=np.int32)
        valid = np.zeros((0, 0), dtype=bool)
        img, dist = render_first_person(depth, semantic, valid, 320, 240)
        assert img.shape == (240, 320, 3)
        assert dist >= 9.0


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
        fwd, _left, _right = compute_nav_metrics(depth, valid)
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


class TestOverlayOverheadAnnotations:
    """Tests for overhead annotation overlay."""

    def test_mutates_panel_in_place(self) -> None:
        panel = np.zeros((200, 200, 3), dtype=np.uint8)
        original = panel.copy()
        poses: list[tuple[float, float, float]] = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.5),
            (2.0, 1.0, 1.0),
        ]
        overlay_overhead_annotations(panel, 1.0, poses)
        assert not np.array_equal(panel, original)

    def test_no_crash_empty_history(self) -> None:
        panel = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay_overhead_annotations(panel, 2.0, [])

    def test_no_crash_single_pose(self) -> None:
        panel = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay_overhead_annotations(panel, 1.0, [(0.0, 0.0, 0.0)])


class TestAddOrientationGuides:
    """Tests for orientation guide overlay."""

    def test_mutates_panel_in_place(self) -> None:
        panel = np.zeros((200, 300, 3), dtype=np.uint8)
        original = panel.copy()
        add_orientation_guides(panel)
        assert not np.array_equal(panel, original)

    def test_custom_labels(self) -> None:
        panel = np.zeros((200, 300, 3), dtype=np.uint8)
        add_orientation_guides(panel, left_label="L", right_label="R")
        # Just ensure no crash
        assert panel.shape == (200, 300, 3)
