"""Tests for the extracted rendering functions."""

from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.renderers import (
    add_orientation_guides,
    center_forward_azimuth,
    compute_nav_metrics,
    depth_to_observer_palette,
    depth_to_viridis,
    distance_color,
    extract_forward_fov,
    overlay_overhead_annotations,
    render_bev_occupancy,
    render_first_person,
    render_forward_polar,
    render_front_depth_grid,
    render_front_hemisphere_heatmap,
    zoom_overhead,
)
from navi_contracts.testing.oracle_house import house_observation


class TestDepthToViridis:
    """Tests for Viridis colourmap conversion."""

    def test_output_shape_matches_input(self) -> None:
        depth = np.random.rand(16, 8).astype(np.float32)
        valid = np.ones((16, 8), dtype=bool)
        result = depth_to_viridis(depth, valid)
        assert result.shape == (16, 8, 3)

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


class TestDepthToObserverPalette:
    """Tests for the observer-facing muted collision-to-distance palette."""

    def test_output_shape_matches_input(self) -> None:
        depth = np.random.rand(16, 8).astype(np.float32)
        valid = np.ones((16, 8), dtype=bool)
        result = depth_to_observer_palette(depth, valid)
        assert result.shape == (16, 8, 3)
        assert result.dtype == np.uint8

    def test_near_bins_are_warmer_than_far_bins(self) -> None:
        depth = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(1, 8)
        valid = np.ones((1, 8), dtype=bool)
        result = depth_to_observer_palette(depth, valid)
        near = result[0, 0].astype(np.int32)
        far = result[0, -1].astype(np.int32)

        assert near[2] > near[0]
        assert far[0] > far[2]

    def test_midrange_bins_stay_more_muted_than_collision_red(self) -> None:
        depth = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(1, 9)
        valid = np.ones((1, 9), dtype=bool)
        result = depth_to_observer_palette(depth, valid)

        near = result[0, 0].astype(np.int32)
        mid = result[0, 4].astype(np.int32)

        near_spread = int(near.max() - near.min())
        mid_spread = int(mid.max() - mid.min())
        assert mid_spread < near_spread

    def test_invalid_regions_get_fog_of_war(self) -> None:
        depth = np.ones((16, 8), dtype=np.float32) * 0.5
        valid = np.zeros((16, 8), dtype=bool)
        result = depth_to_observer_palette(depth, valid)
        assert result.shape == (16, 8, 3)
        assert result.dtype == np.uint8


class TestRenderFirstPerson:
    """Tests for the centered half-sphere heatmap actor view."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(32, 16).astype(np.float32)
        semantic = np.ones((32, 16), dtype=np.int32)
        valid = np.ones((32, 16), dtype=bool)
        img, dist = render_first_person(depth, semantic, valid, 320, 240)
        assert img.shape == (240, 320, 3)
        assert isinstance(dist, float)

    def test_all_invalid_returns_dense_far_view(self) -> None:
        depth = np.zeros((32, 16), dtype=np.float32)
        semantic = np.zeros((32, 16), dtype=np.int32)
        valid = np.zeros((32, 16), dtype=bool)
        img, dist = render_first_person(depth, semantic, valid, 320, 240)
        assert img.shape == (240, 320, 3)
        assert dist == 0.0
        assert float(np.mean(img)) > 5.0

    def test_dense_coverage_with_valid_data(self) -> None:
        """Most pixels should be filled with non-background colours."""
        depth = np.random.rand(85, 128).astype(np.float32) * 0.5 + 0.1
        semantic = np.ones((85, 128), dtype=np.int32)  # WALL
        valid = np.ones((85, 128), dtype=bool)
        img, _dist = render_first_person(depth, semantic, valid, 320, 240)
        # The padded panel contains a bordered viewport; coverage should be high inside it.
        viewport = img[24:212, 16:304]
        non_black = np.any(viewport > 20, axis=-1)
        coverage = float(np.mean(non_black))
        assert coverage > 0.9, f"Expected >90% coverage, got {coverage:.1%}"

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
        assert dist == 0.0

    def test_oracle_house_doorway_changes_center_projection_against_closed_baseline(self) -> None:
        oracle = house_observation()
        closed_valid = np.ones_like(oracle.valid)
        img, _dist = render_first_person(oracle.depth, oracle.semantic, oracle.valid, 320, 240)
        closed_img, _dist_closed = render_first_person(
            oracle.depth, oracle.semantic, closed_valid, 320, 240
        )
        center_patch = img[140:220, 130:190]
        closed_patch = closed_img[140:220, 130:190]
        assert np.mean(np.abs(center_patch.astype(np.int16) - closed_patch.astype(np.int16))) > 5.0

    def test_oracle_house_window_changes_right_upper_projection_against_closed_baseline(
        self,
    ) -> None:
        oracle = house_observation()
        closed_valid = np.ones_like(oracle.valid)
        img, _dist = render_first_person(oracle.depth, oracle.semantic, oracle.valid, 320, 240)
        closed_img, _dist_closed = render_first_person(
            oracle.depth, oracle.semantic, closed_valid, 320, 240
        )
        right_upper_patch = img[70:120, 220:285]
        closed_patch = closed_img[70:120, 220:285]
        assert (
            np.mean(np.abs(right_upper_patch.astype(np.int16) - closed_patch.astype(np.int16)))
            > 3.0
        )


class TestRenderForwardPolar:
    """Tests for polar scan rendering."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(32, 16).astype(np.float32)
        valid = np.ones((32, 16), dtype=bool)
        result = render_forward_polar(depth, valid, 200, 200)
        assert result.shape == (200, 200, 3)

    def test_centered_forward_gap_affects_top_center_arc(self) -> None:
        """A forward opening should appear near the top-center of the polar panel."""
        depth = np.full((32, 8), 0.2, dtype=np.float32)
        valid = np.ones((32, 8), dtype=bool)
        center = depth.shape[0] // 2
        depth[center - 2 : center + 2, :] = 0.8

        result = render_forward_polar(depth, valid, 240, 240)

        top_center_patch = result[120:160, 95:145]
        side_patch = result[120:190, 10:60]
        assert float(np.mean(top_center_patch)) > float(np.mean(side_patch))


class TestRenderFrontHemisphereHeatmap:
    """Tests for the front-hemisphere actor heatmap renderer."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(64, 16).astype(np.float32)
        valid = np.ones((64, 16), dtype=bool)
        result = render_front_hemisphere_heatmap(depth, valid, 320, 240)
        assert result.shape == (240, 320, 3)

    def test_renderer_uses_panel_height_for_center_column(self) -> None:
        depth = np.random.rand(64, 16).astype(np.float32)
        valid = np.ones((64, 16), dtype=bool)
        result = render_front_hemisphere_heatmap(depth, valid, 320, 240)
        center_col = result[:, 160]
        non_background = np.any(center_col != np.array([12, 12, 12], dtype=np.uint8), axis=1)
        assert int(np.count_nonzero(non_background)) > 150


class TestRenderFrontDepthGrid:
    """Tests for the exact front-half depth-grid renderer."""

    def test_output_shape(self) -> None:
        depth = np.random.rand(128, 48).astype(np.float32)
        valid = np.ones((128, 48), dtype=bool)
        result = render_front_depth_grid(depth, valid, 960, 540)
        assert result.shape == (540, 960, 3)

    def test_invalid_bins_remain_muted_instead_of_far_colored(self) -> None:
        depth = np.ones((128, 48), dtype=np.float32)
        valid = np.zeros((128, 48), dtype=bool)
        depth[60:68, 18:30] = 0.25
        valid[60:68, 18:30] = True

        result = render_front_depth_grid(depth, valid, 960, 540)
        feature_patch = result[210:330, 430:530]
        invalid_patch = result[40:120, 40:120]

        feature_mean = np.mean(feature_patch.astype(np.float32), axis=(0, 1))
        invalid_mean = np.mean(invalid_patch.astype(np.float32), axis=(0, 1))

        assert float(np.linalg.norm(feature_mean - invalid_mean)) > 25.0


class TestPanoramaAlignment:
    """Tests for canonical forward-centering before dashboard slicing."""

    def test_center_forward_azimuth_moves_bin_zero_to_middle(self) -> None:
        depth = np.arange(8, dtype=np.float32).reshape(8, 1)

        (centered,) = center_forward_azimuth(depth)

        assert centered.shape == depth.shape
        assert centered[depth.shape[0] // 2, 0] == 0.0

    def test_extract_forward_fov_uses_centered_forward_seam(self) -> None:
        depth = np.arange(12, dtype=np.float32).reshape(12, 1)
        valid = np.ones((12, 1), dtype=bool)

        fov_depth, fov_valid = extract_forward_fov(depth, valid, fov_degrees=120.0)

        assert fov_depth.shape == (4, 1)
        assert fov_valid.shape == (4, 1)
        assert fov_depth[:, 0].tolist() == [10.0, 11.0, 0.0, 1.0]

    def test_oracle_house_forward_fov_preserves_door_opening_at_centered_seam(self) -> None:
        oracle = house_observation()

        fov_depth, fov_valid = extract_forward_fov(oracle.depth, oracle.valid, fov_degrees=120.0)

        assert fov_depth.shape[0] == 4
        assert np.count_nonzero(~fov_valid[:, 1:]) >= 2
        assert np.count_nonzero(~fov_valid[:2, :2]) >= 1

    def test_canonical_half_sphere_extracts_exact_128x48_front_slice(self) -> None:
        depth = np.zeros((256, 48), dtype=np.float32)
        valid = np.ones((256, 48), dtype=bool)

        fov_depth, fov_valid = extract_forward_fov(depth, valid, fov_degrees=180.0)

        assert fov_depth.shape == (128, 48)
        assert fov_valid.shape == (128, 48)


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

    def test_compute_nav_metrics_prefers_farther_forward_sector_when_depth_is_explicitly_larger(
        self,
    ) -> None:
        depth = np.full((24, 2), 0.2, dtype=np.float32)
        valid = np.ones((24, 2), dtype=bool)
        center = depth.shape[0] // 2
        depth[center - 2 : center + 2, :] = 0.7

        fwd, left, right = compute_nav_metrics(depth, valid)

        assert fwd > left
        assert fwd > right


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
