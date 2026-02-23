"""Tests for FrustumLoader."""

from __future__ import annotations

import numpy as np

from navi_environment.frustum import FrustumLoader


class TestFrustumLoader:
    """Unit tests for velocity-based frustum prediction."""

    def test_zero_velocity_returns_empty(self) -> None:
        loader = FrustumLoader(chunk_size=16, half_angle_deg=45.0, near=1, far=3)
        result = loader.compute_frustum(
            (0, 0, 0),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        assert result == []

    def test_forward_velocity_returns_chunks_ahead(self) -> None:
        loader = FrustumLoader(chunk_size=16, half_angle_deg=45.0, near=1, far=3)
        result = loader.compute_frustum(
            (0, 0, 0),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        assert len(result) > 0
        # All returned chunks should have positive x offset from centre
        for cx, _cy, _cz in result:
            assert cx >= 0, "Expected chunks in forward direction"

    def test_results_sorted_by_distance(self) -> None:
        loader = FrustumLoader(chunk_size=16, half_angle_deg=60.0, near=1, far=4)
        result = loader.compute_frustum(
            (5, 5, 5),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        if len(result) >= 2:
            prev_dist = 0.0
            for cx, cy, cz in result:
                dx = cx - 5
                dy = cy - 5
                dz = cz - 5
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                assert dist >= prev_dist - 1e-6
                prev_dist = dist

    def test_near_far_filtering(self) -> None:
        loader = FrustumLoader(chunk_size=16, half_angle_deg=90.0, near=2, far=3)
        result = loader.compute_frustum(
            (0, 0, 0),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        for cx, cy, cz in result:
            dist = (cx * cx + cy * cy + cz * cz) ** 0.5
            assert dist >= 2.0 - 1e-6
            assert dist <= 3.0 + 1e-6
