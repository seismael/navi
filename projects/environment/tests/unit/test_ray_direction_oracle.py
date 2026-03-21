"""Phase 5 — Spherical Ray Direction Oracle Tests.

Validates that ``build_spherical_ray_directions`` produces mathematically
correct unit-norm direction vectors for all cardinal directions, and that
the meshgrid ordering is consistent with the downstream reshape.

No CUDA needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from navi_environment.backends.sdfdag_backend import build_spherical_ray_directions

# ── Helpers ──────────────────────────────────────────────────────────

def _dir_at(dirs: np.ndarray, az_idx: int, el_idx: int, el_bins: int) -> np.ndarray:
    """Extract the direction vector for a given (azimuth, elevation) bin.

    The flat array uses meshgrid(az, el, indexing='ij'), so:
        flat_index = az_idx * el_bins + el_idx
    """
    return dirs[az_idx * el_bins + el_idx]


# ── Tests ────────────────────────────────────────────────────────────

class TestCardinalDirections:
    """Verify that specific (azimuth, elevation) bins produce the expected
    unit vectors in local actor coordinates."""

    AZ = 256
    EL = 48

    @pytest.fixture(autouse=True)
    def _build(self) -> None:
        self.dirs = build_spherical_ray_directions(self.AZ, self.EL)

    def test_forward_at_azimuth_zero_elevation_mid(self) -> None:
        """az=0, el=mid → [0, 0, -1] (forward in local frame)."""
        # With even el_bins the midpoint sits between two samples;
        # the closest to el=0 is el_bins//2 when endpoint=True gives
        # el[mid] ≈ 0.  For 48 bins: index 23 or 24 depending on rounding.
        mid_el = self.EL // 2
        d = _dir_at(self.dirs, 0, mid_el, self.EL)
        # At el ≈ 0, cos(el) ≈ 1, sin(el) ≈ 0
        # az = 0: sin(0) = 0, cos(0) = 1  → d = [0, ~0, -1]
        np.testing.assert_allclose(d, [0.0, 0.0, -1.0], atol=0.07)

    def test_right_at_quarter_azimuth(self) -> None:
        """az=N/4 (π/2), el=mid → [1, 0, 0] (right)."""
        mid_el = self.EL // 2
        d = _dir_at(self.dirs, self.AZ // 4, mid_el, self.EL)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=0.07)

    def test_backward_at_half_azimuth(self) -> None:
        """az=N/2 (π), el=mid → [0, 0, 1] (backward)."""
        mid_el = self.EL // 2
        d = _dir_at(self.dirs, self.AZ // 2, mid_el, self.EL)
        np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=0.07)

    def test_left_at_three_quarter_azimuth(self) -> None:
        """az=3N/4 (3π/2), el=mid → [-1, 0, 0] (left)."""
        mid_el = self.EL // 2
        d = _dir_at(self.dirs, 3 * self.AZ // 4, mid_el, self.EL)
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0], atol=0.07)

    def test_up_at_elevation_zero(self) -> None:
        """el=0 → [0, 1, 0] (straight up, any azimuth)."""
        d = _dir_at(self.dirs, 0, 0, self.EL)
        np.testing.assert_allclose(d, [0.0, 1.0, 0.0], atol=1e-5)

    def test_down_at_last_elevation(self) -> None:
        """el=M-1 → [0, -1, 0] (straight down, any azimuth)."""
        d = _dir_at(self.dirs, 0, self.EL - 1, self.EL)
        np.testing.assert_allclose(d, [0.0, -1.0, 0.0], atol=1e-5)


class TestAllRaysUnitNorm:
    """Every direction vector must be unit-length."""

    @pytest.mark.parametrize(("az", "el"), [(8, 3), (64, 16), (256, 48)])
    def test_norms(self, az: int, el: int) -> None:
        dirs = build_spherical_ray_directions(az, el)
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestPoleDegeneracy:
    """At the poles, all azimuth rays collapse to the same direction.

    This is expected behavior with ``endpoint=True`` on the elevation
    grid — documented here to quantify the observation-bandwidth cost.
    """

    AZ = 256
    EL = 48

    @pytest.fixture(autouse=True)
    def _build(self) -> None:
        self.dirs = build_spherical_ray_directions(self.AZ, self.EL)

    def test_top_pole_all_identical(self) -> None:
        top_dirs = np.array([_dir_at(self.dirs, az, 0, self.EL) for az in range(self.AZ)])
        # All should be [0, 1, 0]
        spread = np.ptp(top_dirs, axis=0)
        assert np.all(spread < 1e-5), f"Top-pole directions should be identical; spread={spread}"

    def test_bottom_pole_all_identical(self) -> None:
        bot_dirs = np.array(
            [_dir_at(self.dirs, az, self.EL - 1, self.EL) for az in range(self.AZ)]
        )
        spread = np.ptp(bot_dirs, axis=0)
        assert np.all(spread < 1e-5), f"Bottom-pole directions should be identical; spread={spread}"


class TestMeshgridReshapeConsistency:
    """Verify that flat-ray index ``az * el_bins + el`` maps correctly
    after reshaping to ``[Az, El, 3]``."""

    AZ = 16
    EL = 5

    def test_index_mapping(self) -> None:
        dirs = build_spherical_ray_directions(self.AZ, self.EL)
        grid = dirs.reshape(self.AZ, self.EL, 3)

        for az in range(self.AZ):
            for el in range(self.EL):
                flat_idx = az * self.EL + el
                np.testing.assert_array_equal(
                    grid[az, el],
                    dirs[flat_idx],
                    err_msg=f"Mismatch at (az={az}, el={el}), flat_idx={flat_idx}",
                )
