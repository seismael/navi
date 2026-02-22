"""Tests for HabitatSemanticLUT category remapping."""

from __future__ import annotations

import numpy as np

from navi_section_manager.backends.habitat_semantic_lut import (
    REPLICACAD_CATEGORY_MAP,
    HabitatSemanticLUT,
)


class TestHabitatSemanticLUT:
    """Validate semantic ID remapping from Habitat instance IDs to Navi 0-10."""

    def test_default_lut_maps_unknown_to_zero(self) -> None:
        lut = HabitatSemanticLUT()
        obs = np.array([[999, 5000, 0]], dtype=np.int32)
        result = lut.remap(obs)
        np.testing.assert_array_equal(result, np.array([[0, 0, 0]], dtype=np.int32))

    def test_remap_shape_preserved(self) -> None:
        lut = HabitatSemanticLUT()
        obs = np.zeros((128, 256), dtype=np.int32)
        result = lut.remap(obs)
        assert result.shape == (128, 256)
        assert result.dtype == np.int32

    def test_category_map_has_expected_entries(self) -> None:
        assert "floor" in REPLICACAD_CATEGORY_MAP
        assert REPLICACAD_CATEGORY_MAP["floor"] == 1
        assert "wall" in REPLICACAD_CATEGORY_MAP
        assert REPLICACAD_CATEGORY_MAP["wall"] == 2
        assert "chair" in REPLICACAD_CATEGORY_MAP
        assert REPLICACAD_CATEGORY_MAP["chair"] == 4
        assert "plant" in REPLICACAD_CATEGORY_MAP
        assert REPLICACAD_CATEGORY_MAP["plant"] == 7

    def test_custom_category_map(self) -> None:
        custom = {"floor": 1, "wall": 3}
        lut = HabitatSemanticLUT(category_map=custom)
        # Without build_from_scene, everything maps to 0
        obs = np.array([[0, 1, 2]], dtype=np.int32)
        result = lut.remap(obs)
        np.testing.assert_array_equal(result, np.array([[0, 0, 0]], dtype=np.int32))

    def test_negative_ids_clamped(self) -> None:
        lut = HabitatSemanticLUT()
        obs = np.array([[-1, -100]], dtype=np.int32)
        result = lut.remap(obs)
        # Clamped to 0 → maps to 0
        np.testing.assert_array_equal(result, np.array([[0, 0]], dtype=np.int32))

    def test_large_ids_clamped(self) -> None:
        lut = HabitatSemanticLUT(max_instances=100)
        obs = np.array([[50, 200]], dtype=np.int32)
        result = lut.remap(obs)
        # 200 clamped to 99 → maps to 0
        assert result[0, 0] == 0
        assert result[0, 1] == 0
