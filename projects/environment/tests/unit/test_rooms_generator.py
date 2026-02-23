"""Tests for RoomsGenerator environment clarity and navigation structure."""

from __future__ import annotations

import numpy as np

from navi_environment.generators.rooms import RoomsGenerator


def test_rooms_spawn_area_is_navigable_in_origin_chunk() -> None:
    generator = RoomsGenerator(seed=42, chunk_size=16)
    chunk = generator.generate_chunk(0, 0, 0)

    center = 8
    # Floor at y=0 and open air at y=1 in spawn column.
    assert chunk[center, 0, center, 0] == 1.0
    assert chunk[center, 1, center, 0] == 0.0


def test_rooms_have_clear_perimeter_barrier_corners() -> None:
    generator = RoomsGenerator(seed=42, chunk_size=16)
    chunk = generator.generate_chunk(1, 0, 1)

    # Corner wall columns should stay solid and be semantic wall.
    for x, z in ((0, 0), (0, 15), (15, 0), (15, 15)):
        assert chunk[x, 1, z, 0] == 1.0
        assert chunk[x, 1, z, 1] == 1.0


def test_rooms_include_obstacle_blockers_in_interior() -> None:
    generator = RoomsGenerator(seed=42, chunk_size=16)
    obstacle_count = 0

    # Aggregate across nearby chunks to avoid relying on one random draw.
    for cx in range(-1, 2):
        for cz in range(-1, 2):
            chunk = generator.generate_chunk(cx, 0, cz)
            obstacle_count += int(np.count_nonzero(chunk[:, :, :, 1] == 6.0))

    assert obstacle_count > 0
