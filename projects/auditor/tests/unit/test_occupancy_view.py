from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.occupancy_view import OccupancyMap


def test_update_marks_free_space_along_expected_forward_column() -> None:
    occupancy = OccupancyMap(cell_size=1.0, grid_extent=20.0, max_distance=10.0)
    depth = np.ones((8, 1), dtype=np.float32)
    valid = np.zeros((8, 1), dtype=np.bool_)
    depth[0, 0] = 0.9
    valid[0, 0] = True

    occupancy.update(depth, valid, x=0.0, z=0.0, yaw=0.0, episode_id=1)

    center = occupancy._n // 2
    assert occupancy._occ[center, center] == 1
    assert occupancy._occ[center - 3, center] == 1
    assert occupancy._occ[center - 3, center - 3] != 1


def test_update_keeps_empty_front_sector_clearer_than_right_wall_sector() -> None:
    occupancy = OccupancyMap(cell_size=1.0, grid_extent=24.0, max_distance=10.0)
    depth = np.ones((8, 1), dtype=np.float32)
    valid = np.zeros((8, 1), dtype=np.bool_)
    depth[2, 0] = 0.3
    valid[2, 0] = True

    occupancy.update(depth, valid, x=0.0, z=0.0, yaw=0.0, episode_id=3)

    center = occupancy._n // 2
    front_band = occupancy._occ[center - 3:center, center - 1:center + 2]
    right_band = occupancy._occ[center - 1:center + 2, center + 2:center + 5]
    assert np.count_nonzero(front_band == 2) < np.count_nonzero(right_band == 2)
