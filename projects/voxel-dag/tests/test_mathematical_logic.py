"""Mathematical logic and Eikonal property validation for voxel-dag."""

from __future__ import annotations

import numpy as np
import pytest
from voxel_dag.compiler import EikonalSdfComputer, MeshIngestor


def test_sdf_satisfies_eikonal_equation() -> None:
    """The Signed Distance Field must satisfy ||grad f|| = 1 almost everywhere."""
    # Create a simple cube mesh
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    indices = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
    ], dtype=np.int32)
    
    bbox_min = np.array([0, 0, 0], dtype=np.float32)
    bbox_max = np.array([1, 1, 1], dtype=np.float32)
    resolution = 32
    
    computer = EikonalSdfComputer()
    grid, voxel_size, cube_min = computer.compute(
        vertices, indices, bbox_min, bbox_max, resolution, padding=0.5
    )
    
    # Compute numerical gradient
    # np.gradient returns a list of arrays, one for each dimension
    # spacing is the voxel_size
    grad_z, grad_y, grad_x = np.gradient(grid, voxel_size)
    
    grad_norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # The Eikonal equation |grad f| = 1 should hold away from the boundary and skeleton
    # We'll check the median to avoid spikes at the mesh surface or medial axis
    mask = np.zeros_like(grid, dtype=bool)
    mask[4:-4, 4:-4, 4:-4] = True # Stay away from boundaries
    
    valid_norms = grad_norm[mask]
    median_norm = np.median(valid_norms)
    
    # Assert median is close to 1.0
    assert np.isclose(median_norm, 1.0, atol=0.1), f"Median gradient norm {median_norm} is not 1.0"
    
    # Also check that most values are near 1.0
    within_tolerance = np.mean(np.abs(valid_norms - 1.0) < 0.2)
    assert within_tolerance > 0.8, f"Only {within_tolerance*100:.1f}% of voxels satisfy Eikonal property"


def test_sdf_is_non_negative_for_unsigned_field() -> None:
    """The current compute_dense_sdf implementation produces an unsigned distance field."""
    vertices = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    indices = np.zeros((0, 3), dtype=np.int32)
    bbox_min = np.array([0, 0, 0], dtype=np.float32)
    bbox_max = np.array([1, 1, 1], dtype=np.float32)
    
    computer = EikonalSdfComputer()
    grid, _, _ = computer.compute(vertices, indices, bbox_min, bbox_max, 16)
    
    assert np.all(grid >= 0.0), "Unsigned distance field contains negative values"


def test_sdf_at_vertex_is_zero() -> None:
    """Distance at a vertex location should be approximately zero."""
    vertices = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    indices = np.zeros((0, 3), dtype=np.int32)
    bbox_min = np.array([0, 0, 0], dtype=np.float32)
    bbox_max = np.array([1, 1, 1], dtype=np.float32)
    resolution = 32
    
    computer = EikonalSdfComputer()
    grid, voxel_size, cube_min = computer.compute(
        vertices, indices, bbox_min, bbox_max, resolution, padding=0.0
    )
    
    # Vertex is at (0.5, 0.5, 0.5). 
    # Voxel indices for this point:
    ix = int((0.5 - cube_min[0]) / voxel_size)
    iy = int((0.5 - cube_min[1]) / voxel_size)
    iz = int((0.5 - cube_min[2]) / voxel_size)
    
    # Distance should be small (at most half-diagonal of a voxel)
    assert grid[iz, iy, ix] < voxel_size * 0.9
