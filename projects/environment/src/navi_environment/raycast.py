"""Vectorized spherical-projection Z-buffer for distance matrix generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["RaycastEngine"]


class RaycastEngine:
    """Projects local voxels into discretized azimuth/elevation bins.

    Uses fully vectorized ``np.minimum.at`` scatter-reduce instead of
    a Python for-loop.  Scales to 256x128+ bins and tens of thousands
    of voxels without measurable overhead.
    """

    def __init__(self, azimuth_bins: int, elevation_bins: int, max_distance: float) -> None:
        self._az = azimuth_bins
        self._el = elevation_bins
        self._max_distance = max_distance
        self._total = azimuth_bins * elevation_bins

    def project(
        self,
        relative_xyz: NDArray[np.float32],
        semantic_ids: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
        """Create depth/semantic/mask planes via vectorized scatter-reduce."""
        az, el = self._az, self._el
        depth = np.ones((az, el), dtype=np.float32)
        semantic = np.zeros((az, el), dtype=np.int32)
        valid_mask = np.zeros((az, el), dtype=np.bool_)

        if relative_xyz.shape[0] == 0:
            return depth, semantic, valid_mask

        distances: NDArray[np.float32] = np.linalg.norm(
            relative_xyz, axis=1,
        ).astype(np.float32)
        keep = distances > 1e-6
        if not np.any(keep):
            return depth, semantic, valid_mask

        pts = relative_xyz[keep]
        sem = semantic_ids[keep]
        dists = distances[keep]

        # --- vectorized spherical coordinates ---
        # X = forward, Z = right (horizontal plane), Y = up (vertical).
        # Azimuth sweeps the horizontal XZ plane; elevation lifts off it.
        azimuth = np.arctan2(pts[:, 2], pts[:, 0])
        planar = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
        elevation = np.arctan2(pts[:, 1], planar)

        az_idx = np.clip(
            ((azimuth + np.pi) / (2.0 * np.pi) * az).astype(np.intp), 0, az - 1,
        )
        el_idx = np.clip(
            ((elevation + np.pi / 2.0) / np.pi * el).astype(np.intp), 0, el - 1,
        )

        normalized: NDArray[np.float32] = np.clip(
            dists / self._max_distance, 0.0, 1.0,
        ).astype(np.float32)

        # --- scatter-reduce: nearest hit per bin (depth) ---
        flat = az_idx * el + el_idx
        depth_flat = np.ones(self._total, dtype=np.float32)
        np.minimum.at(depth_flat, flat, normalized)

        # --- semantic: label of closest voxel per bin ---
        semantic_flat = np.zeros(self._total, dtype=np.int32)
        order = np.argsort(normalized)
        rev = order[::-1]
        semantic_flat[flat[rev]] = sem[rev]
        semantic_flat[flat[order]] = sem[order]

        # --- valid mask: any bin that received a hit ---
        valid_flat = np.zeros(self._total, dtype=np.bool_)
        valid_flat[np.unique(flat)] = True

        depth = depth_flat.reshape(az, el)
        semantic = semantic_flat.reshape(az, el)
        valid_mask = valid_flat.reshape(az, el)

        return depth, semantic, valid_mask
