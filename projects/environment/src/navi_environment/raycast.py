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

        # Pre-allocated buffers — reset per call instead of re-creating.
        # Eliminates per-call allocation + GC pressure at O(N x steps).
        self._depth_buf = np.ones((azimuth_bins, elevation_bins), dtype=np.float32)
        self._sem_buf = np.zeros((azimuth_bins, elevation_bins), dtype=np.int32)
        self._valid_buf = np.zeros((azimuth_bins, elevation_bins), dtype=np.bool_)
        self._depth_flat_buf = np.ones(self._total, dtype=np.float32)
        self._sem_flat_buf = np.zeros(self._total, dtype=np.int32)
        self._valid_flat_buf = np.zeros(self._total, dtype=np.bool_)

    def project(
        self,
        relative_xyz: NDArray[np.float32],
        semantic_ids: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
        """Create depth/semantic/mask planes via vectorized scatter-reduce."""
        az, el = self._az, self._el

        # Reset pre-allocated buffers
        self._depth_buf.fill(1.0)
        self._sem_buf.fill(0)
        self._valid_buf.fill(False)

        if relative_xyz.shape[0] == 0:
            return self._depth_buf.copy(), self._sem_buf.copy(), self._valid_buf.copy()

        distances: NDArray[np.float32] = np.linalg.norm(
            relative_xyz, axis=1,
        ).astype(np.float32)
        keep = distances > 1e-6
        if not np.any(keep):
            return self._depth_buf.copy(), self._sem_buf.copy(), self._valid_buf.copy()

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
        self._depth_flat_buf.fill(1.0)
        np.minimum.at(self._depth_flat_buf, flat, normalized)

        # --- semantic: label of closest voxel per bin ---
        self._sem_flat_buf.fill(0)
        order = np.argsort(normalized)
        rev = order[::-1]
        self._sem_flat_buf[flat[rev]] = sem[rev]
        self._sem_flat_buf[flat[order]] = sem[order]

        # --- valid mask: any bin that received a hit ---
        self._valid_flat_buf.fill(False)
        self._valid_flat_buf[np.unique(flat)] = True

        np.copyto(self._depth_buf, self._depth_flat_buf.reshape(az, el))
        np.copyto(self._sem_buf, self._sem_flat_buf.reshape(az, el))
        np.copyto(self._valid_buf, self._valid_flat_buf.reshape(az, el))

        return self._depth_buf.copy(), self._sem_buf.copy(), self._valid_buf.copy()
