"""Pruning utilities — distance-based pruning and occlusion culling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["DistancePruner", "OcclusionCuller"]


class DistancePruner:
    """Removes voxels beyond a distance threshold from a centre point."""

    __slots__ = ("_threshold",)

    def __init__(self, threshold: float = 64.0) -> None:
        self._threshold = threshold

    def prune(
        self,
        voxels: NDArray[np.float32],
        center: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Drop voxels farther than *threshold* from *center*.

        Args:
            voxels: ``(N, 5)`` array — ``[x, y, z, density, semantic_id]``.
            center: ``(3,)`` world position of the robot.

        Returns:
            Filtered ``(M, 5)`` array with ``M <= N``.
        """
        if voxels.shape[0] == 0:
            return voxels
        diffs = voxels[:, :3] - center[np.newaxis, :]
        distances = np.linalg.norm(diffs, axis=1)
        mask = distances <= self._threshold
        return voxels[mask]


class OcclusionCuller:
    """Rough voxel-level occlusion culling via distance-weighted density.

    This is a simplified culler that removes voxels that are behind
    dense foreground voxels along the view direction.  A full raycast
    is too expensive per frame — instead we bin voxels into angular
    sectors and keep only the closest dense voxel per sector when the
    sector is fully occluded.
    """

    __slots__ = ("_density_threshold", "_sectors_phi", "_sectors_theta")

    def __init__(
        self,
        sectors_theta: int = 32,
        sectors_phi: int = 16,
        density_threshold: float = 0.8,
    ) -> None:
        self._sectors_theta = sectors_theta
        self._sectors_phi = sectors_phi
        self._density_threshold = density_threshold

    def cull(
        self,
        voxels: NDArray[np.float32],
        eye: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Remove occluded voxels from *eye* viewpoint.

        Args:
            voxels: ``(N, 5)`` — ``[x, y, z, density, semantic_id]``.
            eye: ``(3,)`` observer position.

        Returns:
            Culled ``(M, 5)`` array.
        """
        if voxels.shape[0] == 0:
            return voxels

        rel = voxels[:, :3] - eye[np.newaxis, :]
        distances = np.linalg.norm(rel, axis=1)

        # Prevent division by zero
        safe_dist = np.maximum(distances, 1e-8)
        normed = rel / safe_dist[:, np.newaxis]

        # Spherical angles
        theta = np.arctan2(normed[:, 1], normed[:, 0])  # [-pi, pi]
        phi = np.arcsin(np.clip(normed[:, 2], -1.0, 1.0))  # [-pi/2, pi/2]

        # Bin indices
        ti = ((theta + np.pi) / (2 * np.pi) * self._sectors_theta).astype(np.int32)
        pi_ = ((phi + np.pi / 2) / np.pi * self._sectors_phi).astype(np.int32)
        ti = np.clip(ti, 0, self._sectors_theta - 1)
        pi_ = np.clip(pi_, 0, self._sectors_phi - 1)

        # For each sector, find the closest dense voxel distance
        closest_dense = np.full(
            (self._sectors_theta, self._sectors_phi),
            np.inf,
            dtype=np.float64,
        )
        dense_mask = voxels[:, 3] >= self._density_threshold
        for i in range(voxels.shape[0]):
            if dense_mask[i] and distances[i] < closest_dense[ti[i], pi_[i]]:
                closest_dense[ti[i], pi_[i]] = distances[i]

        # Keep a voxel if it is closer than or equal to the closest dense blocker
        # in its sector (with a small margin), or if the sector has no dense blocker
        keep = np.ones(voxels.shape[0], dtype=np.bool_)
        for i in range(voxels.shape[0]):
            blocker = closest_dense[ti[i], pi_[i]]
            if np.isfinite(blocker) and distances[i] > blocker * 1.1:
                keep[i] = False

        return voxels[keep]
