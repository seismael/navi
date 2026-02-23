"""FrustumLoader — velocity-based predictive pre-loading of voxel chunks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["FrustumLoader"]


class FrustumLoader:
    """Computes a frustum of chunk coordinates in the direction of movement.

    Given a velocity vector, predicts which chunks the robot will need soon
    and returns them ordered by distance so the look-ahead buffer can
    prefetch them.
    """

    __slots__ = ("_chunk_size", "_far", "_half_angle", "_near")

    def __init__(
        self,
        chunk_size: int = 16,
        half_angle_deg: float = 45.0,
        near: int = 1,
        far: int = 4,
    ) -> None:
        self._chunk_size = chunk_size
        self._half_angle = math.radians(half_angle_deg)
        self._near = near
        self._far = far

    def compute_frustum(
        self,
        center_chunk: tuple[int, int, int],
        velocity: NDArray[np.float32],
    ) -> list[tuple[int, int, int]]:
        """Return chunk coords inside a frustum along *velocity*.

        Args:
            center_chunk: The current window centre in chunk coords.
            velocity: ``(3,)`` velocity vector ``[vx, vy, vz]``.

        Returns:
            List of ``(cx, cy, cz)`` chunk coordinates sorted by distance.
        """
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-8:
            return []

        direction = velocity / speed
        cx, cy, cz = center_chunk
        candidates: list[tuple[float, tuple[int, int, int]]] = []

        for dx in range(-self._far, self._far + 1):
            for dy in range(-self._far, self._far + 1):
                for dz in range(-self._far, self._far + 1):
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist < self._near or dist > self._far:
                        continue
                    # Check angle between offset and velocity direction
                    offset = np.array([dx, dy, dz], dtype=np.float32)
                    offset_norm = float(np.linalg.norm(offset))
                    if offset_norm < 1e-8:
                        continue
                    cos_angle = float(np.dot(direction, offset / offset_norm))
                    if cos_angle >= math.cos(self._half_angle):
                        candidates.append((dist, (cx + dx, cy + dy, cz + dz)))

        candidates.sort(key=lambda t: t[0])
        return [coord for _, coord in candidates]
