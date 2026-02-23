"""AbstractWorldGenerator — strategy interface for voxel world generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__: list[str] = ["AbstractWorldGenerator"]


class AbstractWorldGenerator(ABC):
    """Strategy interface for producing voxel chunk data.

    Implementations produce a dense ``(chunk_size, chunk_size, chunk_size, 2)``
    array for a given chunk coordinate.  Channel 0 is ``density`` (float32)
    and channel 1 is ``semantic_id`` (float32-encoded integer).

    No color, no reflectivity — purely geometric + semantic.
    """

    @abstractmethod
    def generate_chunk(
        self,
        cx: int,
        cy: int,
        cz: int,
    ) -> NDArray[np.float32]:
        """Generate voxel data for the chunk at ``(cx, cy, cz)``.

        Returns:
            Dense array of shape ``(C, C, C, 2)`` — ``[density, semantic_id]``.
        """
        ...

    @abstractmethod
    def spawn_position(self) -> tuple[float, float, float]:
        """Return a valid starting position in world coordinates."""
        ...
