"""Semantic category mapping from Habitat instance/category IDs to Navi's 0-10 range.

Navi semantic IDs:
    0 — air / empty
    1 — floor / ground
    2 — wall / barrier
    3 — ceiling / overhead
    4 — furniture / obstacle (static)
    5 — object / interactable
    6 — structure (door, window, column)
    7 — vegetation / organic
    8 — water / liquid
    9 — hazard / dynamic obstacle
   10 — target / goal
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = ["HabitatSemanticLUT", "REPLICACAD_CATEGORY_MAP"]

# Default mapping for ReplicaCAD mpcat40 categories.
# Keys are Habitat category names (lowered); values are Navi semantic IDs.
REPLICACAD_CATEGORY_MAP: dict[str, int] = {
    # Structural
    "void": 0,
    "floor": 1,
    "wall": 2,
    "ceiling": 3,
    "door": 6,
    "window": 6,
    "column": 6,
    "beam": 6,
    "stairs": 6,
    "railing": 6,
    # Furniture
    "chair": 4,
    "table": 4,
    "sofa": 4,
    "bed": 4,
    "desk": 4,
    "cabinet": 4,
    "shelf": 4,
    "counter": 4,
    "dresser": 4,
    "nightstand": 4,
    "bench": 4,
    "stool": 4,
    "wardrobe": 4,
    "bookshelf": 4,
    "tv_stand": 4,
    # Objects
    "cushion": 5,
    "lamp": 5,
    "plant": 7,
    "book": 5,
    "picture": 5,
    "mirror": 5,
    "towel": 5,
    "curtain": 5,
    "rug": 5,
    "blanket": 5,
    "appliance": 5,
    "electronics": 5,
    "lighting": 5,
    "clothing": 5,
    # Misc
    "misc": 5,
    "unknown": 0,
}


class HabitatSemanticLUT:
    """Maps Habitat semantic instance IDs to Navi's 0-10 semantic range.

    Builds a uint8 lookup table from habitat-sim's
    ``SemanticScene.objects`` list so that per-pixel remapping is a
    single ``lut[semantic_obs]`` operation.
    """

    def __init__(
        self,
        category_map: dict[str, int] | None = None,
        *,
        max_instances: int = 10_000,
    ) -> None:
        self._category_map = category_map or REPLICACAD_CATEGORY_MAP
        self._max_instances = max_instances
        # Default: everything maps to 0 (air/empty)
        self._lut = np.zeros(max_instances, dtype=np.uint8)

    def build_from_scene(self, semantic_scene: object) -> None:
        """Populate the LUT from a ``habitat_sim.scene.SemanticScene``.

        Parameters
        ----------
        semantic_scene:
            The ``sim.semantic_scene`` object whose ``.objects`` list
            provides ``(id, category.name())`` pairs.
        """
        objects = getattr(semantic_scene, "objects", [])
        for obj in objects:
            obj_id: int = obj.id
            if obj_id < 0 or obj_id >= self._max_instances:
                continue
            cat_name = obj.category.name().lower().strip() if obj.category else "unknown"
            self._lut[obj_id] = self._category_map.get(cat_name, 0)

    def remap(self, semantic_obs: NDArray[np.int32]) -> NDArray[np.int32]:
        """Remap a Habitat semantic observation to Navi IDs.

        Parameters
        ----------
        semantic_obs:
            ``(H, W)`` uint32 / int32 array of Habitat instance IDs.

        Returns
        -------
        NDArray[np.int32]
            ``(H, W)`` array with values in ``[0, 10]``.
        """
        # Clamp to LUT range
        clamped = np.clip(semantic_obs.astype(np.int64), 0, self._max_instances - 1)
        return self._lut[clamped].astype(np.int32)
