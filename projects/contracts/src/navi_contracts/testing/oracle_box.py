"""Analytical axis-aligned box oracle for deterministic pipeline testing.

Provides a simple box mesh with analytically-computable SDF and ray
intersection distances so every pipeline stage can be validated against
exact expected values.

Box definition (world coords):
    X: [-1, +1]
    Y: [ 0, +2]
    Z: [-1, +1]
    Center: (0, 1, 0)
    Size: 2 x 2 x 2

All analytical functions assume *unsigned* distance (non-negative).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def write_unit_box_obj(path: str | Path) -> None:
    """Write a deterministic OBJ mesh for the canonical unit box.

    Vertices span X∈[-1,1], Y∈[0,2], Z∈[-1,1].
    12 triangular faces (2 per quad face, 6 faces).
    """
    vertices = [
        (-1.0, 0.0, -1.0),  # 0: back-left-bottom
        (1.0, 0.0, -1.0),   # 1: back-right-bottom
        (1.0, 2.0, -1.0),   # 2: back-right-top
        (-1.0, 2.0, -1.0),  # 3: back-left-top
        (-1.0, 0.0, 1.0),   # 4: front-left-bottom
        (1.0, 0.0, 1.0),    # 5: front-right-bottom
        (1.0, 2.0, 1.0),    # 6: front-right-top
        (-1.0, 2.0, 1.0),   # 7: front-left-top
    ]
    # OBJ indices are 1-based.  Two triangles per quad, outward normals.
    faces = [
        # Back face  (z = -1)
        (1, 3, 2),
        (1, 4, 3),
        # Front face (z = +1)
        (5, 6, 7),
        (5, 7, 8),
        # Bottom face (y = 0)
        (1, 2, 6),
        (1, 6, 5),
        # Top face (y = 2)
        (3, 4, 8),
        (3, 8, 7),
        # Left face  (x = -1)
        (1, 5, 8),
        (1, 8, 4),
        # Right face (x = +1)
        (2, 3, 7),
        (2, 7, 6),
    ]
    lines = [
        *(f"v {x} {y} {z}" for x, y, z in vertices),
        *(f"f {a} {b} {c}" for a, b, c in faces),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Canonical constants ──────────────────────────────────────────────
BOX_MIN = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
BOX_MAX = np.array([1.0, 2.0, 1.0], dtype=np.float32)
BOX_CENTER = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def analytical_unsigned_sdf(x: float, y: float, z: float) -> float:
    """Exact unsigned distance from point (x, y, z) to the box surface.

    For an axis-aligned box [x_min..x_max, y_min..y_max, z_min..z_max],
    the unsigned SDF at any point P is::

        dx = max(x_min - P.x, P.x - x_max, 0)
        dy = max(y_min - P.y, P.y - y_max, 0)
        dz = max(z_min - P.z, P.z - z_max, 0)

        if P is outside:  sdf = sqrt(dx² + dy² + dz²)
        if P is inside:   sdf = min(P.x - x_min, x_max - P.x,
                                     P.y - y_min, y_max - P.y,
                                     P.z - z_min, z_max - P.z)
                          BUT as unsigned: sdf = 0  (on or inside surface)

    For unsigned SDF used by the voxel-dag compiler, inside = 0 distance.
    Actually the Eikonal SDF computes unsigned field where inside points
    get their true unsigned distance to the *nearest surface*.
    """
    # Distances to each face (positive = outside that face)
    dx = max(-1.0 - x, x - 1.0, 0.0)
    dy = max(0.0 - y, y - 2.0, 0.0)
    dz = max(-1.0 - z, z - 1.0, 0.0)

    outside_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if outside_dist > 0.0:
        return outside_dist

    # Inside the box: distance to nearest face
    return min(
        x - (-1.0),
        1.0 - x,
        y - 0.0,
        2.0 - y,
        z - (-1.0),
        1.0 - z,
    )


def analytical_ray_box_distance(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float | None:
    """Exact ray-box intersection distance for the canonical unit box.

    Uses slab method for AABB ray intersection.

    Returns the *first positive* intersection distance (entering or
    exiting the box from inside), or None if the ray misses.

    ``direction`` must be a unit vector.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction

    bmin = (-1.0, 0.0, -1.0)
    bmax = (1.0, 2.0, 1.0)

    t_near = -math.inf
    t_far = math.inf

    for o, d, lo, hi in zip(
        (ox, oy, oz), (dx, dy, dz), bmin, bmax, strict=True
    ):
        if abs(d) < 1e-12:
            # Ray parallel to slab — miss if origin outside
            if o < lo or o > hi:
                return None
            continue
        inv_d = 1.0 / d
        t1 = (lo - o) * inv_d
        t2 = (hi - o) * inv_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None

    # Return first positive hit
    if t_near > 1e-6:
        return t_near
    if t_far > 1e-6:
        return t_far
    return None
