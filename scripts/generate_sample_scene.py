"""Generate a sample indoor apartment scene as a .glb mesh.

Creates a realistic multi-room indoor environment using trimesh
geometric primitives — walls, floors, ceilings, doorways, furniture,
and obstacles.  The output mesh is compatible with ``MeshSceneBackend``
and uses the same .glb format as Habitat's ReplicaCAD scenes.

Usage::

    uv run --project projects/environment scripts/generate_sample_scene.py

Outputs ``data/scenes/sample_apartment.glb``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _box(
    extents: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> object:
    """Create a box primitive with the given extents and position."""
    import trimesh  # type: ignore[import-untyped]

    box = trimesh.creation.box(extents=extents)
    box.apply_translation(translation)
    return box


def build_apartment() -> object:
    """Build a multi-room apartment mesh.

    Layout (top-down, Z-right, X-forward)::

        ┌──────────┬──────────┐
        │          │          │
        │  Room A  D  Room B  │
        │  6×6     │  6×6     │
        │          │          │
        ├────D─────┼──────────┤
        │          │          │
        │  Hall    │  Room C  │
        │  6×3     D  4×6     │
        │          │          │
        └──────────┴──────────┘

    Total footprint: ~12m × 12m, ceiling 3m
    """
    import trimesh  # type: ignore[import-untyped]

    meshes: list[object] = []
    wall_h = 3.0
    wall_t = 0.15  # wall thickness
    floor_y = 0.0

    # ── Floor ────────────────────────────────────────────────
    meshes.append(_box((14.0, 0.1, 14.0), (7.0, floor_y - 0.05, 7.0)))

    # ── Ceiling ──────────────────────────────────────────────
    meshes.append(_box((14.0, 0.1, 14.0), (7.0, wall_h + 0.05, 7.0)))

    # ── Outer walls ──────────────────────────────────────────
    # South wall (Z=0)
    meshes.append(_box((14.0, wall_h, wall_t), (7.0, wall_h / 2, 0.0)))
    # North wall (Z=14)
    meshes.append(_box((14.0, wall_h, wall_t), (7.0, wall_h / 2, 14.0)))
    # West wall (X=0)
    meshes.append(_box((wall_t, wall_h, 14.0), (0.0, wall_h / 2, 7.0)))
    # East wall (X=14)
    meshes.append(_box((wall_t, wall_h, 14.0), (14.0, wall_h / 2, 7.0)))

    # ── Interior walls ───────────────────────────────────────
    # Horizontal divider at X=7 (partial — with doorways)
    # Left segment: Z=0 to Z=3
    meshes.append(_box((wall_t, wall_h, 3.0), (7.0, wall_h / 2, 1.5)))
    # Middle segment: Z=4 to Z=6.5
    meshes.append(_box((wall_t, wall_h, 2.5), (7.0, wall_h / 2, 5.25)))
    # Right segment: Z=8.5 to Z=14
    meshes.append(_box((wall_t, wall_h, 5.5), (7.0, wall_h / 2, 11.25)))

    # Vertical divider at Z=7 (partial — with doorways)
    # Bottom segment: X=0 to X=3
    meshes.append(_box((3.0, wall_h, wall_t), (1.5, wall_h / 2, 7.0)))
    # Middle gap (doorway) X=3 to X=4.2
    # Top segment: X=4.2 to X=7
    meshes.append(_box((2.8, wall_h, wall_t), (5.6, wall_h / 2, 7.0)))
    # Right side: X=7 to X=10.5
    meshes.append(_box((3.5, wall_h, wall_t), (8.75, wall_h / 2, 7.0)))
    # Gap (doorway) X=10.5 to X=11.7
    # Segment: X=11.7 to X=14
    meshes.append(_box((2.3, wall_h, wall_t), (12.85, wall_h / 2, 7.0)))

    # ── Furniture / obstacles ────────────────────────────────
    # Room A: table + chairs
    meshes.append(_box((1.6, 0.75, 0.8), (3.0, 0.375, 3.5)))    # table
    meshes.append(_box((0.4, 0.9, 0.4), (2.0, 0.45, 3.5)))      # chair
    meshes.append(_box((0.4, 0.9, 0.4), (4.0, 0.45, 3.5)))      # chair
    meshes.append(_box((0.8, 0.5, 0.8), (1.5, 0.25, 1.5)))      # low table

    # Room B: sofa + bookshelf
    meshes.append(_box((2.0, 0.8, 0.8), (10.5, 0.4, 10.0)))     # sofa
    meshes.append(_box((0.4, 2.0, 1.5), (13.5, 1.0, 10.5)))     # bookshelf

    # Room C: desk + cabinet
    meshes.append(_box((1.2, 0.75, 0.6), (10.0, 0.375, 3.0)))   # desk
    meshes.append(_box((0.6, 1.5, 0.5), (13.0, 0.75, 1.0)))     # cabinet

    # Hallway: bench
    meshes.append(_box((1.0, 0.45, 0.4), (3.5, 0.225, 9.0)))    # bench

    # Scatter some pillar-like obstacles
    for px, pz in [(5.0, 11.0), (9.0, 5.0), (2.0, 12.0), (11.0, 12.0)]:
        meshes.append(_box((0.3, wall_h, 0.3), (px, wall_h / 2, pz)))

    # Combine all meshes
    combined = trimesh.util.concatenate(meshes)  # type: ignore[arg-type]

    return combined


def main() -> None:
    """Generate and save the sample apartment scene."""
    output_dir = Path(__file__).resolve().parent.parent / "data" / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_apartment.glb"

    print("Building sample apartment scene...")
    mesh = build_apartment()

    print(f"  Vertices: {mesh.vertices.shape[0]}")  # type: ignore[union-attr]
    print(f"  Faces:    {mesh.faces.shape[0]}")  # type: ignore[union-attr]

    # Export as .glb
    mesh.export(str(output_path), file_type="glb")  # type: ignore[union-attr]
    print(f"  Saved to: {output_path}")

    # Also create a minimal PointNav episodes JSON
    episodes_dir = output_dir
    episodes_path = episodes_dir / "sample_episodes.json"

    rng = np.random.default_rng(42)
    episodes = []
    for i in range(50):
        # Random start positions within the apartment
        sx = float(rng.uniform(1.5, 12.5))
        sz = float(rng.uniform(1.5, 12.5))
        sy = 1.5  # standing height

        # Random goal position
        gx = float(rng.uniform(1.5, 12.5))
        gz = float(rng.uniform(1.5, 12.5))

        # Random start rotation (yaw only, as quaternion around Y)
        yaw = float(rng.uniform(0, 2 * np.pi))
        qw = float(np.cos(yaw / 2))
        qy = float(np.sin(yaw / 2))

        episodes.append({
            "episode_id": str(i),
            "scene_id": str(output_path),
            "start_position": [sx, sy, sz],
            "start_rotation": [0.0, qy, 0.0, qw],
            "goals": [{"position": [gx, 1.5, gz], "radius": 0.5}],
        })

    import json

    episodes_data = {"episodes": episodes}
    episodes_path.write_text(json.dumps(episodes_data, indent=2), encoding="utf-8")
    print(f"  Episodes: {episodes_path} ({len(episodes)} episodes)")
    print("Done!")


if __name__ == "__main__":
    sys.exit(main() or 0)
