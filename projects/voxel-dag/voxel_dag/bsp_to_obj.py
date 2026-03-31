"""Quake 3 BSP → Wavefront OBJ converter for the voxel-dag compilation pipeline.

Parses Quake 3 IBSP format (.bsp) files — including full biquadratic Bezier
patch tessellation — and emits clean Wavefront .obj geometry suitable for SDF
voxelization via the voxel-dag compiler.

PK3 archives (.pk3) are also supported: the converter extracts BSP files from
the ``maps/`` directory inside the ZIP and converts each one.

Reference: https://www.mralligator.com/q3/ — Unofficial Quake 3 Map Specs.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "Q3BspParser",
    "BezierPatchTessellator",
    "Q3BspToObjConverter",
    "extract_bsp_from_pk3",
    "main",
]

# ── Q3 BSP constants ────────────────────────────────────────────

_IBSP_MAGIC = b"IBSP"
_IBSP_VERSION = 0x2E  # 46 — Quake 3 Arena

# Lump indices per the spec.
_LUMP_ENTITIES = 0
_LUMP_TEXTURES = 1
_LUMP_PLANES = 2
_LUMP_NODES = 3
_LUMP_LEAFS = 4
_LUMP_LEAFFACES = 5
_LUMP_LEAFBRUSHES = 6
_LUMP_MODELS = 7
_LUMP_BRUSHES = 8
_LUMP_BRUSHSIDES = 9
_LUMP_VERTEXES = 10
_LUMP_MESHVERTS = 11
_LUMP_EFFECTS = 12
_LUMP_FACES = 13
_LUMP_LIGHTMAPS = 14
_LUMP_LIGHTVOLS = 15
_LUMP_VISDATA = 16
_NUM_LUMPS = 17

# Face types.
_FACE_POLYGON = 1
_FACE_PATCH = 2
_FACE_MESH = 3
_FACE_BILLBOARD = 4

# Record sizes (bytes).
_VERTEX_SIZE = 44  # float[3]+float[2][2]+float[3]+ubyte[4]
_MESHVERT_SIZE = 4  # int
_FACE_SIZE = 104  # see spec

# Quake 3 uses ~1 unit ≈ 1 inch.  Convert to metres.
_Q3_UNITS_TO_METRES = 0.0254


# ── Data structures ─────────────────────────────────────────────

@dataclass(frozen=True)
class Q3Vertex:
    """A single vertex from lump 10."""

    position: np.ndarray  # float32[3]
    normal: np.ndarray  # float32[3]


@dataclass(frozen=True)
class Q3Face:
    """A single face record from lump 13."""

    face_type: int
    vertex_index: int
    n_vertexes: int
    meshvert_index: int
    n_meshverts: int
    patch_size: tuple[int, int]  # only meaningful for type-2 faces


@dataclass
class TriangleMesh:
    """Accumulated triangle-soup geometry ready for OBJ export."""

    vertices: list[np.ndarray] = field(default_factory=list)  # float32[3] each
    indices: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices)


@dataclass
class Q3SpawnPoint:
    """A player spawn extracted from the entities lump."""

    origin: np.ndarray  # float32[3] in metres
    angle: float  # yaw in degrees


# ── BSP Parser ──────────────────────────────────────────────────


class Q3BspParser:
    """Binary parser for Quake 3 IBSP files.

    Extracts the data needed for geometry reconstruction:
    vertexes, meshverts, and faces.  Entity spawn points are also parsed
    so downstream tooling can place drones at valid locations.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._lump_entries: list[tuple[int, int]] = []
        self.vertices: list[Q3Vertex] = []
        self.meshverts: list[int] = []
        self.faces: list[Q3Face] = []
        self.spawns: list[Q3SpawnPoint] = []
        self._parse()

    # ── internal ────────────────────────────────────────────────

    def _parse(self) -> None:
        self._parse_header()
        self._parse_vertices()
        self._parse_meshverts()
        self._parse_faces()
        self._parse_entities()

    def _parse_header(self) -> None:
        if len(self._data) < 8 + _NUM_LUMPS * 8:
            msg = "File too small for an IBSP header."
            raise ValueError(msg)

        magic = self._data[:4]
        if magic != _IBSP_MAGIC:
            msg = f"Invalid IBSP magic: {magic!r} (expected {_IBSP_MAGIC!r})."
            raise ValueError(msg)

        (version,) = struct.unpack_from("<i", self._data, 4)
        if version != _IBSP_VERSION:
            msg = f"Unsupported IBSP version {version:#x} (expected {_IBSP_VERSION:#x})."
            raise ValueError(msg)

        for i in range(_NUM_LUMPS):
            offset_pos = 8 + i * 8
            offset, length = struct.unpack_from("<ii", self._data, offset_pos)
            self._lump_entries.append((offset, length))

    def _lump_data(self, index: int) -> memoryview:
        offset, length = self._lump_entries[index]
        return memoryview(self._data)[offset : offset + length]

    def _parse_vertices(self) -> None:
        raw = self._lump_data(_LUMP_VERTEXES)
        count = len(raw) // _VERTEX_SIZE
        for i in range(count):
            base = i * _VERTEX_SIZE
            x, y, z = struct.unpack_from("<fff", raw, base)
            # Skip texcoords (2*2 floats = 16 bytes) at offset 12.
            nx, ny, nz = struct.unpack_from("<fff", raw, base + 28)
            pos = np.array([x, y, z], dtype=np.float32)
            nrm = np.array([nx, ny, nz], dtype=np.float32)
            self.vertices.append(Q3Vertex(position=pos, normal=nrm))

    def _parse_meshverts(self) -> None:
        raw = self._lump_data(_LUMP_MESHVERTS)
        count = len(raw) // _MESHVERT_SIZE
        for i in range(count):
            (offset,) = struct.unpack_from("<i", raw, i * _MESHVERT_SIZE)
            self.meshverts.append(offset)

    def _parse_faces(self) -> None:
        raw = self._lump_data(_LUMP_FACES)
        count = len(raw) // _FACE_SIZE
        for i in range(count):
            base = i * _FACE_SIZE
            # Unpack the face fields we need.
            # texture(4) effect(4) type(4) vertex(4) n_verts(4)
            # meshvert(4) n_meshverts(4) lm_index(4) lm_start(8)
            # lm_size(8) lm_origin(12) lm_vecs(24) normal(12) size(8)
            _tex, _eff, ftype = struct.unpack_from("<iii", raw, base)
            vertex_idx, n_verts = struct.unpack_from("<ii", raw, base + 12)
            meshvert_idx, n_meshverts = struct.unpack_from("<ii", raw, base + 20)
            # patch size at offset 96 (two ints).
            sx, sy = struct.unpack_from("<ii", raw, base + 96)
            self.faces.append(
                Q3Face(
                    face_type=ftype,
                    vertex_index=vertex_idx,
                    n_vertexes=n_verts,
                    meshvert_index=meshvert_idx,
                    n_meshverts=n_meshverts,
                    patch_size=(sx, sy),
                )
            )

    def _parse_entities(self) -> None:
        """Extract ``info_player_deathmatch`` spawn points from lump 0."""
        raw = self._lump_data(_LUMP_ENTITIES)
        try:
            text = bytes(raw).decode("ascii", errors="replace")
        except Exception:
            return

        # Strip trailing null bytes (Q3 entity strings are null-terminated).
        text = text.rstrip("\x00")

        # Simple brace-block parser for the entity descriptions.
        in_block = False
        current: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "{":
                in_block = True
                current = {}
            elif stripped == "}":
                if in_block:
                    classname = current.get("classname", "")
                    origin_str = current.get("origin", "")
                    if classname in (
                        "info_player_deathmatch",
                        "info_player_start",
                        "info_player_intermission",
                    ) and origin_str:
                        parts = origin_str.split()
                        if len(parts) >= 3:
                            try:
                                ox = float(parts[0]) * _Q3_UNITS_TO_METRES
                                oy = float(parts[1]) * _Q3_UNITS_TO_METRES
                                oz = float(parts[2]) * _Q3_UNITS_TO_METRES
                                angle = float(current.get("angle", "0"))
                                # Q3 is Z-up; Navi expects Y-up.
                                self.spawns.append(
                                    Q3SpawnPoint(
                                        origin=np.array(
                                            [ox, oz, -oy], dtype=np.float32
                                        ),
                                        angle=angle,
                                    )
                                )
                            except ValueError:
                                pass
                in_block = False
            elif in_block and stripped.startswith('"'):
                # Parse key-value: "key" "value"
                parts = stripped.split('"')
                if len(parts) >= 5:
                    key = parts[1]
                    value = parts[3]
                    current[key] = value


# ── Bezier Patch Tessellation ───────────────────────────────────


class BezierPatchTessellator:
    """Tessellate biquadratic Bezier patches from Q3 BSP type-2 faces.

    Each 3×3 block of control points defines one biquadratic patch.
    Adjacent patches share a row or column of control vertices.

    The tessellation level controls how many samples per patch edge.
    Higher levels produce smoother curves at the cost of more triangles.
    """

    def __init__(self, tessellation_level: int = 4) -> None:
        if tessellation_level < 1:
            msg = "Tessellation level must be >= 1."
            raise ValueError(msg)
        self.level = tessellation_level

    def tessellate_face(
        self,
        control_verts: np.ndarray,
        patch_width: int,
        patch_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tessellate a complete BSP patch face.

        Parameters
        ----------
        control_verts:
            ``(patch_height, patch_width, 3)`` grid of control-point positions.
        patch_width, patch_height:
            Face ``size`` from the BSP face record.

        Returns
        -------
        vertices : ndarray, shape ``(N, 3)``
        indices  : ndarray, shape ``(M, 3)`` — triangle indices into *vertices*.
        """
        n_patches_x = (patch_width - 1) // 2
        n_patches_y = (patch_height - 1) // 2

        all_verts: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        vert_offset = 0

        for py in range(n_patches_y):
            for px in range(n_patches_x):
                # Extract the 3×3 control grid for this sub-patch.
                r = py * 2
                c = px * 2
                cp = control_verts[r : r + 3, c : c + 3]  # (3, 3, 3)

                verts, tris = self._tessellate_single_patch(cp)
                tris_offset = tris + vert_offset
                all_verts.append(verts)
                all_tris.append(tris_offset)
                vert_offset += len(verts)

        if not all_verts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(
                (0, 3), dtype=np.int32
            )

        return np.concatenate(all_verts, axis=0), np.concatenate(all_tris, axis=0)

    def _tessellate_single_patch(
        self,
        cp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tessellate a single 3×3 biquadratic Bezier patch.

        Returns vertex positions ``(L+1)²×3`` and triangle indices ``2L²×3``.
        """
        L = self.level  # noqa: N806 — conventional name
        n_verts = (L + 1) * (L + 1)

        verts = np.empty((n_verts, 3), dtype=np.float32)
        idx = 0
        for row in range(L + 1):
            t = row / L
            for col in range(L + 1):
                s = col / L
                verts[idx] = self._evaluate(cp, s, t)
                idx += 1

        # Build triangle strip indices.
        tris: list[tuple[int, int, int]] = []
        for row in range(L):
            for col in range(L):
                a = row * (L + 1) + col
                b = a + 1
                c = (row + 1) * (L + 1) + col
                d = c + 1
                tris.append((a, c, b))
                tris.append((b, c, d))

        return verts, np.array(tris, dtype=np.int32)

    @staticmethod
    def _evaluate(cp: np.ndarray, s: float, t: float) -> np.ndarray:
        """Evaluate a biquadratic Bezier surface at parameter ``(s, t)``.

        ``cp`` has shape ``(3, 3, 3)`` — a 3×3 grid of 3-D control points.
        """
        # Bernstein basis for degree 2.
        bs = np.array(
            [(1 - s) * (1 - s), 2 * s * (1 - s), s * s], dtype=np.float64
        )
        bt = np.array(
            [(1 - t) * (1 - t), 2 * t * (1 - t), t * t], dtype=np.float64
        )

        result = np.zeros(3, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                result += bt[i] * bs[j] * cp[i, j].astype(np.float64)
        return result.astype(np.float32)


# ── Converter: BSP → OBJ ───────────────────────────────────────


class Q3BspToObjConverter:
    """Convert parsed Q3 BSP geometry into a :class:`TriangleMesh`.

    Handles face types 1 (polygon), 2 (patch), and 3 (triangle mesh).
    Type-4 (billboard) faces are skipped — they have no useful solid geometry.

    The output mesh is in *metres* with a Y-up right-hand coordinate system
    (Q3 native is Z-up, so we swap Y↔Z and negate Y→-Y for a right-hand flip).
    """

    def __init__(
        self,
        bsp: Q3BspParser,
        tessellation_level: int = 4,
        convert_to_metres: bool = True,
    ) -> None:
        self.bsp = bsp
        self.tessellator = BezierPatchTessellator(tessellation_level)
        self.convert_to_metres = convert_to_metres

    def convert(self) -> TriangleMesh:
        """Walk all BSP faces and accumulate geometry."""
        mesh = TriangleMesh()

        for face in self.bsp.faces:
            if face.face_type == _FACE_POLYGON:
                self._emit_polygon(face, mesh)
            elif face.face_type == _FACE_PATCH:
                self._emit_patch(face, mesh)
            elif face.face_type == _FACE_MESH:
                self._emit_trimesh(face, mesh)
            # Type 4 (billboard) — skip.

        return mesh

    def _transform(self, pos: np.ndarray) -> np.ndarray:
        """Q3 Z-up → Y-up coordinate swap + optional inch→metre conversion."""
        # Q3: (x, y, z) with Z-up → Y-up: (x, z, -y)
        result = np.array([pos[0], pos[2], -pos[1]], dtype=np.float32)
        if self.convert_to_metres:
            result *= _Q3_UNITS_TO_METRES
        return result

    # ── polygon / triangle mesh faces ───────────────────────────

    def _emit_polygon(self, face: Q3Face, mesh: TriangleMesh) -> None:
        """Type 1: polygon — meshverts define the triangle fan."""
        if face.n_meshverts < 3:
            return
        base = mesh.vertex_count
        for i in range(face.n_vertexes):
            v = self.bsp.vertices[face.vertex_index + i]
            mesh.vertices.append(self._transform(v.position))

        for i in range(0, face.n_meshverts, 3):
            idx0 = self.bsp.meshverts[face.meshvert_index + i]
            idx1 = self.bsp.meshverts[face.meshvert_index + i + 1]
            idx2 = self.bsp.meshverts[face.meshvert_index + i + 2]
            mesh.indices.append((base + idx0, base + idx1, base + idx2))

    def _emit_trimesh(self, face: Q3Face, mesh: TriangleMesh) -> None:
        """Type 3: triangle mesh — identical vertex/meshvert interpretation."""
        self._emit_polygon(face, mesh)

    # ── Bezier patches ──────────────────────────────────────────

    def _emit_patch(self, face: Q3Face, mesh: TriangleMesh) -> None:
        """Type 2: biquadratic Bezier patch."""
        pw, ph = face.patch_size
        if pw < 3 or ph < 3:
            return

        # Read control vertices into a (ph, pw, 3) grid.
        control = np.empty((ph, pw, 3), dtype=np.float32)
        for row in range(ph):
            for col in range(pw):
                v = self.bsp.vertices[face.vertex_index + row * pw + col]
                control[row, col] = self._transform(v.position)

        verts, tris = self.tessellator.tessellate_face(control, pw, ph)

        base = mesh.vertex_count
        for v in verts:
            mesh.vertices.append(v)
        for tri in tris:
            mesh.indices.append((base + int(tri[0]), base + int(tri[1]), base + int(tri[2])))


# ── OBJ Writer ──────────────────────────────────────────────────


def write_obj(mesh: TriangleMesh, path: Path) -> None:
    """Write a :class:`TriangleMesh` as a Wavefront .obj file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Quake 3 BSP → OBJ  ({mesh.vertex_count} verts, "
                 f"{mesh.triangle_count} tris)\n")
        for v in mesh.vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for a, b, c in mesh.indices:
            # OBJ indices are 1-based.
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ── PK3 helpers ─────────────────────────────────────────────────


def extract_bsp_from_pk3(pk3_path: Path) -> list[tuple[str, bytes]]:
    """Extract all ``maps/*.bsp`` entries from a PK3 (ZIP) archive.

    Returns a list of ``(bsp_name, bsp_bytes)`` pairs.
    """
    results: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(pk3_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.startswith("maps/") and lower.endswith(".bsp"):
                bsp_name = Path(name).stem
                results.append((bsp_name, zf.read(name)))
    return results


# ── Spawn export helpers ────────────────────────────────────────


def write_spawns_json(spawns: Sequence[Q3SpawnPoint], path: Path) -> None:
    """Write spawn points as a simple JSON array."""
    import json

    entries = []
    for sp in spawns:
        entries.append(
            {
                "origin": [float(sp.origin[0]), float(sp.origin[1]), float(sp.origin[2])],
                "angle": float(sp.angle),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────


def convert_bsp_file(
    bsp_data: bytes,
    output_path: Path,
    tessellation: int = 4,
    convert_to_metres: bool = True,
    export_spawns: bool = False,
) -> TriangleMesh:
    """Full conversion pipeline: BSP bytes → OBJ file on disk."""
    bsp = Q3BspParser(bsp_data)
    converter = Q3BspToObjConverter(
        bsp,
        tessellation_level=tessellation,
        convert_to_metres=convert_to_metres,
    )
    mesh = converter.convert()
    write_obj(mesh, output_path)

    if export_spawns and bsp.spawns:
        spawn_path = output_path.with_suffix(".spawns.json")
        write_spawns_json(bsp.spawns, spawn_path)

    return mesh


def convert_pk3_file(
    pk3_path: Path,
    output_dir: Path,
    tessellation: int = 4,
    convert_to_metres: bool = True,
    export_spawns: bool = False,
) -> list[Path]:
    """Convert all BSP maps inside a PK3 archive to OBJ files.

    Returns a list of generated .obj file paths.
    """
    bsp_entries = extract_bsp_from_pk3(pk3_path)
    if not bsp_entries:
        print(f"  [warn] No maps/*.bsp found in {pk3_path.name}", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    for bsp_name, bsp_data in bsp_entries:
        obj_path = output_dir / f"{bsp_name}.obj"
        mesh = convert_bsp_file(
            bsp_data,
            obj_path,
            tessellation=tessellation,
            convert_to_metres=convert_to_metres,
            export_spawns=export_spawns,
        )
        results.append(obj_path)
        print(
            f"  [{bsp_name}] {mesh.vertex_count:,} verts, "
            f"{mesh.triangle_count:,} tris → {obj_path.name}"
        )

    return results


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for BSP → OBJ conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Quake 3 BSP/PK3 maps to Wavefront OBJ.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input .bsp or .pk3 file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output .obj file (for BSP) or output directory (for PK3).",
    )
    parser.add_argument(
        "--tessellation",
        "-t",
        type=int,
        default=4,
        help="Bezier patch tessellation level (default: 4).",
    )
    parser.add_argument(
        "--no-metric-conversion",
        action="store_true",
        help="Keep Q3 native units (inches) instead of converting to metres.",
    )
    parser.add_argument(
        "--export-spawns",
        action="store_true",
        help="Also export player spawn points as .spawns.json alongside each OBJ.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    to_metres = not args.no_metric_conversion

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = input_path.suffix.lower()

    if suffix == ".pk3":
        obj_files = convert_pk3_file(
            input_path,
            output_path,
            tessellation=args.tessellation,
            convert_to_metres=to_metres,
            export_spawns=args.export_spawns,
        )
        if obj_files:
            print(f"Converted {len(obj_files)} map(s) from {input_path.name}")
        else:
            print(f"No maps found in {input_path.name}", file=sys.stderr)
            sys.exit(1)

    elif suffix == ".bsp":
        bsp_data = input_path.read_bytes()
        mesh = convert_bsp_file(
            bsp_data,
            output_path,
            tessellation=args.tessellation,
            convert_to_metres=to_metres,
            export_spawns=args.export_spawns,
        )
        print(
            f"Converted {input_path.name}: "
            f"{mesh.vertex_count:,} verts, {mesh.triangle_count:,} tris"
        )

    else:
        print(
            f"Error: unsupported file type '{suffix}'. Expected .bsp or .pk3.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
