"""Tests for the Q3 BSP → OBJ converter module.

Validates header parsing, vertex extraction, face type dispatch,
Bezier patch tessellation, OBJ output format, and PK3 extraction.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from voxel_dag.bsp_to_obj import (
    BezierPatchTessellator,
    Q3BspParser,
    Q3BspToObjConverter,
    TriangleMesh,
    convert_bsp_file,
    extract_bsp_from_pk3,
    write_obj,
)

# ── Helpers for building synthetic BSP data ─────────────────────


def _build_header(
    version: int = 0x2E,
    lump_entries: list[tuple[int, int]] | None = None,
) -> bytearray:
    """Build a minimal IBSP header with 17 lump directory entries."""
    buf = bytearray()
    buf += b"IBSP"
    buf += struct.pack("<i", version)
    if lump_entries is None:
        # 17 zeroed lump entries
        lump_entries = [(0, 0)] * 17
    for offset, length in lump_entries:
        buf += struct.pack("<ii", offset, length)
    return buf


_HEADER_SIZE = 4 + 4 + 17 * 8  # 144 bytes


def _build_vertex(x: float, y: float, z: float, nx: float = 0.0, ny: float = 0.0, nz: float = 1.0) -> bytes:
    """Build a single 44-byte Q3 vertex record."""
    buf = struct.pack("<fff", x, y, z)      # position (12)
    buf += struct.pack("<ff", 0.0, 0.0)     # texcoord[0] (8)
    buf += struct.pack("<ff", 0.0, 0.0)     # texcoord[1] (8)
    buf += struct.pack("<fff", nx, ny, nz)  # normal (12)
    buf += struct.pack("<BBBB", 255, 255, 255, 255)  # color (4)
    assert len(buf) == 44  # noqa: S101
    return buf


def _build_meshvert(offset: int) -> bytes:
    return struct.pack("<i", offset)


def _build_face(
    face_type: int = 1,
    vertex_index: int = 0,
    n_vertexes: int = 3,
    meshvert_index: int = 0,
    n_meshverts: int = 3,
    patch_width: int = 0,
    patch_height: int = 0,
) -> bytes:
    """Build a single 104-byte Q3 face record."""
    buf = struct.pack("<iii", 0, 0, face_type)       # texture, effect, type (12)
    buf += struct.pack("<ii", vertex_index, n_vertexes)  # (8)
    buf += struct.pack("<ii", meshvert_index, n_meshverts)  # (8)
    buf += b"\x00" * 68  # lm_index + lm_start + lm_size + lm_origin + lm_vecs + normal (68)
    buf += struct.pack("<ii", patch_width, patch_height)  # size (8)
    assert len(buf) == 104  # noqa: S101
    return buf


def _build_entity_string(entities: list[dict[str, str]]) -> bytes:
    """Build a Q3 entities lump string from a list of key-value dicts."""
    lines: list[str] = []
    for ent in entities:
        lines.append("{")
        for k, v in ent.items():
            lines.append(f'"{k}" "{v}"')
        lines.append("}")
    text = "\n".join(lines) + "\0"
    return text.encode("ascii")


def _build_minimal_bsp(
    vertices: list[tuple[float, float, float]] | None = None,
    meshverts: list[int] | None = None,
    faces: list[dict] | None = None,
    entities: list[dict[str, str]] | None = None,
) -> bytes:
    """Construct a complete minimal synthetic BSP for testing."""
    # Build the per-lump data
    vert_data = b""
    if vertices:
        for v in vertices:
            vert_data += _build_vertex(*v)

    mv_data = b""
    if meshverts:
        for mv in meshverts:
            mv_data += _build_meshvert(mv)

    face_data = b""
    if faces:
        for f in faces:
            face_data += _build_face(**f)

    entity_data = b""
    if entities:
        entity_data = _build_entity_string(entities)

    # Compute offsets — lumps are packed right after the header
    lumps = [(0, 0)] * 17
    offset = _HEADER_SIZE

    # Lump 0: entities
    lumps[0] = (offset, len(entity_data))
    offset += len(entity_data)

    # Lump 10: vertexes
    lumps[10] = (offset, len(vert_data))
    offset += len(vert_data)

    # Lump 11: meshverts
    lumps[11] = (offset, len(mv_data))
    offset += len(mv_data)

    # Lump 13: faces
    lumps[13] = (offset, len(face_data))
    offset += len(face_data)

    header = _build_header(lump_entries=lumps)
    return bytes(header) + entity_data + vert_data + mv_data + face_data


# ── Header validation tests ─────────────────────────────────────


class TestQ3BspParserHeader:
    def test_valid_header(self) -> None:
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0), (100, 0, 0), (0, 100, 0)],
            meshverts=[0, 1, 2],
            faces=[{"face_type": 1, "n_vertexes": 3, "n_meshverts": 3}],
        )
        parser = Q3BspParser(bsp_data)
        assert len(parser.vertices) == 3
        assert len(parser.meshverts) == 3
        assert len(parser.faces) == 1

    def test_invalid_magic(self) -> None:
        bad_data = b"XBSP" + b"\x00" * 200
        with pytest.raises(ValueError, match="Invalid IBSP magic"):
            Q3BspParser(bad_data)

    def test_wrong_version(self) -> None:
        data = bytearray(b"IBSP")
        data += struct.pack("<i", 0x30)  # Wrong version
        data += b"\x00" * 200
        with pytest.raises(ValueError, match="Unsupported IBSP version"):
            Q3BspParser(bytes(data))

    def test_truncated_header(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            Q3BspParser(b"IBSP\x2e\x00\x00\x00")


# ── Vertex/face parsing tests ───────────────────────────────────


class TestQ3BspParserGeometry:
    def test_vertex_positions(self) -> None:
        bsp_data = _build_minimal_bsp(
            vertices=[(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)],
            meshverts=[],
            faces=[],
        )
        parser = Q3BspParser(bsp_data)
        assert len(parser.vertices) == 2
        np.testing.assert_allclose(parser.vertices[0].position, [10, 20, 30], atol=1e-5)
        np.testing.assert_allclose(parser.vertices[1].position, [40, 50, 60], atol=1e-5)

    def test_face_type_dispatch(self) -> None:
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0)] * 9,
            meshverts=[0, 1, 2],
            faces=[
                {"face_type": 1, "n_vertexes": 3, "n_meshverts": 3},
                {"face_type": 2, "n_vertexes": 9, "patch_width": 3, "patch_height": 3},
                {"face_type": 3, "n_vertexes": 3, "n_meshverts": 3},
                {"face_type": 4, "n_vertexes": 0, "n_meshverts": 0},
            ],
        )
        parser = Q3BspParser(bsp_data)
        assert len(parser.faces) == 4
        assert parser.faces[0].face_type == 1
        assert parser.faces[1].face_type == 2
        assert parser.faces[2].face_type == 3
        assert parser.faces[3].face_type == 4

    def test_meshvert_offsets(self) -> None:
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0)] * 3,
            meshverts=[0, 2, 1],
            faces=[],
        )
        parser = Q3BspParser(bsp_data)
        assert parser.meshverts == [0, 2, 1]


# ── Entity (spawn) parsing tests ────────────────────────────────


class TestQ3BspParserEntities:
    def test_spawn_extraction(self) -> None:
        entities = [
            {"classname": "info_player_deathmatch", "origin": "100 200 300", "angle": "90"},
            {"classname": "info_player_start", "origin": "0 0 0"},
            {"classname": "light", "origin": "50 50 50"},  # Should be ignored
        ]
        bsp_data = _build_minimal_bsp(entities=entities)
        parser = Q3BspParser(bsp_data)
        # Only deathmatch and player_start are extracted
        assert len(parser.spawns) == 2

    def test_spawn_coordinate_transform(self) -> None:
        """Verify Q3 Z-up inches → Navi Y-up metres conversion."""
        entities = [
            {"classname": "info_player_deathmatch", "origin": "100 200 300"},
        ]
        bsp_data = _build_minimal_bsp(entities=entities)
        parser = Q3BspParser(bsp_data)
        assert len(parser.spawns) == 1
        sp = parser.spawns[0]
        # Q3(x,y,z) → Navi(x,z,-y) * 0.0254
        expected_x = 100 * 0.0254
        expected_y = 300 * 0.0254  # z becomes y
        expected_z = -200 * 0.0254  # -y becomes z
        np.testing.assert_allclose(sp.origin, [expected_x, expected_y, expected_z], atol=1e-4)


# ── Bezier patch tessellation tests ─────────────────────────────


class TestBezierPatchTessellator:
    def test_flat_patch_preserves_plane(self) -> None:
        """A flat control grid should tessellate into coplanar vertices."""
        tess = BezierPatchTessellator(tessellation_level=2)
        # Flat grid on XZ plane at y=0
        cp = np.array([
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
            [[0, 0, 2], [1, 0, 2], [2, 0, 2]],
        ], dtype=np.float32)
        verts, tris = tess.tessellate_face(cp, 3, 3)
        # All Y coords should be 0
        np.testing.assert_allclose(verts[:, 1], 0.0, atol=1e-5)
        # Vertex count: (level+1)^2 = 9
        assert len(verts) == 9
        # Triangle count: 2 * level^2 = 8
        assert len(tris) == 8

    def test_corner_vertices_match_control_points(self) -> None:
        """The four corner vertices should match the corner control points."""
        tess = BezierPatchTessellator(tessellation_level=4)
        cp = np.array([
            [[0, 0, 0], [5, 0, 0], [10, 0, 0]],
            [[0, 5, 0], [5, 5, 5], [10, 5, 0]],
            [[0, 10, 0], [5, 10, 0], [10, 10, 0]],
        ], dtype=np.float32)
        verts, _ = tess.tessellate_face(cp, 3, 3)
        L = 4
        # Corner (0,0) → row 0, col 0
        np.testing.assert_allclose(verts[0], cp[0, 0], atol=1e-5)
        # Corner (1,0) → row 0, col L
        np.testing.assert_allclose(verts[L], cp[0, 2], atol=1e-5)
        # Corner (0,1) → row L, col 0
        np.testing.assert_allclose(verts[L * (L + 1)], cp[2, 0], atol=1e-5)
        # Corner (1,1) → row L, col L
        np.testing.assert_allclose(verts[L * (L + 1) + L], cp[2, 2], atol=1e-5)

    def test_curved_patch_midpoint(self) -> None:
        """The center of a curved patch should be pulled toward the center control point."""
        tess = BezierPatchTessellator(tessellation_level=4)
        # Dome-like: center CP raised to y=10 while corners at y=0
        cp = np.array([
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 1], [1, 10, 1], [2, 0, 1]],
            [[0, 0, 2], [1, 0, 2], [2, 0, 2]],
        ], dtype=np.float32)
        verts, _ = tess.tessellate_face(cp, 3, 3)
        L = 4
        # Center vertex at (s=0.5, t=0.5) → row L//2, col L//2
        mid_idx = (L // 2) * (L + 1) + (L // 2)
        # With biquadratic Bezier, the mid at (0.5, 0.5) evaluates to y = 2.5
        # because B(0.5)·B(0.5)·10 = 0.5·0.5·10 = 2.5 for the center term
        assert verts[mid_idx][1] > 0.0, "Midpoint should be elevated"

    def test_invalid_tessellation_level(self) -> None:
        with pytest.raises(ValueError, match="Tessellation level must be >= 1"):
            BezierPatchTessellator(tessellation_level=0)


# ── Converter tests ─────────────────────────────────────────────


class TestQ3BspToObjConverter:
    def test_polygon_face_conversion(self) -> None:
        """Type-1 polygon face produces correct triangle count."""
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0), (100, 0, 0), (0, 0, 100), (100, 0, 100)],
            meshverts=[0, 1, 2, 1, 3, 2],
            faces=[{"face_type": 1, "n_vertexes": 4, "n_meshverts": 6}],
        )
        parser = Q3BspParser(bsp_data)
        converter = Q3BspToObjConverter(parser, convert_to_metres=False)
        mesh = converter.convert()
        assert mesh.vertex_count == 4
        assert mesh.triangle_count == 2

    def test_billboard_face_skipped(self) -> None:
        """Type-4 billboard faces produce no geometry."""
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0)],
            meshverts=[],
            faces=[{"face_type": 4, "n_vertexes": 1, "n_meshverts": 0}],
        )
        parser = Q3BspParser(bsp_data)
        converter = Q3BspToObjConverter(parser, convert_to_metres=False)
        mesh = converter.convert()
        assert mesh.vertex_count == 0
        assert mesh.triangle_count == 0

    def test_coordinate_transform_zup_to_yup(self) -> None:
        """Verify Z-up → Y-up coordinate swap without metre conversion."""
        bsp_data = _build_minimal_bsp(
            vertices=[(10, 20, 30)],
            meshverts=[],
            faces=[],
        )
        parser = Q3BspParser(bsp_data)
        converter = Q3BspToObjConverter(parser, convert_to_metres=False)
        # Access the transform directly
        pos = parser.vertices[0].position
        transformed = converter._transform(pos)
        # Q3 (x,y,z) → Navi (x, z, -y)
        np.testing.assert_allclose(transformed, [10, 30, -20], atol=1e-5)

    def test_metre_conversion(self) -> None:
        """Verify inch → metre scaling."""
        bsp_data = _build_minimal_bsp(
            vertices=[(100, 0, 0)],
            meshverts=[],
            faces=[],
        )
        parser = Q3BspParser(bsp_data)
        converter = Q3BspToObjConverter(parser, convert_to_metres=True)
        pos = parser.vertices[0].position
        transformed = converter._transform(pos)
        np.testing.assert_allclose(transformed[0], 100 * 0.0254, atol=1e-5)


# ── OBJ writer tests ────────────────────────────────────────────


class TestObjWriter:
    def test_obj_format(self, tmp_path: Path) -> None:
        mesh = TriangleMesh()
        mesh.vertices = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        mesh.indices = [(0, 1, 2)]

        obj_path = tmp_path / "test.obj"
        write_obj(mesh, obj_path)

        text = obj_path.read_text()
        lines = [l for l in text.splitlines() if not l.startswith("#")]
        assert lines[0] == "v 1.000000 2.000000 3.000000"
        assert lines[1] == "v 4.000000 5.000000 6.000000"
        assert lines[2] == "v 7.000000 8.000000 9.000000"
        # OBJ is 1-based
        assert lines[3] == "f 1 2 3"


# ── PK3 extraction tests ────────────────────────────────────────


class TestPk3Extraction:
    def test_extract_bsp_from_pk3(self, tmp_path: Path) -> None:
        """Create a synthetic PK3 with a maps/*.bsp entry and extract it."""
        pk3_path = tmp_path / "test.pk3"
        # A minimal valid BSP header (just header, no geometry)
        bsp_bytes = _build_minimal_bsp()

        with zipfile.ZipFile(pk3_path, "w") as zf:
            zf.writestr("maps/test_map.bsp", bsp_bytes)
            zf.writestr("textures/notabsp.txt", "ignored")

        results = extract_bsp_from_pk3(pk3_path)
        assert len(results) == 1
        name, data = results[0]
        assert name == "test_map"
        assert data[:4] == b"IBSP"

    def test_pk3_no_maps(self, tmp_path: Path) -> None:
        """PK3 with no maps/ directory returns empty list."""
        pk3_path = tmp_path / "empty.pk3"
        with zipfile.ZipFile(pk3_path, "w") as zf:
            zf.writestr("textures/foo.txt", "nothing")

        results = extract_bsp_from_pk3(pk3_path)
        assert results == []


# ── Full pipeline test ───────────────────────────────────────────


class TestConvertBspFile:
    def test_convert_synthetic_bsp(self, tmp_path: Path) -> None:
        """End-to-end: synthetic BSP → OBJ file with correct content."""
        bsp_data = _build_minimal_bsp(
            vertices=[(0, 0, 0), (100, 0, 0), (0, 100, 0)],
            meshverts=[0, 1, 2],
            faces=[{"face_type": 1, "n_vertexes": 3, "n_meshverts": 3}],
            entities=[
                {"classname": "info_player_deathmatch", "origin": "50 50 50", "angle": "0"},
            ],
        )
        obj_path = tmp_path / "test.obj"
        mesh = convert_bsp_file(
            bsp_data, obj_path, tessellation=2, convert_to_metres=True, export_spawns=True,
        )
        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert obj_path.exists()

        # Check spawn file
        spawn_path = obj_path.with_suffix(".spawns.json")
        assert spawn_path.exists()
        spawns = json.loads(spawn_path.read_text())
        assert len(spawns) == 1
        assert len(spawns[0]["origin"]) == 3
