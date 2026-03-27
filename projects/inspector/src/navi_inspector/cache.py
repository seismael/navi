"""Disk cache for extracted meshes.

Caches exported PLY files keyed by a hash of the .gmdag file path,
resolution, and isosurface level.  Invalidation is based on mtime
comparison: if the source .gmdag is newer than the cached PLY, the
cache entry is stale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyvista as pv

__all__: list[str] = ["MeshCache"]


class MeshCache:
    """File-system cache for extracted meshes."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir).resolve()

    def _cache_key(self, gmdag_path: Path, resolution: int, level: float | None) -> str:
        canon = str(gmdag_path.resolve())
        level_str = "auto" if level is None else f"{level:.6f}"
        raw = f"{canon}::{resolution}::{level_str}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        level_label = "Lauto" if level is None else f"L{level:.2f}"
        return f"{gmdag_path.stem}_{resolution}_{level_label}_{digest}"

    def _cache_path(self, gmdag_path: Path, resolution: int, level: float | None) -> Path:
        key = self._cache_key(gmdag_path, resolution, level)
        return self._dir / f"{key}.ply"

    def get(
        self, gmdag_path: Path, resolution: int, level: float | None
    ) -> pv.PolyData | None:
        """Return cached mesh if it exists and is fresh, else None."""
        cp = self._cache_path(gmdag_path, resolution, level)
        if not cp.exists():
            return None
        src = Path(gmdag_path).resolve()
        if src.exists() and src.stat().st_mtime > cp.stat().st_mtime:
            # Source is newer — stale cache
            return None
        return pv.read(str(cp))

    def put(
        self,
        gmdag_path: Path,
        resolution: int,
        level: float | None,
        mesh: pv.PolyData,
    ) -> Path:
        """Store mesh in cache. Returns the cache file path."""
        cp = self._cache_path(gmdag_path, resolution, level)
        cp.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(str(cp))
        return cp
