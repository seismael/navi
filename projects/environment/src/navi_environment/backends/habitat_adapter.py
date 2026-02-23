"""HabitatAdapter — converts Habitat equirectangular sensor output to canonical arrays.

This adapter implements the ``DatasetAdapter`` protocol, transforming
raw Habitat observations into Navi's ``(1, Az, El)`` canonical shape.

Responsibilities:

* Axis transpose: Habitat's ``(El, Az)`` → Navi's ``(Az, El)``
* Depth normalisation: metres → ``[0, 1]`` via ``max_distance``
* Semantic remapping: Habitat instance IDs → Navi ``[0, 10]``
* Delta-depth computation (frame differencing)
* Valid-mask computation
* Overhead minimap generation from depth
* Leading env dimension: ``(Az, El)`` → ``(1, Az, El)``
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from navi_environment.backends.adapter import AdapterMetadata
from navi_environment.backends.habitat_semantic_lut import HabitatSemanticLUT

__all__: list[str] = ["HabitatAdapter"]


class HabitatAdapter:
    """Converts raw Habitat equirectangular observations to canonical arrays.

    Satisfies the ``DatasetAdapter`` protocol without inheriting from it
    (structural subtyping via ``@runtime_checkable``).
    """

    def __init__(
        self,
        az_bins: int,
        el_bins: int,
        max_distance: float,
        semantic_lut: HabitatSemanticLUT,
    ) -> None:
        self._az_bins = az_bins
        self._el_bins = el_bins
        self._max_distance = max_distance
        self._semantic_lut = semantic_lut
        self._prev_depth: NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # DatasetAdapter protocol
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> AdapterMetadata:
        """Return static metadata about this adapter's output."""
        return AdapterMetadata(
            azimuth_bins=self._az_bins,
            elevation_bins=self._el_bins,
            max_distance=self._max_distance,
            semantic_classes=11,  # Navi uses 0-10
        )

    def adapt(
        self,
        raw_obs: dict[str, Any],
        step_id: int,
    ) -> dict[str, NDArray[Any]]:
        """Convert Habitat sensor observations to canonical arrays.

        The equirectangular depth sensor produces ``(El, Az)`` float32 depth
        in metres.  We transpose to ``(Az, El)``, normalise to ``[0, 1]``,
        compute delta-depth, remap semantics, and add the leading env dim.

        Returns:
            Dictionary with keys ``depth``, ``delta_depth``, ``semantic``,
            ``valid_mask``, ``overhead`` — all in canonical shapes.
        """
        # ----- Depth: (El, Az) → (Az, El) → (1, Az, El) -----
        raw_depth = np.asarray(raw_obs["equirect_depth"], dtype=np.float32)
        if raw_depth.ndim == 3:
            raw_depth = raw_depth.squeeze(-1)  # (El, Az, 1) → (El, Az)
        raw_depth = raw_depth.T  # (Az, El)

        depth_norm = np.clip(raw_depth, 0.0, self._max_distance) / self._max_distance

        # ----- Delta-depth -----
        if self._prev_depth is not None:
            delta_depth = depth_norm - self._prev_depth
        else:
            delta_depth = np.zeros_like(depth_norm)
        self._prev_depth = depth_norm.copy()

        # ----- Semantic: (El, Az) → (Az, El) → (1, Az, El) -----
        raw_semantic = np.asarray(raw_obs["equirect_semantic"])
        if raw_semantic.ndim == 3:
            raw_semantic = raw_semantic.squeeze(-1)
        semantic = self._semantic_lut.remap(raw_semantic.astype(np.int32))
        semantic = semantic.T  # (Az, El)

        # ----- Valid mask -----
        valid_mask = raw_depth > 0.0  # (Az, El)

        # ----- Overhead minimap -----
        overhead = self._build_overhead(raw_depth)

        _ = step_id  # reserved for future use
        return {
            "depth": depth_norm[np.newaxis, ...].astype(np.float32),
            "delta_depth": delta_depth[np.newaxis, ...].astype(np.float32),
            "semantic": semantic[np.newaxis, ...].astype(np.int32),
            "valid_mask": valid_mask[np.newaxis, ...],
            "overhead": overhead,
        }

    def reset(self) -> None:
        """Reset frame-difference state."""
        self._prev_depth = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_overhead(self, raw_depth: NDArray[np.float32]) -> NDArray[np.float32]:
        """Build a top-down minimap from ``(Az, El)`` depth.

        Uses minimum depth across ALL elevation bins per azimuth ray
        and projects onto a 256x256 grid with filled cells and a
        Turbo-style distance colormap.
        """
        size = 256
        overhead = np.zeros((size, size, 3), dtype=np.float32)

        # Minimum depth across all elevations per azimuth
        band = np.where(raw_depth > 0, raw_depth, np.inf).min(axis=1)  # (az_bins,)
        finite = np.isfinite(band)
        if not np.any(finite):
            return overhead

        azimuths = np.linspace(0, 2 * math.pi, len(band), endpoint=False)
        xs = band * np.cos(azimuths)
        zs = band * np.sin(azimuths)

        scale = size / (2 * self._max_distance)
        px = (xs * scale + size / 2).astype(np.int32)
        pz = (zs * scale + size / 2).astype(np.int32)

        # Keep only finite and in-bounds points
        mask = finite & (px >= 1) & (px < size - 1) & (pz >= 1) & (pz < size - 1)
        px = px[mask]
        pz = pz[mask]
        depths = band[mask]

        # Turbo-style colormap: near = warm, far = cool (float [0, 1] BGR)
        depth_norm = np.clip(depths / self._max_distance, 0.0, 1.0)
        t = 1.0 - depth_norm  # invert: near → 1 (red), far → 0 (blue)

        # Simple Turbo approximation: 5-stop gradient
        r = np.clip(np.where(t > 0.5, 1.0, t * 2.0), 0.0, 1.0)
        g = np.clip(np.where(t < 0.25, t * 4.0, np.where(t > 0.75, (1.0 - t) * 4.0, 1.0)), 0.0, 1.0)
        b = np.clip(np.where(t < 0.5, (0.5 - t) * 2.0, 0.0), 0.0, 1.0)

        # Draw 3x3 filled blocks for visibility
        for i in range(len(px)):
            overhead[pz[i] - 1: pz[i] + 2, px[i] - 1: px[i] + 2, 0] = b[i]
            overhead[pz[i] - 1: pz[i] + 2, px[i] - 1: px[i] + 2, 1] = g[i]
            overhead[pz[i] - 1: pz[i] + 2, px[i] - 1: px[i] + 2, 2] = r[i]

        # Robot marker — bright cyan with heading
        c = size // 2
        overhead[c - 2: c + 3, c - 2: c + 3, :] = 0.0
        overhead[c - 2: c + 3, c - 2: c + 3, 1] = 1.0  # green channel = cyan
        overhead[c - 2: c + 3, c - 2: c + 3, 2] = 1.0  # red channel = cyan

        return overhead
