"""DatasetAdapter protocol — formalizes the boundary between external data sources and
the training engine's canonical DistanceMatrix contract.

Any external simulator / dataset (Habitat, iGibson, Matterport, etc.) provides
raw observations in its own format.  A ``DatasetAdapter`` converts those raw
observations into the canonical shape that the actor training engine expects:

    depth / semantic / delta_depth / valid_mask:  ``(1, Az, El)``
    matrix_shape:  ``(azimuth_bins, elevation_bins)``
    depth range:   ``[0, 1]``  (normalized by ``max_distance``)
    semantic:      ``int32`` class IDs in ``[0, N_CLASSES)``

The training engine (actor) is **never** modified to accommodate an adapter —
adapters always transform **to** the engine's format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray

__all__: list[str] = ["AdapterMetadata", "DatasetAdapter"]


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    """Static metadata describing an adapter's output contract.

    Used by the backend to configure its sensor resolution and
    validate that the adapter's output matches expectations.
    """

    azimuth_bins: int
    """Number of azimuth bins in the output spherical grid."""

    elevation_bins: int
    """Number of elevation bins in the output spherical grid."""

    max_distance: float
    """Maximum observation distance in metres (used for depth normalisation)."""

    semantic_classes: int
    """Number of semantic classes produced by the adapter (0..N-1)."""


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol that formalises the external-data → DistanceMatrix boundary.

    Implementations convert raw sensor observations from an external
    simulator into the canonical ``(Az, El)`` arrays that get packed
    into a ``DistanceMatrix``.  The adapter is responsible for:

    * Axis ordering — raw ``(El, Az)`` → canonical ``(Az, El)``
    * Depth normalisation — metres → ``[0, 1]`` via ``max_distance``
    * Semantic remapping — external IDs → Navi's ``[0, N_CLASSES)``
    * Valid-mask computation
    * Delta-depth computation (frame differencing)

    Backends (e.g. ``HabitatBackend``) own the adapter and call
    ``adapt()`` on every raw observation dict.
    """

    @property
    def metadata(self) -> AdapterMetadata:
        """Return static metadata about this adapter's output."""
        ...

    def adapt(
        self,
        raw_obs: dict[str, Any],
        step_id: int,
    ) -> dict[str, NDArray[Any]]:
        """Convert raw external observations to canonical arrays.

        Args:
            raw_obs: Observation dictionary from the external simulator
                     (e.g. ``{\"equirect_depth\": ..., \"equirect_semantic\": ...}``).
            step_id: Current step counter (passed through for metadata).

        Returns:
            Dictionary with the following canonical keys:

            * ``depth``       — ``(1, Az, El)`` float32 in ``[0, 1]``
            * ``delta_depth`` — ``(1, Az, El)`` float32
            * ``semantic``    — ``(1, Az, El)`` int32
            * ``valid_mask``  — ``(1, Az, El)`` bool
            * ``overhead``    — ``(H, W, 3)`` float32 minimap

        """
        ...

    def reset(self) -> None:
        """Reset adapter state (e.g. previous-frame buffers)."""
        ...
