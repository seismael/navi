"""Abstract simulator backend for Ghost-Matrix simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult

__all__: list[str] = ["SimulatorBackend"]


class SimulatorBackend(ABC):
    """Strategy interface for a pluggable simulation backend.

    Implementations produce ``DistanceMatrix`` observations and
    ``StepResult`` step outcomes via a unified ``reset`` / ``step``
    contract.  The current voxel-based pipeline (``VoxelBackend``) and
    future Habitat integration (``HabitatBackend``) both implement
    this interface.

    All backends produce ``DistanceMatrix`` observations with shape
    ``(1, Az, El)`` for depth / semantic / delta_depth / valid_mask
    arrays, where ``matrix_shape = (azimuth_bins, elevation_bins)``.

    External dataset backends (e.g. Habitat) should use a
    ``DatasetAdapter`` internally to convert raw sensor data into
    the canonical DistanceMatrix format.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Reset the environment and return the initial observation.

        Args:
            episode_id: Monotonically increasing episode counter.
            actor_id: Actor index within a multi-actor session.

        Returns:
            First ``DistanceMatrix`` of the new episode.

        """
        ...

    @abstractmethod
    def step(
        self, action: Action, step_id: int, *, actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        """Apply *action* and advance the environment by one tick.

        Args:
            action: Movement command from the policy.
            step_id: Global step counter.
            actor_id: Actor index within a multi-actor session.

        Returns:
            Tuple of ``(observation, result)`` where *observation* is
            the next ``DistanceMatrix`` and *result* carries reward,
            done flags, etc.

        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources (GPU contexts, file handles)."""
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def pose(self) -> RobotPose:
        """Current 6-DOF robot pose."""
        ...

    @property
    @abstractmethod
    def episode_id(self) -> int:
        """Current episode counter."""
        ...
