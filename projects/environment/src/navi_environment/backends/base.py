"""Abstract simulator backend for Ghost-Matrix simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult

__all__: list[str] = ["SimulatorBackend"]


class SimulatorBackend(ABC):
    """Strategy interface for the environment runtime backend.

    Implementations produce ``DistanceMatrix`` observations and
    ``StepResult`` step outcomes via a unified ``reset`` / ``step``
    contract. The canonical runtime currently uses ``SdfDagBackend``.

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

    def perf_snapshot(self) -> object | None:
        """Return an optional coarse performance snapshot for telemetry."""
        return None

    # ------------------------------------------------------------------
    # Batched stepping (default: sequential fallback)
    # ------------------------------------------------------------------

    def batch_step(
        self,
        actions: tuple[Action, ...],
        step_id: int,
    ) -> tuple[tuple[DistanceMatrix, ...], tuple[StepResult, ...]]:
        """Step all actors in one call.

        The default implementation loops over *actions* sequentially,
        calling :meth:`step` for each. Backends that can exploit
        parallelism (e.g. GPU-batched raycasting) should override this
        method.

        Args:
            actions: One ``Action`` per actor, ordered by actor index.
            step_id: Global step counter shared by all actors.

        Returns:
            Tuple of ``(observations, results)`` — each a tuple with
            one element per actor in the same order as *actions*.

        """
        observations: list[DistanceMatrix] = []
        results: list[StepResult] = []
        for idx, action in enumerate(actions):
            actor_id = (
                int(action.env_ids[0])
                if len(action.env_ids) > 0
                else idx
            )
            obs, result = self.step(action, step_id, actor_id=actor_id)
            observations.append(obs)
            results.append(result)
        return tuple(observations), tuple(results)

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
