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

__all__: list[str] = ["AdapterMetadata", "CommandHoldAdapter", "DatasetAdapter"]


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


# ── CommandHoldAdapter: real-drone bridge interface ─────────────────


@dataclass(frozen=True, slots=True)
class VelocityState:
    """Snapshot of the current velocity command being held.

    Units: m/s (linear) and rad/s (angular).
    """

    forward: float
    """Body-frame forward velocity (m/s).  Positive = forward."""

    vertical: float
    """World-frame vertical velocity (m/s).  Positive = climb."""

    lateral: float
    """Body-frame lateral velocity (m/s).  Positive = strafe right."""

    yaw_rate: float
    """Yaw rotation rate (rad/s).  Positive = counter-clockwise."""


@runtime_checkable
class CommandHoldAdapter(Protocol):
    """Protocol for the real-drone velocity-command bridge.

    The actor sets a desired velocity state (e.g. "go forward at 0.8 m/s,
    yaw left at -0.3 rad/s") and the adapter *holds* that command,
    continuously sending it to the flight controller at the hardware's
    native command rate until a new state is received.

    This is the standard velocity-mode interface used by mainstream
    drone SDKs (DJI Mobile SDK ``VirtualStickControl``, PX4
    ``SET_POSITION_TARGET_LOCAL_NED`` with velocity flags, MAVLink
    ``SET_ATTITUDE_TARGET``).

    The adapter owns:

    * Rate conversion — actor decides at ~10–50 Hz; the adapter
      re-sends the held command to hardware at the SDK's required
      rate (often 50–200 Hz).
    * Safety bounds — clamping, geofence, altitude limits.
    * Smoothing — optional additional low-pass filtering beyond what
      the kinematics layer already provides.
    * Emergency stop — zero all velocities immediately.

    Implementations live in ``environment/backends/`` alongside
    their backend.  They never import from ``actor`` or ``auditor``.

    **No real hardware implementation is provided yet.**  This protocol
    exists to formalise the integration contract so that the training
    action space exactly matches what real hardware will receive.
    """

    @property
    def command_rate_hz(self) -> float:
        """Rate at which velocity commands are forwarded to hardware (Hz)."""
        ...

    @property
    def current_state(self) -> VelocityState:
        """Return the velocity command currently being held."""
        ...

    def set_velocity_state(
        self,
        forward: float,
        vertical: float,
        lateral: float,
        yaw_rate: float,
    ) -> None:
        """Set the desired velocity state.

        The adapter will continuously send this command to hardware
        at ``command_rate_hz`` until ``set_velocity_state`` is called
        again with different values or ``emergency_stop`` is invoked.

        Args:
            forward: Body-frame forward velocity (m/s).
            vertical: World-frame vertical velocity (m/s).
            lateral: Body-frame lateral velocity (m/s).
            yaw_rate: Yaw rotation rate (rad/s).
        """
        ...

    def emergency_stop(self) -> None:
        """Immediately zero all velocity commands.

        This must be callable from any thread and take effect within
        one command cycle (1 / command_rate_hz seconds).
        """
        ...

    def close(self) -> None:
        """Release hardware resources and stop the command loop."""
        ...
