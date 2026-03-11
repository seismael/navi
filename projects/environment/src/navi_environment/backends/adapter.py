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

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from navi_contracts import DistanceMatrix, RobotPose

__all__: list[str] = [
    "AdapterMetadata",
    "CommandHoldAdapter",
    "DatasetAdapter",
    "EquirectangularDatasetAdapter",
    "RigidTransformSpec",
    "apply_rigid_transform",
    "compute_delta_depth",
    "habitat_camera_transform_spec",
    "materialize_distance_matrix",
    "normalize_depth_metres",
    "remap_semantic_ids",
    "transpose_equirectangular_grid",
]

_VALID_HANDEDNESS = {"left-handed", "right-handed"}
_VALID_FORWARD_AXES = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


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


@dataclass(frozen=True, slots=True)
class RigidTransformSpec:
    """Explicit rigid transform metadata for dataset adapters."""

    name: str
    matrix: NDArray[np.float32]
    handedness: str
    source_forward_axis: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            msg = f"transform matrix for {self.name} must have shape (4, 4); got {matrix.shape}"
            raise ValueError(msg)
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6, rtol=0.0):
            msg = f"transform matrix for {self.name} must end with homogeneous row [0, 0, 0, 1]"
            raise ValueError(msg)
        if self.handedness not in _VALID_HANDEDNESS:
            msg = f"handedness for {self.name} must be one of {sorted(_VALID_HANDEDNESS)}; got {self.handedness!r}"
            raise ValueError(msg)
        if self.source_forward_axis not in _VALID_FORWARD_AXES:
            msg = (
                f"source_forward_axis for {self.name} must be one of {sorted(_VALID_FORWARD_AXES)}; "
                f"got {self.source_forward_axis!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "matrix", matrix)


def habitat_camera_transform_spec() -> RigidTransformSpec:
    """Return the canonical Habitat camera transform into Navi coordinates.

    Habitat camera/world coordinates are y-up, right-handed, with forward along
    `-Z`, which matches Navi's current canonical local convention. The adapter
    transform is therefore explicit identity rather than an implied assumption.
    """
    return RigidTransformSpec(
        name="habitat_camera_y_up_identity",
        matrix=np.eye(4, dtype=np.float32),
        handedness="right-handed",
        source_forward_axis="-Z",
    )


@dataclass(slots=True)
class EquirectangularDatasetAdapter:
    """Concrete adapter for raw equirectangular depth and semantic grids.

    The adapter expects raw observations with:

    - ``equirect_depth``: depth in metres shaped ``(El, Az)``
    - ``equirect_semantic``: semantic ids shaped ``(El, Az)``
    - optional ``overhead``: float image shaped ``(H, W, 3)``

    The adapter keeps the previous normalized frame so delta-depth remains an
    adapter-side concern rather than leaking into the actor or backend.
    """

    azimuth_bins: int
    elevation_bins: int
    max_distance: float
    semantic_remap: dict[int, int]
    transform_spec: RigidTransformSpec
    unknown_semantic_id: int = 0
    depth_key: str = "equirect_depth"
    semantic_key: str = "equirect_semantic"
    overhead_key: str = "overhead"
    default_overhead_shape: tuple[int, int, int] = (256, 256, 3)
    _previous_depth: NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _metadata: AdapterMetadata = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.azimuth_bins <= 0:
            msg = f"azimuth_bins must be positive; got {self.azimuth_bins}"
            raise ValueError(msg)
        if self.elevation_bins <= 0:
            msg = f"elevation_bins must be positive; got {self.elevation_bins}"
            raise ValueError(msg)
        if self.max_distance <= 0.0:
            msg = f"max_distance must be positive; got {self.max_distance}"
            raise ValueError(msg)
        if len(self.default_overhead_shape) != 3 or self.default_overhead_shape[2] != 3:
            msg = (
                "default_overhead_shape must describe an image with three color channels; "
                f"got {self.default_overhead_shape}"
            )
            raise ValueError(msg)
        if any(int(dim) <= 0 for dim in self.default_overhead_shape):
            msg = f"default_overhead_shape must be strictly positive; got {self.default_overhead_shape}"
            raise ValueError(msg)

        canonical_ids = {int(self.unknown_semantic_id), *(int(value) for value in self.semantic_remap.values())}
        if min(canonical_ids) < 0:
            msg = f"semantic remap must produce non-negative canonical ids; got {sorted(canonical_ids)}"
            raise ValueError(msg)
        expected_ids = set(range(max(canonical_ids) + 1))
        if canonical_ids != expected_ids:
            msg = (
                "semantic remap must produce contiguous canonical ids starting at 0; "
                f"got {sorted(canonical_ids)}"
            )
            raise ValueError(msg)

        self._metadata = AdapterMetadata(
            azimuth_bins=self.azimuth_bins,
            elevation_bins=self.elevation_bins,
            max_distance=self.max_distance,
            semantic_classes=max(canonical_ids) + 1,
        )

    @property
    def metadata(self) -> AdapterMetadata:
        """Return static metadata about this adapter's canonical output."""
        return self._metadata

    def adapt(
        self,
        raw_obs: dict[str, Any],
        step_id: int,
    ) -> dict[str, NDArray[Any]]:
        """Convert raw equirectangular grids into canonical observation arrays."""
        del step_id
        depth_raw = self._require_rank2_grid(raw_obs, self.depth_key)
        semantic_raw = self._require_rank2_grid(raw_obs, self.semantic_key)
        if depth_raw.shape != semantic_raw.shape:
            msg = (
                f"{self.depth_key} and {self.semantic_key} must share the same (El, Az) shape; "
                f"got {depth_raw.shape} and {semantic_raw.shape}"
            )
            raise ValueError(msg)

        depth_norm_2d, valid_2d = normalize_depth_metres(depth_raw, max_distance=self.max_distance)
        semantic_2d = remap_semantic_ids(
            semantic_raw,
            self.semantic_remap,
            unknown_id=self.unknown_semantic_id,
        )

        depth = transpose_equirectangular_grid(depth_norm_2d).astype(np.float32, copy=False)
        valid_mask = transpose_equirectangular_grid(valid_2d).astype(np.bool_, copy=False)
        semantic = transpose_equirectangular_grid(semantic_2d).astype(np.int32, copy=False)
        self._validate_output_shape(depth, name="depth")
        self._validate_output_shape(valid_mask, name="valid_mask")
        self._validate_output_shape(semantic, name="semantic")

        delta_depth = compute_delta_depth(depth, self._previous_depth)
        overhead = self._resolve_overhead(raw_obs)
        self._previous_depth = depth.copy()
        return {
            "depth": depth,
            "delta_depth": delta_depth,
            "semantic": semantic,
            "valid_mask": valid_mask,
            "overhead": overhead,
        }

    def adapt_distance_matrix(
        self,
        raw_obs: dict[str, Any],
        *,
        episode_id: int,
        env_id: int,
        robot_pose: RobotPose,
        step_id: int,
        timestamp: float | None = None,
    ) -> DistanceMatrix:
        """Adapt raw observations directly into the public DistanceMatrix contract."""
        adapted = self.adapt(raw_obs, step_id=step_id)
        return materialize_distance_matrix(
            episode_id=episode_id,
            env_id=env_id,
            depth=adapted["depth"],
            delta_depth=adapted["delta_depth"],
            semantic=adapted["semantic"],
            valid_mask=adapted["valid_mask"],
            robot_pose=robot_pose,
            step_id=step_id,
            overhead=adapted["overhead"],
            timestamp=timestamp,
        )

    def reset(self) -> None:
        """Clear any frame-to-frame adapter state."""
        self._previous_depth = None

    def transform_points(self, points_xyz: NDArray[Any]) -> NDArray[np.float32]:
        """Apply the adapter's explicit rigid transform to raw points."""
        return apply_rigid_transform(points_xyz, self.transform_spec)

    def _require_rank2_grid(self, raw_obs: dict[str, Any], key: str) -> NDArray[Any]:
        if key not in raw_obs:
            msg = f"raw observation is missing required key {key!r}"
            raise KeyError(msg)
        value = np.asarray(raw_obs[key])
        if value.ndim != 2:
            msg = f"raw observation key {key!r} must be a rank-2 (El, Az) grid; got shape {value.shape}"
            raise ValueError(msg)
        return value

    def _validate_output_shape(self, values: NDArray[Any], *, name: str) -> None:
        expected = (1, self.azimuth_bins, self.elevation_bins)
        if values.shape != expected:
            msg = f"{name} must have canonical shape {expected}; got {values.shape}"
            raise ValueError(msg)

    def _resolve_overhead(self, raw_obs: dict[str, Any]) -> NDArray[np.float32]:
        if self.overhead_key not in raw_obs:
            return np.zeros(self.default_overhead_shape, dtype=np.float32)
        overhead = np.asarray(raw_obs[self.overhead_key], dtype=np.float32)
        if overhead.ndim != 3 or overhead.shape[2] != 3:
            msg = (
                f"raw observation key {self.overhead_key!r} must be an image shaped (H, W, 3); "
                f"got {overhead.shape}"
            )
            raise ValueError(msg)
        return overhead.astype(np.float32, copy=False)


def transpose_equirectangular_grid(values: NDArray[Any]) -> NDArray[Any]:
    """Convert raw `(El, Az)` arrays into canonical `(1, Az, El)` layout."""
    array = np.asarray(values)
    if array.ndim != 2:
        msg = f"equirectangular grids must be rank-2 `(El, Az)` arrays; got shape {array.shape}"
        raise ValueError(msg)
    return np.expand_dims(np.transpose(array, (1, 0)), axis=0)


def normalize_depth_metres(
    depth_metres: NDArray[Any],
    *,
    max_distance: float,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Normalize metre depths into `[0, 1]` and return the canonical valid mask."""
    if max_distance <= 0.0:
        msg = f"max_distance must be positive; got {max_distance}"
        raise ValueError(msg)

    depth = np.asarray(depth_metres, dtype=np.float32)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0) & (depth <= float(max_distance))
    clamped = np.clip(np.where(finite, depth, float(max_distance)), 0.0, float(max_distance))
    normalized = (clamped / float(max_distance)).astype(np.float32, copy=False)
    return normalized, valid.astype(np.bool_, copy=False)


def remap_semantic_ids(
    semantic_ids: NDArray[Any],
    remap_table: dict[int, int],
    *,
    unknown_id: int = 0,
) -> NDArray[np.int32]:
    """Remap external semantic ids into Navi's canonical contiguous ids."""
    semantic_array = np.asarray(semantic_ids, dtype=np.int32)
    remapped = np.fromiter(
        (remap_table.get(int(value), int(unknown_id)) for value in semantic_array.flat),
        dtype=np.int32,
        count=int(semantic_array.size),
    )
    return remapped.reshape(semantic_array.shape)


def compute_delta_depth(
    current_depth: NDArray[Any],
    previous_depth: NDArray[Any] | None,
) -> NDArray[np.float32]:
    """Return canonical depth delta with a zero first frame."""
    current = np.asarray(current_depth, dtype=np.float32)
    if previous_depth is None:
        return np.zeros_like(current, dtype=np.float32)

    previous = np.asarray(previous_depth, dtype=np.float32)
    if previous.shape != current.shape:
        msg = f"previous_depth must match current_depth shape {current.shape}; got {previous.shape}"
        raise ValueError(msg)
    return (current - previous).astype(np.float32, copy=False)


def apply_rigid_transform(
    points_xyz: NDArray[Any],
    transform: RigidTransformSpec,
) -> NDArray[np.float32]:
    """Apply an explicit homogeneous rigid transform to a point cloud."""
    matrix = np.asarray(transform.matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        msg = f"transform matrix for {transform.name} must have shape (4, 4); got {matrix.shape}"
        raise ValueError(msg)

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        msg = f"points_xyz must end with xyz coordinates; got shape {points.shape}"
        raise ValueError(msg)

    flat_points = points.reshape(-1, 3)
    homogeneous = np.concatenate(
        [flat_points, np.ones((flat_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].reshape(points.shape).astype(np.float32, copy=False)


def materialize_distance_matrix(
    *,
    episode_id: int,
    env_id: int,
    depth: NDArray[Any],
    delta_depth: NDArray[Any],
    semantic: NDArray[Any],
    valid_mask: NDArray[Any],
    robot_pose: RobotPose,
    step_id: int,
    overhead: NDArray[Any] | None = None,
    timestamp: float | None = None,
) -> DistanceMatrix:
    """Build a canonical public observation from already-adapted batched arrays."""
    depth_array = _require_canonical_observation_array(depth, name="depth", dtype=np.float32)
    delta_array = _require_canonical_observation_array(delta_depth, name="delta_depth", dtype=np.float32)
    semantic_array = _require_canonical_observation_array(semantic, name="semantic", dtype=np.int32)
    valid_array = _require_canonical_observation_array(valid_mask, name="valid_mask", dtype=np.bool_)

    expected_shape = depth_array.shape
    for name, value in (
        ("delta_depth", delta_array),
        ("semantic", semantic_array),
        ("valid_mask", valid_array),
    ):
        if value.shape != expected_shape:
            msg = f"{name} must match depth shape {expected_shape}; got {value.shape}"
            raise ValueError(msg)

    if overhead is None:
        overhead_array = np.zeros((256, 256, 3), dtype=np.float32)
    else:
        overhead_array = np.asarray(overhead, dtype=np.float32)
        if overhead_array.ndim != 3 or overhead_array.shape[2] != 3:
            msg = f"overhead must be shaped (H, W, 3); got {overhead_array.shape}"
            raise ValueError(msg)

    matrix_shape = (int(depth_array.shape[1]), int(depth_array.shape[2]))
    return DistanceMatrix(
        episode_id=int(episode_id),
        env_ids=np.array([env_id], dtype=np.int32),
        matrix_shape=matrix_shape,
        depth=depth_array,
        delta_depth=delta_array,
        semantic=semantic_array,
        valid_mask=valid_array,
        overhead=overhead_array.astype(np.float32, copy=False),
        robot_pose=robot_pose,
        step_id=int(step_id),
        timestamp=float(time.time() if timestamp is None else timestamp),
    )


def _require_canonical_observation_array(
    values: NDArray[Any],
    *,
    name: str,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 3 or array.shape[0] != 1:
        msg = f"{name} must have canonical shape (1, Az, El); got {array.shape}"
        raise ValueError(msg)
    return array


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

    External data-source backends own the adapter and call
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

    * Rate conversion -- actor decides at ~10-50 Hz; the adapter
      re-sends the held command to hardware at the SDK's required
      rate (often 50-200 Hz).
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
