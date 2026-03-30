"""Demonstration recorder for behavioral cloning pre-training.

Captures (observation, action) pairs during manual exploration and saves
them as numpy archives for subsequent supervised training by the actor's
``bc-pretrain`` pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from navi_contracts import DistanceMatrix

__all__: list[str] = ["DemonstrationRecorder"]

_LOG = logging.getLogger(__name__)


class DemonstrationRecorder:
    """Accumulates (observation, action) pairs during manual navigation.

    Observations are stored as ``(3, Az, El)`` float32 arrays matching the
    actor policy input contract (depth, semantic, valid_mask).  Actions are
    stored as ``(4,)`` float32 arrays in the [-1, 1] normalized policy
    action space.

    Parameters
    ----------
    drone_max_speed:
        Max forward speed (m/s) for normalizing the forward velocity command.
    drone_climb_rate:
        Max climb rate (m/s) for normalizing the vertical velocity command.
    drone_strafe_speed:
        Max strafe speed (m/s) for normalizing the lateral velocity command.
    drone_yaw_rate:
        Max yaw rate (rad/s) for normalizing the yaw command.
    scene_name:
        Optional scene identifier stored in the demo metadata.
    """

    def __init__(
        self,
        *,
        drone_max_speed: float = 5.0,
        drone_climb_rate: float = 2.0,
        drone_strafe_speed: float = 3.0,
        drone_yaw_rate: float = 3.0,
        scene_name: str = "",
    ) -> None:
        self._drone_max_speed = drone_max_speed
        self._drone_climb_rate = drone_climb_rate
        self._drone_strafe_speed = drone_strafe_speed
        self._drone_yaw_rate = drone_yaw_rate
        self._scene_name = scene_name

        self._observations: list[NDArray[np.float32]] = []
        self._actions: list[NDArray[np.float32]] = []
        self._recording = False

    # ── public API ───────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def step_count(self) -> int:
        return len(self._observations)

    def start(self) -> None:
        """Begin a new recording session, clearing any previous data."""
        self._observations.clear()
        self._actions.clear()
        self._recording = True
        _LOG.info("Demonstration recording started")

    def stop(self) -> None:
        """Stop recording (data is retained for saving)."""
        self._recording = False
        _LOG.info("Demonstration recording stopped — %d steps captured", self.step_count)

    def capture(
        self,
        dm: DistanceMatrix,
        linear_velocity: float,
        yaw_rate: float,
        vertical_velocity: float,
    ) -> None:
        """Record one (observation, action) pair.

        Parameters
        ----------
        dm:
            The current distance-matrix observation from the environment.
        linear_velocity:
            Raw forward velocity (m/s) as sent to the environment.
        yaw_rate:
            Raw yaw rate (rad/s) as sent to the environment.
        vertical_velocity:
            Raw vertical velocity (m/s) as sent to the environment.
        """
        if not self._recording:
            return

        # ── Build (3, Az, El) observation ────────────────────────
        depth = np.asarray(dm.depth[0], dtype=np.float32)       # (Az, El)
        semantic = np.asarray(dm.semantic[0], dtype=np.float32)  # (Az, El)
        valid = np.asarray(dm.valid_mask[0], dtype=np.float32)   # (Az, El)
        obs = np.stack([depth, semantic, valid], axis=0)          # (3, Az, El)

        # ── Normalize action to [-1, 1] policy space ─────────────
        fwd_norm = np.clip(linear_velocity / self._drone_max_speed, -1.0, 1.0)
        vert_norm = np.clip(vertical_velocity / self._drone_climb_rate, -1.0, 1.0)
        lat_norm = 0.0  # No lateral strafe key in dashboard currently
        yaw_norm = np.clip(yaw_rate / self._drone_yaw_rate, -1.0, 1.0)
        action = np.array([fwd_norm, vert_norm, lat_norm, yaw_norm], dtype=np.float32)

        self._observations.append(obs)
        self._actions.append(action)

    def save(self, directory: str | Path | None = None) -> Path:
        """Save accumulated demonstration data to a numpy archive.

        Parameters
        ----------
        directory:
            Target directory.  Defaults to ``artifacts/demonstrations/``.

        Returns
        -------
        Path to the saved ``.npz`` file.
        """
        if not self._observations:
            msg = "No demonstration data to save"
            raise ValueError(msg)

        if directory is None:
            directory = Path("artifacts/demonstrations")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}.npz"
        path = directory / filename

        observations = np.stack(self._observations, axis=0)  # (N, 3, Az, El)
        actions = np.stack(self._actions, axis=0)             # (N, 4)
        az, el = observations.shape[2], observations.shape[3]

        np.savez_compressed(
            path,
            observations=observations,
            actions=actions,
            format_version=np.array(1, dtype=np.int32),
            scene=np.array(self._scene_name),
            azimuth_bins=np.array(az, dtype=np.int32),
            elevation_bins=np.array(el, dtype=np.int32),
            total_steps=np.array(len(self._observations), dtype=np.int32),
            recording_date=np.array(timestamp),
            drone_max_speed=np.array(self._drone_max_speed, dtype=np.float32),
            drone_climb_rate=np.array(self._drone_climb_rate, dtype=np.float32),
            drone_strafe_speed=np.array(self._drone_strafe_speed, dtype=np.float32),
            drone_yaw_rate=np.array(self._drone_yaw_rate, dtype=np.float32),
        )

        _LOG.info(
            "Demonstration saved: %s (%d steps, obs=%s, actions=%s)",
            path,
            len(self._observations),
            observations.shape,
            actions.shape,
        )
        return path
