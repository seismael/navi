"""Adapter-backed utilities for canonical dataset fixture playback."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from navi_contracts import DistanceMatrix, RobotPose
from navi_environment.backends.adapter import DatasetAdapter, materialize_distance_matrix

__all__ = [
    "DatasetFixtureFrame",
    "adapt_fixture_frame",
    "adapt_fixture_sequence",
]


@dataclass(frozen=True, slots=True)
class DatasetFixtureFrame:
    """One raw dataset frame plus the metadata needed for public materialization."""

    raw_obs: dict[str, Any]
    episode_id: int
    env_id: int
    robot_pose: RobotPose
    step_id: int
    timestamp: float | None = None


def adapt_fixture_frame(
    adapter: DatasetAdapter,
    frame: DatasetFixtureFrame,
) -> DistanceMatrix:
    """Convert one raw fixture frame into the canonical public observation."""
    adapted = adapter.adapt(frame.raw_obs, step_id=frame.step_id)
    return materialize_distance_matrix(
        episode_id=frame.episode_id,
        env_id=frame.env_id,
        depth=adapted["depth"],
        delta_depth=adapted["delta_depth"],
        semantic=adapted["semantic"],
        valid_mask=adapted["valid_mask"],
        overhead=adapted.get("overhead"),
        robot_pose=frame.robot_pose,
        step_id=frame.step_id,
        timestamp=frame.timestamp,
    )


def adapt_fixture_sequence(
    adapter: DatasetAdapter,
    frames: Iterable[DatasetFixtureFrame],
    *,
    reset_on_episode_change: bool = True,
) -> tuple[DistanceMatrix, ...]:
    """Materialize a fixture stream through the canonical adapter boundary.

    This keeps dataset QA and future dataset-auditor work on the same
    `DistanceMatrix` contract used by runtime-backed publication without
    introducing a second production backend.
    """
    observations: list[DistanceMatrix] = []
    previous_episode_id: int | None = None

    for frame in frames:
        if (
            reset_on_episode_change
            and previous_episode_id is not None
            and frame.episode_id != previous_episode_id
        ):
            adapter.reset()
        observations.append(adapt_fixture_frame(adapter, frame))
        previous_episode_id = frame.episode_id

    return tuple(observations)
