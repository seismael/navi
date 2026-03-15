from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OracleObservation:
    depth: np.ndarray
    valid: np.ndarray
    semantic: np.ndarray


def write_square_house_obj(path: str | Path) -> None:
    """Write a deterministic square-house fixture with door/window hints.

    The current Python verification compiler does not yet preserve mesh openings
    with full geometric fidelity, so this source fixture is used primarily for
    deterministic compile metadata and shared scene identity across layers.
    """
    target = Path(path)
    vertices = [
        (-2.0, 0.0, -2.0),
        (2.0, 0.0, -2.0),
        (2.0, 2.5, -2.0),
        (-2.0, 2.5, -2.0),
        (-2.0, 0.0, 2.0),
        (2.0, 0.0, 2.0),
        (2.0, 2.5, 2.0),
        (-2.0, 2.5, 2.0),
        (-0.6, 0.0, -2.0),
        (0.6, 0.0, -2.0),
        (0.6, 1.8, -2.0),
        (-0.6, 1.8, -2.0),
        (2.0, 1.0, -0.5),
        (2.0, 2.0, -0.5),
        (2.0, 2.0, 0.5),
        (2.0, 1.0, 0.5),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7),
        (4, 1, 5), (4, 5, 8),
        (9, 10, 11), (9, 11, 12),
        (13, 14, 15), (13, 15, 16),
    ]
    lines = [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a} {b} {c}" for a, b, c in faces)]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def canonical_house_bbox() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([-2.0, 0.0, -2.0], dtype=np.float32),
        np.array([2.0, 2.5, 2.0], dtype=np.float32),
    )


def house_observation() -> OracleObservation:
    """Return one deterministic spherical observation for door/window fidelity tests."""
    az_bins = 12
    el_bins = 3
    depth = np.full((az_bins, el_bins), 0.22, dtype=np.float32)
    valid = np.ones((az_bins, el_bins), dtype=np.bool_)
    semantic = np.full((az_bins, el_bins), 1, dtype=np.int32)

    for az in (0, az_bins - 1):
        valid[az, 1:] = False
        depth[az, 1:] = 1.0
        semantic[az, 1:] = 0

    for az in (3, 4):
        valid[az, 0:2] = False
        depth[az, 0:2] = 1.0
        semantic[az, 0:2] = 9

    depth[:, 0] = np.where(valid[:, 0], 0.35, depth[:, 0])
    return OracleObservation(depth=depth, valid=valid, semantic=semantic)


def house_observation_after_forward_motion() -> OracleObservation:
    first = house_observation()
    depth = first.depth.copy()
    valid = first.valid.copy()
    semantic = first.semantic.copy()
    depth[1:3, 1:] = 0.16
    depth[-3:-1, 1:] = 0.16
    depth[5:8, :] = np.minimum(depth[5:8, :], 0.18)
    return OracleObservation(depth=depth, valid=valid, semantic=semantic)


def house_observation_delta() -> np.ndarray:
    return house_observation_after_forward_motion().depth - house_observation().depth


def house_metric_distances(max_distance: float) -> np.ndarray:
    return house_observation().depth * float(max_distance)
