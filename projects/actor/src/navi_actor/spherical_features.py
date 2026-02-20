"""Shared spherical observation featurization for actor policies and training."""

from __future__ import annotations

import numpy as np

from navi_contracts import DistanceMatrix

__all__: list[str] = ["extract_spherical_features"]


def extract_spherical_features(obs: DistanceMatrix) -> np.ndarray:
    """Extract compact features from full 360° depth matrix.

    Uses all azimuth bins by aggregating global sectors and near-front/rear slices.
    Includes elevation-aware features for 3D drone navigation:
    - Floor proximity (min depth in lower elevation bins)
    - Ceiling proximity (min depth in upper elevation bins)
    - Vertical clearance ratio (floor vs ceiling proximity)
    - Target detection signal (fraction of near-depth bins suggesting close objects)

    Output shape is fixed at ``(17,)``.
    """
    depth = obs.depth[0]
    valid = obs.valid_mask[0]
    if depth.ndim != 2:
        msg = "DistanceMatrix depth tensor must be rank-2 after batch slice."
        raise ValueError(msg)

    safe_depth = np.where(valid, depth, 1.0)
    per_az_min = np.clip(np.min(safe_depth, axis=1), 0.0, 1.0)
    az_bins = max(1, per_az_min.shape[0])
    el_bins = max(1, depth.shape[1])

    center = az_bins // 2
    rolled = np.roll(per_az_min, -center)
    span = max(2, az_bins // 16)

    front_vals = np.concatenate([rolled[:span], rolled[-span:]])
    rear_lo = max(0, az_bins // 2 - span)
    rear_hi = min(az_bins, az_bins // 2 + span)
    rear_vals = rolled[rear_lo:rear_hi]
    left_vals = rolled[span : max(span + 1, az_bins // 2)]
    right_vals = rolled[max(az_bins // 2, span) : max(az_bins // 2 + 1, az_bins - span)]

    sector_count = 8
    sectors = np.array_split(per_az_min, sector_count)
    sector_means = np.array(
        [float(np.mean(sec)) if sec.size > 0 else 1.0 for sec in sectors],
        dtype=np.float32,
    )

    # ── Elevation-aware features for 3D drone control ────────────────
    # Lower elevation bins = floor/below, upper = ceiling/above
    el_lower = max(1, el_bins // 4)
    el_upper_start = max(el_lower, el_bins - el_bins // 4)

    safe_floor = safe_depth[:, :el_lower]
    safe_ceil = safe_depth[:, el_upper_start:]
    valid_floor = valid[:, :el_lower]
    valid_ceil = valid[:, el_upper_start:]

    floor_min = float(np.min(safe_floor[valid_floor])) if np.any(valid_floor) else 1.0
    ceil_min = float(np.min(safe_ceil[valid_ceil])) if np.any(valid_ceil) else 1.0
    vert_clearance = float(np.clip(ceil_min / max(floor_min, 1e-4), 0.0, 1.0))

    # Near-object detection: fraction of bins with depth < 0.15
    near_threshold = 0.15
    near_fraction = float(np.sum(safe_depth[valid] < near_threshold)) / max(1.0, float(np.sum(valid)))

    feats = np.concatenate(
        [
            np.array(
                [
                    float(np.min(front_vals)) if front_vals.size > 0 else 1.0,
                    float(np.mean(front_vals)) if front_vals.size > 0 else 1.0,
                    float(np.min(rear_vals)) if rear_vals.size > 0 else 1.0,
                    float(np.mean(left_vals)) if left_vals.size > 0 else 1.0,
                    float(np.mean(right_vals)) if right_vals.size > 0 else 1.0,
                ],
                dtype=np.float32,
            ),
            sector_means,
            np.array(
                [
                    float(np.clip(floor_min, 0.0, 1.0)),
                    float(np.clip(ceil_min, 0.0, 1.0)),
                    vert_clearance,
                    float(np.clip(near_fraction, 0.0, 1.0)),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return np.clip(feats, 0.0, 1.0).astype(np.float32)
