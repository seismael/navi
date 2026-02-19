"""Shared spherical observation featurization for actor policies and training."""

from __future__ import annotations

import numpy as np

from navi_contracts import DistanceMatrix

__all__: list[str] = ["extract_spherical_features"]


def extract_spherical_features(obs: DistanceMatrix) -> np.ndarray:
    """Extract compact features from full 360° depth matrix.

    Uses all azimuth bins by aggregating global sectors and near-front/rear slices.
    Output shape is fixed at ``(13,)``.
    """
    depth = obs.depth[0]
    valid = obs.valid_mask[0]
    if depth.ndim != 2:
        msg = "DistanceMatrix depth tensor must be rank-2 after batch slice."
        raise ValueError(msg)

    safe_depth = np.where(valid, depth, 1.0)
    per_az_min = np.clip(np.min(safe_depth, axis=1), 0.0, 1.0)
    az_bins = max(1, per_az_min.shape[0])

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
        ]
    )
    return np.clip(feats, 0.0, 1.0).astype(np.float32)
