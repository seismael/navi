"""Type aliases for the Navi wire format."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = [
    "DeltaDepthMatrix",
    "DepthMatrix",
    "EnvIdVector",
    "MatrixShape",
    "SemanticMatrix",
    "TelemetryPayload",
    "ValidMask",
    "VelocityMatrix",
]

MatrixShape: TypeAlias = tuple[int, int]
"""(azimuth_bins, elevation_bins) — distance matrix resolution."""

DepthMatrix: TypeAlias = NDArray[np.float32]
"""(batch, azimuth_bins, elevation_bins) — normalized distance values in [0, 1]."""

DeltaDepthMatrix: TypeAlias = NDArray[np.float32]
"""(batch, azimuth_bins, elevation_bins) — temporal depth deltas."""

SemanticMatrix: TypeAlias = NDArray[np.int32]
"""(batch, azimuth_bins, elevation_bins) — semantic identifiers per cell."""

ValidMask: TypeAlias = NDArray[np.bool_]
"""(batch, azimuth_bins, elevation_bins) — True where a ray hit is valid."""

EnvIdVector: TypeAlias = NDArray[np.int32]
"""(batch,) — active environment IDs."""

VelocityMatrix: TypeAlias = NDArray[np.float32]
"""(batch, 3) — linear or angular velocity commands."""

TelemetryPayload: TypeAlias = NDArray[np.float32]
"""(N, M) — generic numeric telemetry payload for logging."""
