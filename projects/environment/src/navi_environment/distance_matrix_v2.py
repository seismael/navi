"""DistanceMatrix v2 builder for Ghost-Matrix simulation output."""

from __future__ import annotations

import cv2
import numpy as np

from navi_contracts import DistanceMatrix, RobotPose
from navi_environment.raycast import RaycastEngine

__all__: list[str] = ["DistanceMatrixBuilder"]

# Semantic-to-BGR colour table for the overhead minimap.
_SEMANTIC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (40, 40, 40),      # AIR — dark grey background
    1: (80, 80, 80),      # WALL — medium grey
    2: (100, 130, 90),    # FLOOR — muted green
    3: (120, 100, 80),    # CEILING — brown-grey
    4: (60, 160, 200),    # PILLAR — cyan
    5: (50, 120, 200),    # RAMP — orange-ish
    6: (80, 80, 200),     # OBSTACLE — red-ish
    7: (140, 140, 140),   # ROAD — light grey
    8: (100, 70, 50),     # BUILDING — dark blue-brown
    9: (200, 200, 100),   # WINDOW — light cyan
}
_MINIMAP_SIZE = 256  # overhead minimap is 256x256 pixels


class DistanceMatrixBuilder:
    """Builds canonical DistanceMatrix v2 payloads from local voxel snapshots.

    Tracks the previous depth plane so ``delta_depth`` (temporal velocity
    awareness channel) is computed as the per-bin change since the last step.
    """

    def __init__(self, azimuth_bins: int, elevation_bins: int, max_distance: float) -> None:
        self._azimuth_bins = azimuth_bins
        self._elevation_bins = elevation_bins
        self._raycast = RaycastEngine(
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            max_distance=max_distance,
        )
        self._prev_depth: np.ndarray | None = None
        self._max_distance = max_distance

    def _build_overhead(
        self,
        voxels: np.ndarray,
        pose: RobotPose,
    ) -> np.ndarray:
        """Render a top-down overhead minimap centred on the robot.

        Projects all voxels onto the XZ plane, colours by semantic ID,
        and draws the robot position and heading arrow.
        """
        size = _MINIMAP_SIZE
        img = np.full((size, size, 3), 40, dtype=np.uint8)

        if voxels.shape[0] == 0:
            return img

        # View radius in world units — tighter than max_distance so
        # nearby geometry fills the minimap.
        view_radius = min(self._max_distance, 18.0)
        scale = size / (2.0 * view_radius)

        # Project voxels → pixel coordinates centred on robot
        dx = voxels[:, 0] - pose.x
        dz = voxels[:, 2] - pose.z

        px = (dx * scale + size / 2.0).astype(np.intp)
        pz = (dz * scale + size / 2.0).astype(np.intp)

        # Filter to visible voxels (within image bounds)
        in_bounds = (px >= 0) & (px < size) & (pz >= 0) & (pz < size)
        px = px[in_bounds]
        pz = pz[in_bounds]
        sem = np.rint(voxels[in_bounds, 4]).astype(np.int32)

        # Paint voxels by semantic colour (2x2 filled blocks for visibility)
        for sid, colour in _SEMANTIC_COLORS.items():
            mask = sem == sid
            if np.any(mask):
                mpx = px[mask]
                mpz = pz[mask]
                # Draw 2x2 block for each point
                for dx in range(2):
                    for dz in range(2):
                        cpx = np.clip(mpx + dx, 0, size - 1)
                        cpz = np.clip(mpz + dz, 0, size - 1)
                        img[cpz, cpx] = colour

        # Draw robot position — bright magenta dot (clearly distinct
        # from floor green and wall grey).
        cx, cz = size // 2, size // 2
        img = np.ascontiguousarray(img)

        # Heading arrow
        arrow_len = max(10, min(int(12 * scale / max(size / (2.0 * view_radius), 1e-3) + 10), 35))
        end_x = int(cx + arrow_len * np.cos(pose.yaw))
        end_z = int(cz - arrow_len * np.sin(pose.yaw))

        cv2.circle(img, (cx, cz), 4, (255, 0, 255), thickness=-1)
        cv2.arrowedLine(
            img, (cx, cz), (end_x, end_z), (255, 0, 255), 2, tipLength=0.35,
        )

        return img

    def build(
        self,
        voxels: np.ndarray,
        pose: RobotPose,
        step_id: int,
        timestamp: float,
        episode_id: int = 0,
        env_id: int = 0,
    ) -> DistanceMatrix:
        """Construct a DistanceMatrix message from voxel world data."""
        relative = np.zeros((0, 3), dtype=np.float32)
        semantic = np.zeros((0,), dtype=np.int32)

        if voxels.shape[0] > 0:
            relative = voxels[:, :3] - np.array([pose.x, pose.y, pose.z], dtype=np.float32)

            # Rotate into heading-relative frame so the depth panorama
            # is centred on the robot's forward direction.
            cos_y = float(np.cos(-pose.yaw))
            sin_y = float(np.sin(-pose.yaw))
            rx = relative[:, 0] * cos_y - relative[:, 2] * sin_y
            rz = relative[:, 0] * sin_y + relative[:, 2] * cos_y
            relative = np.column_stack([rx, relative[:, 1], rz])

            semantic = np.rint(voxels[:, 4]).astype(np.int32)

        depth_2d, semantic_2d, valid_2d = self._raycast.project(
            relative_xyz=relative,
            semantic_ids=semantic,
        )

        # --- delta_depth: temporal change channel (velocity awareness) ---
        if self._prev_depth is not None and self._prev_depth.shape == depth_2d.shape:
            delta_2d = depth_2d - self._prev_depth
        else:
            delta_2d = np.zeros_like(depth_2d, dtype=np.float32)
        self._prev_depth = depth_2d.copy()

        depth = depth_2d[np.newaxis, ...]
        delta_depth = delta_2d[np.newaxis, ...]
        semantic_3d = semantic_2d[np.newaxis, ...]
        valid_mask = valid_2d[np.newaxis, ...]

        # --- overhead minimap ---
        overhead = self._build_overhead(voxels, pose)

        return DistanceMatrix(
            episode_id=episode_id,
            env_ids=np.array([env_id], dtype=np.int32),
            matrix_shape=(self._azimuth_bins, self._elevation_bins),
            depth=depth,
            delta_depth=delta_depth,
            semantic=semantic_3d,
            valid_mask=valid_mask,
            overhead=overhead,
            robot_pose=pose,
            step_id=step_id,
            timestamp=timestamp,
        )
