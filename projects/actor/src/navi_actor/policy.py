"""Shallow policy for DistanceMatrix v2 observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navi_actor.spherical_features import extract_spherical_features
from navi_contracts import Action, DistanceMatrix

__all__: list[str] = ["LearnedSphericalPolicy", "PolicyCheckpoint", "ShallowPolicy"]


@dataclass(frozen=True)
class PolicyCheckpoint:
    """Serializable checkpoint payload for learned spherical policy."""

    w_forward: np.ndarray
    b_forward: float
    w_yaw: np.ndarray
    b_yaw: float
    w_vertical: np.ndarray
    b_vertical: float
    w_lateral: np.ndarray
    b_lateral: float
    max_forward: float
    max_yaw: float
    max_vertical: float
    max_lateral: float


class LearnedSphericalPolicy:
    """Deterministic policy driven by full spherical observation features."""

    def __init__(
        self,
        checkpoint: PolicyCheckpoint,
        policy_id: str = "brain-v2-learned-spherical",
    ) -> None:
        self._checkpoint = checkpoint
        self._policy_id = policy_id

    @property
    def policy_id(self) -> str:
        """Return stable policy identifier used on the wire."""
        return self._policy_id

    def act(self, observation: DistanceMatrix, step_id: int) -> Action:
        """Infer deterministic 4-DOF velocity command from spherical features."""
        features = extract_spherical_features(observation)

        f_pre = float(np.dot(self._checkpoint.w_forward, features) + self._checkpoint.b_forward)
        y_pre = float(np.dot(self._checkpoint.w_yaw, features) + self._checkpoint.b_yaw)
        v_pre = float(np.dot(self._checkpoint.w_vertical, features) + self._checkpoint.b_vertical)
        l_pre = float(np.dot(self._checkpoint.w_lateral, features) + self._checkpoint.b_lateral)

        forward = float(self._checkpoint.max_forward / (1.0 + np.exp(-f_pre)))
        yaw = float(self._checkpoint.max_yaw * np.tanh(y_pre))
        vertical = float(self._checkpoint.max_vertical * np.tanh(v_pre))
        lateral = float(self._checkpoint.max_lateral * np.tanh(l_pre))

        linear_velocity = np.array([[forward, vertical, lateral]], dtype=np.float32)
        angular_velocity = np.array([[0.0, 0.0, yaw]], dtype=np.float32)

        return Action(
            env_ids=observation.env_ids.astype(np.int32),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            policy_id=self._policy_id,
            step_id=step_id,
            timestamp=float(observation.timestamp),
        )

    @staticmethod
    def save_checkpoint(path: str, checkpoint: PolicyCheckpoint) -> None:
        """Persist policy checkpoint to compressed NPZ file."""
        np.savez_compressed(
            path,
            w_forward=checkpoint.w_forward.astype(np.float32),
            b_forward=np.float32(checkpoint.b_forward),
            w_yaw=checkpoint.w_yaw.astype(np.float32),
            b_yaw=np.float32(checkpoint.b_yaw),
            w_vertical=checkpoint.w_vertical.astype(np.float32),
            b_vertical=np.float32(checkpoint.b_vertical),
            w_lateral=checkpoint.w_lateral.astype(np.float32),
            b_lateral=np.float32(checkpoint.b_lateral),
            max_forward=np.float32(checkpoint.max_forward),
            max_yaw=np.float32(checkpoint.max_yaw),
            max_vertical=np.float32(checkpoint.max_vertical),
            max_lateral=np.float32(checkpoint.max_lateral),
        )

    @staticmethod
    def load_checkpoint(path: str) -> PolicyCheckpoint:
        """Load policy checkpoint from compressed NPZ file."""
        data = np.load(path)
        feature_dim = len(data["w_forward"])
        return PolicyCheckpoint(
            w_forward=np.array(data["w_forward"], dtype=np.float32),
            b_forward=float(data["b_forward"]),
            w_yaw=np.array(data["w_yaw"], dtype=np.float32),
            b_yaw=float(data["b_yaw"]),
            w_vertical=np.array(
                data["w_vertical"] if "w_vertical" in data
                else np.zeros(feature_dim, dtype=np.float32),
                dtype=np.float32,
            ),
            b_vertical=float(data["b_vertical"]) if "b_vertical" in data else 0.0,
            w_lateral=np.array(
                data["w_lateral"] if "w_lateral" in data
                else np.zeros(feature_dim, dtype=np.float32),
                dtype=np.float32,
            ),
            b_lateral=float(data["b_lateral"]) if "b_lateral" in data else 0.0,
            max_forward=float(data["max_forward"]),
            max_yaw=float(data["max_yaw"]),
            max_vertical=float(data["max_vertical"]) if "max_vertical" in data else 0.8,
            max_lateral=float(data["max_lateral"]) if "max_lateral" in data else 0.8,
        )


class ShallowPolicy:
    """Small policy that maps distance matrices to batched actions."""

    def __init__(self, policy_id: str = "brain-v2-shallow", gain: float = 0.5) -> None:
        self._policy_id = policy_id
        self._gain = gain

    @property
    def policy_id(self) -> str:
        """Return stable policy identifier used on the wire."""
        return self._policy_id

    def act(self, observation: DistanceMatrix, step_id: int) -> Action:
        """Infer linear/angular velocity from a DistanceMatrix v2 message."""
        depth = observation.depth
        valid = observation.valid_mask
        batch = depth.shape[0]

        forward = np.full((batch,), 0.2, dtype=np.float32)
        yaw = np.zeros((batch,), dtype=np.float32)

        left_sector = depth[:, : depth.shape[1] // 2, :]
        right_sector = depth[:, depth.shape[1] // 2 :, :]
        left_valid = valid[:, : depth.shape[1] // 2, :]
        right_valid = valid[:, depth.shape[1] // 2 :, :]

        left_mean = np.where(left_valid.any(axis=(1, 2)), left_sector.mean(axis=(1, 2)), 1.0)
        right_mean = np.where(right_valid.any(axis=(1, 2)), right_sector.mean(axis=(1, 2)), 1.0)
        yaw = np.clip((right_mean - left_mean) * self._gain, -0.5, 0.5).astype(np.float32)

        occupancy = depth.mean(axis=(1, 2))
        forward = np.where(occupancy < 0.1, 0.0, forward).astype(np.float32)

        linear_velocity = np.stack(
            [forward, np.zeros_like(forward), np.zeros_like(forward)],
            axis=1,
        ).astype(np.float32)
        angular_velocity = np.stack(
            [np.zeros_like(yaw), np.zeros_like(yaw), yaw],
            axis=1,
        ).astype(np.float32)

        return Action(
            env_ids=observation.env_ids.astype(np.int32),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            policy_id=self._policy_id,
            step_id=step_id,
            timestamp=float(observation.timestamp),
        )
