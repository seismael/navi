"""CUDA-backed SDF/DAG simulator backend.

This backend preserves the canonical ``DistanceMatrix`` observation contract while
moving scene queries onto the internal ``torch-sdf`` CUDA runtime over compiled
``.gmdag`` assets produced by ``projects/voxel-dag``.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult
from navi_environment.backends.adapter import materialize_distance_matrix
from navi_environment.backends.base import SimulatorBackend
from navi_environment.integration import GmDagAsset, load_gmdag_asset

if TYPE_CHECKING:
    from navi_environment.config import EnvironmentConfig

__all__: list[str] = [
    "SdfDagBackend",
    "SdfDagPerfSnapshot",
    "SdfDagTensorStepBatch",
    "build_spherical_ray_directions",
]

_LOG = logging.getLogger(__name__)

_COLLISION_CLEARANCE: float = 0.15
_EXPLORATION_REWARD: float = 0.3
_PROGRESS_REWARD_SCALE: float = 0.8
_COLLISION_PENALTY: float = -2.0
_MAX_STEPS_PER_EPISODE: int = 2_000
_SCENE_EPISODES_PER_SCENE: int = 16
_OBSTACLE_CLEARANCE_REWARD_SCALE: float = 0.6
_OBSTACLE_CLEARANCE_WINDOW: float = 3.0
_STARVATION_RATIO_THRESHOLD: float = 0.8
_STARVATION_PENALTY_SCALE: float = 1.5
_PROXIMITY_DISTANCE_THRESHOLD: float = 2.0
_PROXIMITY_PENALTY_SCALE: float = 0.8
_STRUCTURE_BAND_MIN_DISTANCE: float = 1.5
_STRUCTURE_BAND_MAX_DISTANCE: float = 30.0
_STRUCTURE_BAND_REWARD_SCALE: float = 0.35
_FORWARD_STRUCTURE_REWARD_SCALE: float = 0.2
_INSPECTION_REWARD_SCALE: float = 0.25
_INSPECTION_ACTIVATION_THRESHOLD: float = 0.05
_HEADING_NOVELTY_SCALE: float = 0.05
_HEADING_SECTORS: int = 16
_FRONTIER_BONUS_SCALE: float = 0.1
_STARVATION_DEAD_ZONE: float = 0.2
_VOID_STARVATION_THRESHOLD: float = 0.85
_SPAWN_MAX_STARVATION: float = 0.70
_SPAWN_CANDIDATES_PER_AXIS: int = 5
_SPAWN_HEIGHT_SAMPLES: int = 7
_SPAWN_PROBE_AZIMUTH_BINS: int = 48
_SPAWN_PROBE_ELEVATION_BINS: int = 12
_PERF_EMA_ALPHA: float = 0.1
_RAY_DIRECTION_NORM_EPS: float = 1e-4
_SCRATCH_SLOT_COUNT: int = 2


def _obstacle_clearance_reward(
    previous_clearance: float | None,
    current_clearance: float,
    *,
    proximity_window: float,
    reward_scale: float,
) -> float:
    if previous_clearance is None or proximity_window <= 0.0 or reward_scale == 0.0:
        return 0.0
    if previous_clearance > proximity_window and current_clearance > proximity_window:
        return 0.0
    normalized_delta = (current_clearance - previous_clearance) / proximity_window
    normalized_delta = max(-1.0, min(1.0, normalized_delta))
    return reward_scale * float(normalized_delta)


def _starvation_penalty(
    starvation_ratio: float,
    *,
    ratio_threshold: float,
    penalty_scale: float,
) -> float:
    if penalty_scale <= 0.0:
        return 0.0
    starvation_ratio = max(0.0, min(1.0, starvation_ratio))
    # Dead zone: starvation below _STARVATION_DEAD_ZONE is free (normal viewing).
    effective = max(0.0, starvation_ratio - _STARVATION_DEAD_ZONE)
    baseline = 0.35 * effective
    # Overflow accelerates penalty above the raw-ratio threshold.
    overflow = max(0.0, starvation_ratio - max(0.0, ratio_threshold))
    return -penalty_scale * float(min(baseline + overflow, 1.0))


def _proximity_penalty(
    proximity_ratio: float,
    *,
    penalty_scale: float,
) -> float:
    if penalty_scale <= 0.0 or proximity_ratio <= 0.0:
        return 0.0
    return -penalty_scale * float(min(proximity_ratio, 1.0))


def _structure_band_reward(
    structure_band_ratio: float,
    *,
    reward_scale: float,
) -> float:
    if reward_scale <= 0.0 or structure_band_ratio <= 0.0:
        return 0.0
    return reward_scale * float(min(max(structure_band_ratio, 0.0), 1.0))


def _forward_structure_reward(
    forward_structure_ratio: float,
    *,
    reward_scale: float,
) -> float:
    if reward_scale <= 0.0 or forward_structure_ratio <= 0.0:
        return 0.0
    return reward_scale * float(min(max(forward_structure_ratio, 0.0), 1.0))


def _inspection_reward(
    previous_structure_band_ratio: float | None,
    current_structure_band_ratio: float,
    *,
    previous_forward_structure_ratio: float | None,
    current_forward_structure_ratio: float,
    reward_scale: float,
    activation_threshold: float,
) -> float:
    if reward_scale <= 0.0:
        return 0.0
    if previous_structure_band_ratio is None or previous_forward_structure_ratio is None:
        return 0.0
    activation = max(
        previous_structure_band_ratio,
        current_structure_band_ratio,
        previous_forward_structure_ratio,
        current_forward_structure_ratio,
    )
    if activation < activation_threshold:
        return 0.0
    delta = (current_structure_band_ratio - previous_structure_band_ratio) + 0.5 * (
        current_forward_structure_ratio - previous_forward_structure_ratio
    )
    return reward_scale * float(max(-1.0, min(1.0, delta)))


def _observation_profile(
    metric_depth: np.ndarray,
    valid_2d: np.ndarray,
    *,
    max_distance: float = 100.0,
    proximity_distance_threshold: float,
    structure_band_min_distance: float,
    structure_band_max_distance: float,
) -> tuple[float, float, float, float]:
    """Compute observation ratios from metric distances (in meters)."""
    # Starvation: fraction of rays at max distance (informational void) rather
    # than fraction of invalid rays — matches the tensor-path semantics.
    saturated = metric_depth >= max_distance - 1e-3
    starvation_ratio = float(np.mean(saturated.astype(np.float32)))
    structure_band_ratio = 0.0
    forward_structure_ratio = 0.0
    if structure_band_max_distance > structure_band_min_distance:
        structure_mask = np.logical_and(
            valid_2d,
            np.logical_and(
                metric_depth >= structure_band_min_distance,
                metric_depth <= structure_band_max_distance,
            ),
        )
        structure_band_ratio = float(np.mean(structure_mask, dtype=np.float32))
        forward_half_bins = max(1, metric_depth.shape[0] // 6)
        forward_sector = np.concatenate(
            (structure_mask[:forward_half_bins, :], structure_mask[-forward_half_bins:, :]),
            axis=0,
        )
        forward_structure_ratio = float(np.mean(forward_sector, dtype=np.float32))
    if proximity_distance_threshold <= 0.0:
        return starvation_ratio, 0.0, structure_band_ratio, forward_structure_ratio
    near_mask = np.logical_and(valid_2d, metric_depth <= proximity_distance_threshold)
    proximity_ratio = float(np.mean(near_mask, dtype=np.float32))
    return (
        starvation_ratio,
        max(0.0, min(1.0, proximity_ratio)),
        max(0.0, min(1.0, structure_band_ratio)),
        max(0.0, min(1.0, forward_structure_ratio)),
    )


def build_spherical_ray_directions(azimuth_bins: int, elevation_bins: int) -> np.ndarray:
    az = np.linspace(0.0, 2.0 * math.pi, azimuth_bins, endpoint=False, dtype=np.float32)
    el = np.linspace(
        math.pi / 2.0, -math.pi / 2.0, elevation_bins, endpoint=True, dtype=np.float32
    )
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    az_flat = az_grid.reshape(-1)
    el_flat = el_grid.reshape(-1)
    cos_el = np.cos(el_flat)
    return np.stack(
        [
            cos_el * np.sin(az_flat),
            np.sin(el_flat),
            -cos_el * np.cos(az_flat),
        ],
        axis=-1,
    ).astype(np.float32)


def _select_spawn_yaw_from_observation(
    metric_depth: np.ndarray,
    valid_2d: np.ndarray,
    *,
    structure_band_min_distance: float,
    structure_band_max_distance: float,
) -> float:
    """Select best spawn yaw from metric distances (in meters)."""
    azimuth_bins = int(metric_depth.shape[0])
    if azimuth_bins <= 0:
        return 0.0
    structure_mask = np.logical_and(
        valid_2d,
        np.logical_and(
            metric_depth >= structure_band_min_distance,
            metric_depth <= structure_band_max_distance,
        ),
    )
    forward_half_bins = max(1, azimuth_bins // 6)
    best_shift = 0
    best_score = float("-inf")
    for shift in range(azimuth_bins):
        rolled_structure = np.roll(structure_mask, -shift, axis=0)
        rolled_valid = np.roll(valid_2d, -shift, axis=0)
        forward_structure = np.concatenate(
            (rolled_structure[:forward_half_bins, :], rolled_structure[-forward_half_bins:, :]),
            axis=0,
        )
        forward_valid = np.concatenate(
            (rolled_valid[:forward_half_bins, :], rolled_valid[-forward_half_bins:, :]),
            axis=0,
        )
        score = 2.0 * float(np.mean(forward_structure, dtype=np.float32)) + float(
            np.mean(forward_valid, dtype=np.float32)
        )
        if score > best_score:
            best_score = score
            best_shift = shift
    # Yaw rotation maps local az=0 to world az=-yaw; to point forward at
    # the structure found at azimuth (2π * shift / bins), use yaw = -shift_angle.
    return float(-(2.0 * math.pi * best_shift) / azimuth_bins) % (2.0 * math.pi)


def _spawn_candidate_score(
    metric_depth: np.ndarray,
    valid_2d: np.ndarray,
    *,
    max_distance: float,
    proximity_distance_threshold: float,
    structure_band_min_distance: float,
    structure_band_max_distance: float,
) -> float:
    starvation_ratio, proximity_ratio, structure_band_ratio, forward_structure_ratio = (
        _observation_profile(
            metric_depth,
            valid_2d,
            max_distance=max_distance,
            proximity_distance_threshold=proximity_distance_threshold,
            structure_band_min_distance=structure_band_min_distance,
            structure_band_max_distance=structure_band_max_distance,
        )
    )
    return float(
        (3.0 * structure_band_ratio)
        + (2.0 * forward_structure_ratio)
        - starvation_ratio
        - (1.5 * proximity_ratio)
    )


def _validate_unit_direction_tensor(
    torch_module: Any,
    ray_dirs: Any,
    *,
    name: str,
    eps: float = _RAY_DIRECTION_NORM_EPS,
) -> None:
    if ray_dirs.ndim < 2 or ray_dirs.shape[-1] != 3:
        msg = f"{name} must have trailing dimension 3; got shape {tuple(int(dim) for dim in ray_dirs.shape)}"
        raise RuntimeError(msg)
    norms = torch_module.linalg.vector_norm(ray_dirs, dim=-1)
    if bool((~torch_module.isfinite(norms)).any().item()):
        msg = f"{name} must contain only finite direction vectors"
        raise RuntimeError(msg)
    max_error = float(torch_module.abs(norms - 1.0).amax().detach().cpu().item())
    if max_error > eps:
        msg = f"{name} must be normalized within tolerance {eps}; max error was {max_error:.6g}"
        raise RuntimeError(msg)


def _maybe_compile_callable(
    torch_module: Any,
    fn: Any,
    *,
    enabled: bool,
    name: str,
) -> Any:
    if not enabled:
        return fn
    # 1. Try torch.compile only on supported CUDA GPUs.
    compile_supported = False
    compile_reason = "torch.compile unavailable"
    try:
        if bool(torch_module.cuda.is_available()):
            capability = torch_module.cuda.get_device_capability()
            if tuple(int(part) for part in capability) >= (7, 0):
                compile_supported = True
            else:
                compile_reason = f"CUDA capability {capability[0]}.{capability[1]} is below the Triton minimum 7.0"
        else:
            compile_reason = "CUDA is unavailable"
    except Exception as exc:
        compile_reason = f"CUDA capability probe failed ({exc})"

    compile_fn = getattr(torch_module, "compile", None)
    if compile_supported and compile_fn is not None:
        try:
            compiled = compile_fn(fn, fullgraph=True, dynamic=False, mode="reduce-overhead")
            _LOG.info("torch.compile enabled for %s", name)
            return compiled
        except Exception as exc:
            _LOG.info("torch.compile unavailable for %s (%s), trying torch.jit.script", name, exc)
    elif compile_fn is not None:
        _LOG.info(
            "Skipping torch.compile for %s (%s); trying torch.jit.script", name, compile_reason
        )
    # 2. Try torch.jit.script (works on all CUDA SMs)
    jit_mod = getattr(torch_module, "jit", None)
    if jit_mod is not None:
        script_fn = getattr(jit_mod, "script", None)
        if script_fn is not None:
            try:
                scripted = script_fn(fn)
                _LOG.info("torch.jit.script enabled for %s", name)
                return scripted
            except Exception as exc:
                _LOG.warning("torch.jit.script failed for %s: %s", name, exc)
    # 3. Eager fallback
    _LOG.warning("No compilation backend available for %s; using eager path", name)
    return fn


def _compute_observation_ratios_tensor(
    clamped_metric: torch.Tensor,
    valid_batch: torch.Tensor,
    *,
    max_distance: float,
    structure_band_min_distance: float,
    structure_band_max_distance: float,
    proximity_distance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Starvation: fraction of rays at max distance (informational void) rather
    # than fraction of invalid rays — every finite ray is now a valid sensor
    # reading so starvation must be distance-threshold based.
    saturated = clamped_metric >= max_distance - 1e-3
    starvation_ratios = saturated.to(dtype=torch.float32).mean(dim=(1, 2))
    structure_band_ratios = torch.zeros_like(starvation_ratios)
    forward_structure_ratios = torch.zeros_like(starvation_ratios)
    if structure_band_max_distance > structure_band_min_distance:
        structure_mask = (
            valid_batch
            & (clamped_metric >= structure_band_min_distance)
            & (clamped_metric <= structure_band_max_distance)
        )
        structure_band_ratios = structure_mask.to(dtype=torch.float32).mean(dim=(1, 2))
        forward_half_bins = max(1, int(clamped_metric.shape[1]) // 6)
        forward_sector = torch.cat(
            (structure_mask[:, :forward_half_bins, :], structure_mask[:, -forward_half_bins:, :]),
            dim=1,
        )
        forward_structure_ratios = forward_sector.to(dtype=torch.float32).mean(dim=(1, 2))
    if proximity_distance_threshold <= 0.0:
        proximity_ratios = torch.zeros_like(starvation_ratios)
    else:
        proximity_mask = valid_batch & (clamped_metric <= proximity_distance_threshold)
        proximity_ratios = proximity_mask.to(dtype=torch.float32).mean(dim=(1, 2))
    return (
        starvation_ratios.clamp(min=0.0, max=1.0),
        proximity_ratios.clamp(min=0.0, max=1.0),
        structure_band_ratios.clamp(min=0.0, max=1.0),
        forward_structure_ratios.clamp(min=0.0, max=1.0),
    )


def _postprocess_cast_outputs_tensor(
    out_distances: torch.Tensor,
    out_semantics: torch.Tensor,
    *,
    az_bins: int,
    el_bins: int,
    max_distance: float,
    structure_band_min_distance: float,
    structure_band_max_distance: float,
    proximity_distance_threshold: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    actor_count = int(out_distances.shape[0])
    metric_distances = out_distances.reshape(actor_count, az_bins, el_bins)
    semantic_batch = out_semantics.reshape(actor_count, az_bins, el_bins).to(dtype=torch.int32)
    # Valid = every finite ray reading.  Rays that escape the bounding box or
    # exhaust max_steps still carry useful information ("nothing in that
    # direction") and must remain valid sensor readings — the old
    # distance<=max_distance gate created excessive fog-of-war holes on floors,
    # ceilings, and open areas.  Far-away rays are clamped to max_distance and
    # render as faint gray instead of absent hatching.
    if max_distance > 0.0:
        valid_batch = torch.isfinite(metric_distances)
        clamped_metric = metric_distances.clamp(min=0.0, max=max_distance)
        # Logarithmic normalization: preserves near-field detail while
        # compressing far-field distances into a usable gradient range.
        # log1p is numerically stable and maps [0, max_distance] → [0, 1].
        _log_denom = math.log1p(max_distance)
        depth_batch = (torch.log1p(clamped_metric) / _log_denom).clamp(min=0.0, max=1.0)
    else:
        valid_batch = torch.isfinite(metric_distances)
        clamped_metric = torch.where(
            valid_batch, metric_distances, torch.zeros_like(metric_distances)
        )
        depth_batch = torch.zeros_like(metric_distances)
    min_distances = clamped_metric.amin(dim=(1, 2))
    starvation_ratios, proximity_ratios, structure_band_ratios, forward_structure_ratios = (
        _compute_observation_ratios_tensor(
            clamped_metric,
            valid_batch,
            max_distance=max_distance,
            structure_band_min_distance=structure_band_min_distance,
            structure_band_max_distance=structure_band_max_distance,
            proximity_distance_threshold=proximity_distance_threshold,
        )
    )
    return (
        depth_batch,
        semantic_batch,
        valid_batch,
        min_distances,
        starvation_ratios,
        proximity_ratios,
        structure_band_ratios,
        forward_structure_ratios,
        clamped_metric,
    )


def _reward_components_tensor(
    exploration_rewards: torch.Tensor,
    previous_positions: torch.Tensor,
    current_positions: torch.Tensor,
    collisions: torch.Tensor,
    previous_clearances: torch.Tensor,
    current_clearances: torch.Tensor,
    starvation_ratios: torch.Tensor,
    proximity_ratios: torch.Tensor,
    current_structure_band_ratios: torch.Tensor,
    current_forward_structure_ratios: torch.Tensor,
    prev_struct: torch.Tensor,
    prev_fwd: torch.Tensor,
    *,
    progress_reward_scale: float,
    collision_penalty: float,
    obstacle_clearance_window: float,
    obstacle_clearance_reward_scale: float,
    starvation_ratio_threshold: float,
    starvation_penalty_scale: float,
    proximity_penalty_scale: float,
    structure_band_reward_scale: float,
    forward_structure_reward_scale: float,
    inspection_activation_threshold: float,
    inspection_reward_scale: float,
    starvation_dead_zone: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    progress_rewards = torch.linalg.vector_norm(current_positions - previous_positions, dim=-1)
    progress_rewards = progress_rewards * progress_reward_scale
    # Discount progress near obstacles: approaching walls yields diminishing reward.
    progress_rewards = progress_rewards * (1.0 - proximity_ratios.clamp(min=0.0, max=1.0))
    clearance_rewards = torch.zeros_like(current_clearances)
    if obstacle_clearance_window > 0.0:
        within_window_mask = (previous_clearances <= obstacle_clearance_window) | (
            current_clearances <= obstacle_clearance_window
        )
        norm_deltas = (current_clearances - previous_clearances) / obstacle_clearance_window
        clearance_rewards = (
            within_window_mask.float()
            * norm_deltas.clamp(min=-1.0, max=1.0)
            * obstacle_clearance_reward_scale
        )
    # Dead zone: starvation below starvation_dead_zone is free (normal viewing).
    effective_starvation = (starvation_ratios - starvation_dead_zone).clamp(min=0.0)
    starvation_baselines = 0.35 * effective_starvation
    # Overflow accelerates penalty above the raw-ratio threshold.
    starvation_overflows = (starvation_ratios - starvation_ratio_threshold).clamp(min=0.0)
    starvation_penalties = -(starvation_baselines + starvation_overflows).clamp(max=1.0)
    starvation_penalties = starvation_penalties * starvation_penalty_scale
    proximity_penalties = -proximity_ratios.clamp(max=1.0) * proximity_penalty_scale
    structure_rewards = (
        current_structure_band_ratios.clamp(min=0.0, max=1.0) * structure_band_reward_scale
    )
    forward_structure_rewards = current_forward_structure_ratios.clamp(min=0.0, max=1.0)
    forward_structure_rewards = forward_structure_rewards * forward_structure_reward_scale
    activation = torch.stack(
        [prev_struct, current_structure_band_ratios, prev_fwd, current_forward_structure_ratios]
    ).amax(dim=0)
    gain_deltas = (current_structure_band_ratios - prev_struct) + 0.5 * (
        current_forward_structure_ratios - prev_fwd
    )
    inspection_rewards = (activation >= inspection_activation_threshold).float()
    inspection_rewards = inspection_rewards * gain_deltas.clamp(min=-1.0, max=1.0)
    inspection_rewards = inspection_rewards * inspection_reward_scale
    # Velocity-scale collision penalty: fast crashes hurt more than gentle grazing.
    speed_norm = progress_rewards / max(progress_reward_scale, 1e-6)
    collision_penalties = collisions.float() * collision_penalty * (1.0 + speed_norm)
    components = torch.stack(
        [
            exploration_rewards,
            progress_rewards,
            clearance_rewards,
            starvation_penalties,
            proximity_penalties,
            structure_rewards,
            forward_structure_rewards,
            inspection_rewards,
            collision_penalties,
        ],
        dim=-1,
    )
    return components.sum(dim=-1), components


def _step_kinematics_tensor(
    previous_depths: torch.Tensor,
    actions_linear: torch.Tensor,
    actions_angular: torch.Tensor,
    previous_linear: torch.Tensor,
    previous_angular: torch.Tensor,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    *,
    max_distance: float,
    speed_fwd: float,
    speed_vert: float,
    speed_lat: float,
    speed_yaw: float,
    smoothing: float,
    dt: float,
    speed_limiter_distance: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actor_count = int(previous_depths.shape[0])
    span = max(1, int(previous_depths.shape[1]) // 8)
    front_left = previous_depths[:, :span, :].reshape(actor_count, -1).amin(dim=1)
    front_right = previous_depths[:, -span:, :].reshape(actor_count, -1).amin(dim=1)
    min_front = torch.minimum(front_left, front_right)
    # previous_depths are log-normalized: norm = log1p(metric) / log1p(max_distance).
    # Invert to recover metric distance for the speed limiter.
    _log_denom = math.log1p(max_distance)
    min_front_metric = torch.expm1(min_front * _log_denom)
    speed_factors = (min_front_metric / speed_limiter_distance).clamp(min=0.05, max=1.0).unsqueeze(1)

    linear_cmd = actions_linear.clone()
    linear_cmd[:, 0] *= speed_fwd
    linear_cmd[:, 1] *= speed_vert
    linear_cmd[:, 2] *= speed_lat
    linear_cmd *= speed_factors

    angular_cmd = actions_angular.clone()
    angular_cmd[:, 2] *= speed_yaw

    smooth_linear = (1.0 - smoothing) * linear_cmd + smoothing * previous_linear
    smooth_angular = (1.0 - smoothing) * angular_cmd + smoothing * previous_angular

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    fwd = smooth_linear[:, 0]
    lat = smooth_linear[:, 2]
    # Coordinate convention: yaw=0 faces -Z (matching ray direction az=0).
    # Forward = (-sin(yaw), 0, -cos(yaw)),  Right = (cos(yaw), 0, -sin(yaw))
    dx = -fwd * sin_yaw + lat * cos_yaw
    dz = -fwd * cos_yaw - lat * sin_yaw

    updated_positions = positions.clone()
    updated_positions[:, 0] += dx * dt
    updated_positions[:, 1] += smooth_linear[:, 1] * dt
    updated_positions[:, 2] += dz * dt
    updated_yaws = (yaws + smooth_angular[:, 2] * dt) % (2.0 * math.pi)
    return updated_positions, updated_yaws, smooth_linear, smooth_angular


@dataclass(frozen=True)
class SdfDagPerfSnapshot:
    """Rolling batch-step metrics for the canonical SDF/DAG runtime."""

    total_batches: int
    total_actor_steps: int
    last_actor_count: int
    last_batch_step_ms: float
    ema_batch_step_ms: float
    avg_batch_step_ms: float
    avg_actor_step_ms: float
    sps: float


@dataclass(frozen=True)
class SdfDagTensorStepBatch:
    """Tensor-native canonical rollout batch with optional publish materialization."""

    observation_tensor: Any
    reward_tensor: Any
    done_tensor: Any
    truncated_tensor: Any
    episode_id_tensor: Any
    env_id_tensor: Any
    published_observations: dict[int, DistanceMatrix]
    reward_component_tensor: Any | None = None


class SdfDagBackend(SimulatorBackend):
    """Simulator backend using GPU sphere tracing against a compiled ``.gmdag``."""

    def __init__(self, config: EnvironmentConfig) -> None:
        if not config.gmdag_file and not config.scene_pool:
            msg = "SdfDagBackend requires --gmdag-file"
            raise ValueError(msg)

        self._config = config
        self._torch = self._import_required_module("torch")
        self._torch_sdf = self._import_required_module("torch_sdf")
        if not self._torch.cuda.is_available():
            msg = "CUDA is not available. The canonical sdfdag backend does not support CPU fallback."
            raise RuntimeError(msg)

        self._device = self._torch.device("cuda")
        self._sdfdag_torch_compile_requested = bool(getattr(config, "sdfdag_torch_compile", True))
        self._sdfdag_torch_compile = self._sdfdag_torch_compile_requested
        self._az_bins = config.azimuth_bins
        self._el_bins = config.elevation_bins
        self._max_distance = config.max_distance
        self._n_actors = config.n_actors
        self._n_rays = self._az_bins * self._el_bins
        self._max_steps_per_episode = max(
            1, int(getattr(config, "max_steps_per_episode", _MAX_STEPS_PER_EPISODE))
        )
        self._scene_episodes_per_scene = max(
            1,
            int(getattr(config, "scene_episodes_per_scene", _SCENE_EPISODES_PER_SCENE)),
        )
        self._obstacle_clearance_reward_scale = float(
            getattr(config, "obstacle_clearance_reward_scale", _OBSTACLE_CLEARANCE_REWARD_SCALE)
        )
        self._obstacle_clearance_window = float(
            getattr(config, "obstacle_clearance_window", _OBSTACLE_CLEARANCE_WINDOW)
        )
        self._starvation_ratio_threshold = float(
            getattr(config, "starvation_ratio_threshold", _STARVATION_RATIO_THRESHOLD)
        )
        self._starvation_penalty_scale = float(
            getattr(config, "starvation_penalty_scale", _STARVATION_PENALTY_SCALE)
        )
        self._proximity_distance_threshold = float(
            getattr(config, "proximity_distance_threshold", _PROXIMITY_DISTANCE_THRESHOLD)
        )
        self._proximity_penalty_scale = float(
            getattr(config, "proximity_penalty_scale", _PROXIMITY_PENALTY_SCALE)
        )
        self._structure_band_min_distance = float(
            getattr(config, "structure_band_min_distance", _STRUCTURE_BAND_MIN_DISTANCE)
        )
        self._structure_band_max_distance = float(
            getattr(config, "structure_band_max_distance", _STRUCTURE_BAND_MAX_DISTANCE)
        )
        self._structure_band_reward_scale = float(
            getattr(config, "structure_band_reward_scale", _STRUCTURE_BAND_REWARD_SCALE)
        )
        self._forward_structure_reward_scale = float(
            getattr(config, "forward_structure_reward_scale", _FORWARD_STRUCTURE_REWARD_SCALE)
        )
        self._inspection_reward_scale = float(
            getattr(config, "inspection_reward_scale", _INSPECTION_REWARD_SCALE)
        )
        self._inspection_activation_threshold = float(
            getattr(config, "inspection_activation_threshold", _INSPECTION_ACTIVATION_THRESHOLD)
        )
        self._collision_clearance = float(
            getattr(config, "collision_clearance", _COLLISION_CLEARANCE)
        )
        self._speed_limiter_distance = float(
            getattr(config, "speed_limiter_distance", 0.8)
        )
        self._scene_pool: list[str] = list(config.scene_pool) if config.scene_pool else []
        self._scene_pool_idx: int = 0
        self._episodes_in_scene: int = 0
        self._initial_resets_remaining: int = self._n_actors
        self._asset: GmDagAsset | None = None
        self._dag_tensor: Any | None = None
        self._bbox_min: list[float] = []
        self._bbox_max: list[float] = []
        self._postprocess_cast_outputs_compiled = _maybe_compile_callable(
            self._torch,
            _postprocess_cast_outputs_tensor,
            enabled=self._sdfdag_torch_compile,
            name="SdfDagBackend post-cast tensor path",
        )
        self._reward_components_compiled = _maybe_compile_callable(
            self._torch,
            _reward_components_tensor,
            enabled=self._sdfdag_torch_compile,
            name="SdfDagBackend reward tensor path",
        )
        self._step_kinematics_compiled = _maybe_compile_callable(
            self._torch,
            _step_kinematics_tensor,
            enabled=self._sdfdag_torch_compile,
            name="SdfDagBackend kinematics tensor path",
        )
        if self._sdfdag_torch_compile:
            _LOG.info(
                "SdfDagBackend: torch.compile enabled for tensor-only post-cast, reward, and kinematics helpers"
            )

        self._ray_dirs_local = self._build_ray_directions().to(self._device)
        _validate_unit_direction_tensor(self._torch, self._ray_dirs_local, name="ray_dirs_local")
        self._scratch_slot_count = _SCRATCH_SLOT_COUNT
        self._ray_origins = self._torch.empty(
            (self._scratch_slot_count, self._n_actors, self._n_rays, 3),
            device=self._device,
            dtype=self._torch.float32,
        )
        self._ray_dirs_world = self._torch.empty_like(self._ray_origins)
        self._out_distances = self._torch.empty(
            (self._scratch_slot_count, self._n_actors, self._n_rays),
            device=self._device,
            dtype=self._torch.float32,
        )
        self._out_semantics = self._torch.empty(
            (self._scratch_slot_count, self._n_actors, self._n_rays),
            device=self._device,
            dtype=self._torch.int32,
        )
        self._perf_total_batches = 0
        self._perf_total_actor_steps = 0
        self._perf_total_batch_seconds = 0.0
        self._perf_last_batch_seconds = 0.0
        self._perf_last_actor_count = 0
        self._perf_ema_batch_seconds = 0.0

        # Pre-allocate contiguous state tensors on the active device.
        self._actor_positions = self._torch.zeros(
            (self._n_actors, 3), device=self._device, dtype=self._torch.float32
        )
        self._actor_yaws = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )
        self._actor_steps = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.int32
        )
        self._episode_returns = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )
        self._episode_ids = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.int64
        )
        self._needs_reset_mask = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.bool
        )
        self._prev_min_distances = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )
        self._prev_structure_band_ratios = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )
        self._prev_forward_structure_ratios = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )
        self._prev_depth_tensors = self._torch.zeros(
            (self._n_actors, self._az_bins, self._el_bins),
            device=self._device,
            dtype=self._torch.float32,
        )
        self._spawn_positions = self._torch.zeros(
            (self._n_actors, 3), device=self._device, dtype=self._torch.float32
        )
        self._spawn_yaws = self._torch.zeros(
            (self._n_actors,), device=self._device, dtype=self._torch.float32
        )

        # Phase 3: Vectorized kinematics state
        self._prev_linear_vels = self._torch.zeros(
            (self._n_actors, 3), device=self._device, dtype=self._torch.float32
        )
        # Heading novelty: track visited heading sectors per actor per episode
        self._heading_visited = self._torch.zeros(
            (self._n_actors, _HEADING_SECTORS), device=self._device, dtype=self._torch.bool
        )
        self._prev_angular_vels = self._torch.zeros(
            (self._n_actors, 3), device=self._device, dtype=self._torch.float32
        )
        self._smoothing = 0.15
        self._dt = config.physics_dt
        self._speed_fwd = config.drone_max_speed
        self._speed_vert = config.drone_climb_rate
        self._speed_lat = config.drone_strafe_speed
        self._speed_yaw = config.drone_yaw_rate

        self._grid_res = 2.0
        self._grid_res_y = 2.0
        self._visit_grid = self._torch.zeros(
            (self._n_actors, 1, 1, 1), device=self._device, dtype=self._torch.int16
        )
        self._grid_min = self._torch.zeros((3,), device=self._device, dtype=self._torch.float32)
        self._scene_bbox_min_t = self._torch.zeros((3,), device=self._device, dtype=self._torch.float32)
        self._scene_bbox_max_t = self._torch.zeros((3,), device=self._device, dtype=self._torch.float32)

        scene_path = self._scene_pool[0] if self._scene_pool else config.gmdag_file
        self._load_scene(scene_path)

        # Initial placement
        for actor_id in range(self._n_actors):
            spawn = self._spawn_positions[actor_id]
            yaw = (2.0 * math.pi * actor_id) / max(self._n_actors, 1)
            self._actor_positions[actor_id].copy_(spawn)
            self._actor_yaws[actor_id] = yaw

        if self._scene_pool:
            _LOG.info(
                "SdfDagBackend scene pool: %d scenes, starting with %s (episode budget per scene=%d, max_steps=%d)",
                len(self._scene_pool),
                Path(scene_path).name,
                self._scene_episodes_per_scene,
                self._max_steps_per_episode,
            )

    def torch_compile_enabled(self) -> bool:
        return bool(getattr(self, "_sdfdag_torch_compile", False))

    def _maybe_rotate_scene(self, *, is_natural: bool) -> bool:
        """Rotate scene if budget met. Returns True if scene was swapped."""
        if not is_natural or not self._scene_pool:
            return False

        self._episodes_in_scene += 1
        # Total episodes to run in this scene before swapping
        total_budget = self._scene_episodes_per_scene * self._n_actors
        if self._episodes_in_scene < total_budget:
            return False

        self._scene_pool_idx = (self._scene_pool_idx + 1) % len(self._scene_pool)
        self._load_scene(self._scene_pool[self._scene_pool_idx])
        self._episodes_in_scene = 0

        # Phase 6: Global Synchronization.
        # Force ALL actors to reset on the next step so they all start in the new scene together.
        self._needs_reset_mask[:] = True
        _LOG.info(
            "SdfDagBackend: Scene budget met. Rotating to %s and forcing global reset.",
            Path(self._scene_pool[self._scene_pool_idx]).name,
        )
        return True

    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        is_natural = self._initial_resets_remaining <= 0
        if not is_natural:
            self._initial_resets_remaining -= 1

        self._maybe_rotate_scene(is_natural=is_natural)

        spawn = self._spawn_positions[actor_id]
        yaw = self._spawn_yaws[actor_id]

        self._actor_positions[actor_id].copy_(spawn)
        self._actor_yaws[actor_id] = yaw
        self._episode_ids[actor_id] = episode_id
        self._actor_steps[actor_id] = 0
        self._prev_depth_tensors[actor_id].zero_()
        self._prev_min_distances[actor_id] = 0.0
        self._prev_structure_band_ratios[actor_id] = 0.0
        self._prev_forward_structure_ratios[actor_id] = 0.0
        self._episode_returns[actor_id] = 0.0
        self._visit_grid[actor_id].zero_()
        self._heading_visited[actor_id].zero_()
        self._prev_linear_vels[actor_id].zero_()
        self._prev_angular_vels[actor_id].zero_()
        self._needs_reset_mask[actor_id] = False

        depth_batch, semantic_batch, valid_batch, cm = self._cast_actor_batch_tensors((actor_id,))
        _obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
            actor_id=actor_id,
            step_id=0,
            current_clearance=float(cm[0].amin().detach().cpu()),
            depth_2d=depth_batch[0],
            semantic_2d=semantic_batch[0],
            valid_2d=valid_batch[0],
            materialize_depth_cpu=True,
        )
        assert depth_cpu is not None
        # Convert log-normalized depth_cpu back to metric for observation profile.
        metric_cpu = np.expm1(depth_cpu * math.log1p(self._max_distance)).astype(
            np.float32, copy=False
        )
        _starvation_ratio, _proximity_ratio, structure_band_ratio, forward_structure_ratio = (
            _observation_profile(
                metric_cpu,
                valid_batch[0].detach().cpu().numpy().astype(np.bool_, copy=False),
                max_distance=self._max_distance,
                proximity_distance_threshold=self._proximity_distance_threshold,
                structure_band_min_distance=self._structure_band_min_distance,
                structure_band_max_distance=self._structure_band_max_distance,
            )
        )
        self._prev_structure_band_ratios[actor_id] = structure_band_ratio
        self._prev_forward_structure_ratios[actor_id] = forward_structure_ratio
        return self._materialize_observation(
            actor_id=actor_id,
            step_id=0,
            depth_2d=depth_batch[0].detach().cpu().numpy().astype(np.float32, copy=False),
            delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
            semantic_2d=semantic_batch[0].detach().cpu().numpy().astype(np.int32, copy=False),
            valid_2d=valid_batch[0].detach().cpu().numpy().astype(np.bool_, copy=False),
        )

    def reset_tensor(
        self,
        episode_id: int,
        *,
        actor_id: int = 0,
        materialize: bool = False,
    ) -> tuple[Any, DistanceMatrix | None]:
        """Reset one actor and return a tensor-native observation for canonical training."""
        is_natural = self._initial_resets_remaining <= 0
        if not is_natural:
            self._initial_resets_remaining -= 1

        self._maybe_rotate_scene(is_natural=is_natural)

        spawn = self._spawn_positions[actor_id]
        yaw = self._spawn_yaws[actor_id]

        self._actor_positions[actor_id].copy_(spawn)
        self._actor_yaws[actor_id] = yaw
        self._episode_ids[actor_id] = episode_id
        self._actor_steps[actor_id] = 0
        self._prev_depth_tensors[actor_id].zero_()
        self._prev_min_distances[actor_id] = 0.0
        self._prev_structure_band_ratios[actor_id] = 0.0
        self._prev_forward_structure_ratios[actor_id] = 0.0
        self._episode_returns[actor_id] = 0.0
        self._visit_grid[actor_id].zero_()
        self._heading_visited[actor_id].zero_()
        self._prev_linear_vels[actor_id].zero_()
        self._prev_angular_vels[actor_id].zero_()
        self._needs_reset_mask[actor_id] = False

        depth_batch, semantic_batch, valid_batch, clamped_metric_batch = self._cast_actor_batch_tensors((actor_id,))
        current_clearance = clamped_metric_batch[0].amin()
        obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
            actor_id=actor_id,
            step_id=0,
            current_clearance=current_clearance,
            depth_2d=depth_batch[0],
            semantic_2d=semantic_batch[0],
            valid_2d=valid_batch[0],
            materialize_depth_cpu=materialize,
        )
        (
            _starvation_ratios,
            _proximity_ratios,
            structure_band_ratios,
            forward_structure_ratios,
        ) = self._compute_observation_ratios(clamped_metric=clamped_metric_batch, valid_batch=valid_batch)
        self._prev_structure_band_ratios[actor_id] = structure_band_ratios[0]
        self._prev_forward_structure_ratios[actor_id] = forward_structure_ratios[0]
        published = None
        if materialize:
            assert depth_cpu is not None
            published = self._materialize_observation(
                actor_id=actor_id,
                step_id=0,
                depth_2d=depth_batch[0].detach().cpu().numpy().astype(np.float32, copy=False),
                delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
                semantic_2d=semantic_batch[0].detach().cpu().numpy().astype(np.int32, copy=False),
                valid_2d=valid_batch[0].detach().cpu().numpy().astype(np.bool_, copy=False),
            )
        return obs_tensor, published

    def step(
        self,
        action: Action,
        step_id: int,
        *,
        actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        observations, results = self.batch_step((action,), step_id)
        return observations[0], results[0]

    def batch_step(
        self,
        actions: tuple[Action, ...],
        step_id: int,
    ) -> tuple[tuple[DistanceMatrix, ...], tuple[StepResult, ...]]:
        batch_started_at = time.perf_counter()
        observations_by_actor: dict[int, DistanceMatrix] = {}
        results_by_actor: dict[int, StepResult] = {}
        active_actor_ids: list[int] = []
        previous_poses: dict[int, RobotPose] = {}

        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            if self._needs_reset_mask[actor_id]:
                obs = self.reset(int(self._episode_ids[actor_id]) + 1, actor_id=actor_id)
                observations_by_actor[actor_id] = obs
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=int(self._episode_ids[actor_id]),
                    done=False,
                    truncated=False,
                    reward=0.0,
                    episode_return=0.0,
                    timestamp=time.time(),
                )
                continue

            previous_poses[actor_id] = self.actor_pose(actor_id)
            active_actor_ids.append(actor_id)

        if active_actor_ids:
            lin_actions_list = []
            ang_actions_list = []
            for action in actions:
                aid = int(action.env_ids[0]) if len(action.env_ids) > 0 else 0
                if aid in active_actor_ids:
                    lin = (
                        action.linear_velocity[0]
                        if action.linear_velocity.ndim == 2
                        else action.linear_velocity
                    )
                    ang = (
                        action.angular_velocity[0]
                        if action.angular_velocity.ndim == 2
                        else action.angular_velocity
                    )
                    lin_actions_list.append(lin)
                    ang_actions_list.append(ang)

            lin_actions = self._torch.from_numpy(
                np.asarray(lin_actions_list, dtype=np.float32)
            ).to(self._device)
            ang_actions = self._torch.from_numpy(
                np.asarray(ang_actions_list, dtype=np.float32)
            ).to(self._device)

            self._step_kinematics_batch(
                actor_ids=tuple(active_actor_ids),
                actions_linear=lin_actions,
                actions_angular=ang_actions,
            )

            depth_batch, semantic_batch, valid_batch, metric_batch = self._cast_actor_batch(
                tuple(active_actor_ids)
            )
            all_min_distances = np.min(metric_batch, axis=(1, 2))

            for batch_idx, actor_id in enumerate(active_actor_ids):
                previous_pose = previous_poses[actor_id]
                depth_2d = depth_batch[batch_idx]
                semantic_2d = semantic_batch[batch_idx]
                valid_2d = valid_batch[batch_idx]
                metric_2d = metric_batch[batch_idx]

                min_distance = float(all_min_distances[batch_idx])
                collision = min_distance < self._collision_clearance
                previous_clearance = float(self._prev_min_distances[actor_id])

                if collision:
                    self._actor_positions[actor_id].copy_(
                        self._torch.tensor(
                            [previous_pose.x, previous_pose.y, previous_pose.z],
                            device=self._device,
                        )
                    )
                    self._actor_yaws[actor_id] = previous_pose.yaw
                    corrected_depth, corrected_semantic, corrected_valid, corrected_metric = (
                        self._cast_actor_batch((actor_id,))
                    )
                    depth_2d = corrected_depth[0]
                    semantic_2d = corrected_semantic[0]
                    valid_2d = corrected_valid[0]
                    metric_2d = corrected_metric[0]
                    min_distance = float(np.min(metric_2d))

                (
                    starvation_ratio,
                    proximity_ratio,
                    structure_band_ratio,
                    forward_structure_ratio,
                ) = _observation_profile(
                    metric_2d,
                    valid_2d,
                    max_distance=self._max_distance,
                    proximity_distance_threshold=self._proximity_distance_threshold,
                    structure_band_min_distance=self._structure_band_min_distance,
                    structure_band_max_distance=self._structure_band_max_distance,
                )

                obs = self._build_observation(
                    actor_id=actor_id,
                    step_id=step_id,
                    depth_2d=depth_2d,
                    semantic_2d=semantic_2d,
                    valid_2d=valid_2d,
                    min_distance_metric=min_distance,
                )

                self._actor_steps[actor_id] += 1
                truncated = bool(self._actor_steps[actor_id] >= self._max_steps_per_episode)
                # Void-detection: force early truncation for lost actors.
                if starvation_ratio >= _VOID_STARVATION_THRESHOLD:
                    truncated = True
                reward, _reward_components = self._compute_reward(
                    actor_id=actor_id,
                    previous_pose=previous_pose,
                    current_pose=self.actor_pose(actor_id),
                    collision=collision,
                    previous_clearance=previous_clearance,
                    current_clearance=min_distance,
                    starvation_ratio=starvation_ratio,
                    proximity_ratio=proximity_ratio,
                    current_structure_band_ratio=structure_band_ratio,
                    current_forward_structure_ratio=forward_structure_ratio,
                )
                self._prev_structure_band_ratios[actor_id] = structure_band_ratio
                self._prev_forward_structure_ratios[actor_id] = forward_structure_ratio
                self._episode_returns[actor_id] += reward
                self._needs_reset_mask[actor_id] = truncated

                observations_by_actor[actor_id] = obs
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=int(self._episode_ids[actor_id]),
                    done=False,
                    truncated=truncated,
                    reward=reward,
                    episode_return=float(self._episode_returns[actor_id]),
                    timestamp=time.time(),
                )

        ordered_observations: list[DistanceMatrix] = []
        ordered_results: list[StepResult] = []
        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            ordered_observations.append(observations_by_actor[actor_id])
            ordered_results.append(results_by_actor[actor_id])

        self._record_perf_sample(
            batch_seconds=time.perf_counter() - batch_started_at,
            actor_count=len(actions),
        )
        return tuple(ordered_observations), tuple(ordered_results)

    def batch_step_tensor_actions(
        self,
        action_tensor: Any,
        step_id: int,
        *,
        actor_indices: Any | None = None,
        scratch_slot: int = 0,
        publish_actor_ids: tuple[int, ...] = (),
        materialize_results: bool = False,
    ) -> tuple[SdfDagTensorStepBatch, tuple[StepResult, ...]]:
        """Step actors from a tensor/array command batch without Python Action objects."""
        action_rows = self._coerce_action_tensor_rows(action_tensor)
        batch_started_at = time.perf_counter()
        published_observations: dict[int, DistanceMatrix] = {}
        publish_actor_set = set(publish_actor_ids)
        actor_count = int(action_rows.shape[0])
        if actor_count > self._n_actors:
            msg = f"action_tensor batch size {actor_count} exceeds configured actors {self._n_actors}"
            raise ValueError(msg)
        batch_actor_indices_t = self._coerce_batch_actor_indices(
            actor_count=actor_count, actor_indices=actor_indices
        )
        observation_tensor = self._torch.empty(
            (actor_count, 3, self._az_bins, self._el_bins),
            device=self._device,
            dtype=self._torch.float32,
        )
        populated_mask = self._torch.zeros(
            (actor_count,), device=self._device, dtype=self._torch.bool
        )
        reward_tensor = self._torch.zeros(
            (actor_count,), device=self._device, dtype=self._torch.float32
        )
        done_tensor = self._torch.zeros(
            (actor_count,), device=self._device, dtype=self._torch.bool
        )
        truncated_tensor = self._torch.zeros(
            (actor_count,), device=self._device, dtype=self._torch.bool
        )
        episode_id_tensor = self._torch.zeros(
            (actor_count,), device=self._device, dtype=self._torch.int64
        )
        env_id_tensor = batch_actor_indices_t.clone()
        ordered_results: list[StepResult | None] | None = (
            [None] * actor_count if materialize_results else None
        )
        reward_component_tensor = self._torch.zeros(
            (actor_count, 9), device=self._device, dtype=self._torch.float32
        )
        reset_local_indices_t = (
            self._needs_reset_mask.index_select(0, batch_actor_indices_t)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if reset_local_indices_t.numel() > 0:
            self._advance_reset_bookkeeping(int(reset_local_indices_t.numel()))
            reset_local_indices_t = (
                self._needs_reset_mask.index_select(0, batch_actor_indices_t)
                .nonzero(as_tuple=False)
                .flatten()
            )
            if reset_local_indices_t.numel() > 0:
                self._reset_tensor_actor_batch(
                    actor_indices=batch_actor_indices_t.index_select(0, reset_local_indices_t),
                    local_rows=reset_local_indices_t,
                    scratch_slot=scratch_slot,
                    step_id=step_id,
                    publish_actor_set=publish_actor_set,
                    observation_tensor=observation_tensor,
                    populated_mask=populated_mask,
                    reward_tensor=reward_tensor,
                    done_tensor=done_tensor,
                    truncated_tensor=truncated_tensor,
                    episode_id_tensor=episode_id_tensor,
                    ordered_results=ordered_results,
                    published_observations=published_observations,
                )

        active_local_indices_t = (
            self._needs_reset_mask.index_select(0, batch_actor_indices_t)
            .logical_not()
            .nonzero(as_tuple=False)
            .flatten()
        )
        if active_local_indices_t.numel() > 0:
            actor_indices_t = batch_actor_indices_t.index_select(0, active_local_indices_t)
            previous_positions_t = self._actor_positions.index_select(0, actor_indices_t).clone()
            previous_yaws_t = self._actor_yaws.index_select(0, actor_indices_t).clone()

            active_actions = action_rows.index_select(0, active_local_indices_t)
            lin_actions = active_actions[:, :3].contiguous()
            ang_actions = self._torch.zeros(
                (int(actor_indices_t.shape[0]), 3), device=self._device, dtype=self._torch.float32
            )
            ang_actions[:, 2] = active_actions[:, 3]

            self._step_kinematics_indexed(
                actor_indices=actor_indices_t,
                actions_linear=lin_actions,
                actions_angular=ang_actions,
            )

            active_count = int(actor_indices_t.shape[0])
            yaws = self._actor_yaws.index_select(0, actor_indices_t)
            positions = self._actor_positions.index_select(0, actor_indices_t)
            cos_yaw = self._torch.cos(yaws).unsqueeze(1)
            sin_yaw = self._torch.sin(yaws).unsqueeze(1)
            base_dirs = self._ray_dirs_local.unsqueeze(0).expand(active_count, -1, -1)
            origins, dirs_world, out_distances, out_semantics = self._scratch_slot_views(
                scratch_slot=scratch_slot,
                actor_count=active_count,
            )
            dirs_world[..., 0] = base_dirs[..., 0] * cos_yaw + base_dirs[..., 2] * sin_yaw
            dirs_world[..., 1] = base_dirs[..., 1]
            dirs_world[..., 2] = -base_dirs[..., 0] * sin_yaw + base_dirs[..., 2] * cos_yaw

            origins.copy_(positions.unsqueeze(1).expand(-1, self._n_rays, -1))

            self._torch_sdf.cast_rays(
                self._require_dag_tensor(),
                origins,
                dirs_world,
                out_distances,
                out_semantics,
                self._config.sdf_max_steps,
                self._max_distance,
                self._bbox_min,
                self._bbox_max,
                self._require_asset().resolution,
                skip_direction_validation=True,
            )

            (
                depth_batch,
                semantic_batch,
                valid_batch,
                min_distances_t,
                starvation_ratios_t,
                proximity_ratios_t,
                structure_band_ratios_t,
                forward_structure_ratios_t,
                _clamped_metric_t,
            ) = self._postprocess_cast_outputs(out_distances, out_semantics)

            collisions_t = min_distances_t < self._collision_clearance
            current_positions_t = self._torch.where(
                collisions_t.unsqueeze(1), previous_positions_t, positions
            )
            current_yaws_t = self._torch.where(collisions_t, previous_yaws_t, yaws)
            self._actor_positions[actor_indices_t] = current_positions_t
            self._actor_yaws[actor_indices_t] = current_yaws_t

            prev_clearances_t = self._prev_min_distances.index_select(0, actor_indices_t)
            rewards_t, components_t = self._compute_reward_batch(
                actor_indices=actor_indices_t,
                previous_positions=previous_positions_t,
                current_positions=current_positions_t,
                collisions=collisions_t,
                previous_clearances=prev_clearances_t,
                current_clearances=min_distances_t,
                starvation_ratios=starvation_ratios_t,
                proximity_ratios=proximity_ratios_t,
                current_structure_band_ratios=structure_band_ratios_t,
                current_forward_structure_ratios=forward_structure_ratios_t,
            )

            obs_batch_t, deltas_batch_t = self._consume_observation_batch(
                actor_indices=actor_indices_t,
                depth_batch=depth_batch,
                semantic_batch=semantic_batch,
                valid_batch=valid_batch,
                current_clearances=min_distances_t,
            )

            next_steps_t = self._actor_steps.index_select(0, actor_indices_t) + 1
            truncated_mask_t = next_steps_t >= self._max_steps_per_episode
            # Void-detection: force early truncation when the actor is lost in
            # empty space and wasting training steps.
            void_mask = starvation_ratios_t >= _VOID_STARVATION_THRESHOLD
            truncated_mask_t = truncated_mask_t | void_mask
            self._actor_steps[actor_indices_t] = next_steps_t
            self._needs_reset_mask[actor_indices_t] = truncated_mask_t
            self._episode_returns[actor_indices_t] = (
                self._episode_returns.index_select(0, actor_indices_t) + rewards_t
            )
            self._prev_structure_band_ratios[actor_indices_t] = structure_band_ratios_t
            self._prev_forward_structure_ratios[actor_indices_t] = forward_structure_ratios_t

            result_rows = (
                self._materialize_step_result_rows(
                    local_rows=active_local_indices_t,
                    env_ids=actor_indices_t,
                    actor_indices=actor_indices_t,
                    rewards=rewards_t,
                    truncated_mask=truncated_mask_t,
                )
                if materialize_results
                else None
            )
            reward_component_tensor[active_local_indices_t] = components_t
            observation_tensor[active_local_indices_t] = obs_batch_t
            populated_mask[active_local_indices_t] = True
            reward_tensor[active_local_indices_t] = rewards_t
            done_tensor[active_local_indices_t] = False
            truncated_tensor[active_local_indices_t] = truncated_mask_t
            episode_id_tensor[active_local_indices_t] = self._episode_ids.index_select(
                0, actor_indices_t
            )

            publish_rows: list[int] = []
            publish_actor_ids_list: list[int] = []
            if publish_actor_set:
                publish_rows, publish_actor_ids_list = self._select_publish_rows(
                    actor_indices=actor_indices_t,
                    publish_actor_ids=tuple(sorted(publish_actor_set)),
                )
            if result_rows is not None and ordered_results is not None:
                for row in result_rows:
                    local_row = int(row[0])
                    actor_id = int(row[1])
                    ordered_results[local_row] = StepResult(
                        step_id=step_id,
                        env_id=actor_id,
                        episode_id=int(row[2]),
                        done=False,
                        truncated=bool(row[3]),
                        reward=float(row[4]),
                        episode_return=float(row[5]),
                        timestamp=time.time(),
                    )

            if publish_actor_ids_list:
                self._materialize_selected_observations(
                    actor_ids=publish_actor_ids_list,
                    row_indices=publish_rows,
                    step_id=step_id,
                    depth_batch=depth_batch,
                    delta_batch=deltas_batch_t,
                    semantic_batch=semantic_batch,
                    valid_batch=valid_batch,
                    published_observations=published_observations,
                )

        self._record_perf_sample(
            batch_seconds=time.perf_counter() - batch_started_at,
            actor_count=actor_count,
        )
        finalized_results = tuple(
            result for result in (ordered_results or ()) if result is not None
        )
        return (
            SdfDagTensorStepBatch(
                observation_tensor=observation_tensor,
                reward_tensor=reward_tensor,
                done_tensor=done_tensor,
                truncated_tensor=truncated_tensor,
                episode_id_tensor=episode_id_tensor,
                env_id_tensor=env_id_tensor,
                reward_component_tensor=reward_component_tensor,
                published_observations=published_observations,
            ),
            finalized_results,
        )

    def _postprocess_cast_outputs(
        self, out_distances: Any, out_semantics: Any
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        impl = getattr(
            self, "_postprocess_cast_outputs_compiled", _postprocess_cast_outputs_tensor
        )
        return impl(
            out_distances,
            out_semantics,
            az_bins=self._az_bins,
            el_bins=self._el_bins,
            max_distance=self._max_distance,
            structure_band_min_distance=self._structure_band_min_distance,
            structure_band_max_distance=self._structure_band_max_distance,
            proximity_distance_threshold=self._proximity_distance_threshold,
        )

    def _compute_observation_ratios(
        self, *, clamped_metric: Any, valid_batch: Any
    ) -> tuple[Any, Any, Any, Any]:
        return _compute_observation_ratios_tensor(
            clamped_metric,
            valid_batch,
            max_distance=float(self._max_distance),
            structure_band_min_distance=float(
                getattr(self, "_structure_band_min_distance", _STRUCTURE_BAND_MIN_DISTANCE)
            ),
            structure_band_max_distance=float(
                getattr(self, "_structure_band_max_distance", _STRUCTURE_BAND_MAX_DISTANCE)
            ),
            proximity_distance_threshold=float(
                getattr(self, "_proximity_distance_threshold", _PROXIMITY_DISTANCE_THRESHOLD)
            ),
        )

    def _consume_observation_batch(
        self,
        *,
        actor_indices: Any,
        depth_batch: Any,
        semantic_batch: Any,
        valid_batch: Any,
        current_clearances: Any,
    ) -> tuple[Any, Any]:
        previous_depths = self._prev_depth_tensors.index_select(0, actor_indices)
        uninitialized = previous_depths.abs().sum(dim=(1, 2)) < 1e-6
        delta_batch = depth_batch - previous_depths
        delta_batch = self._torch.where(
            uninitialized.view(-1, 1, 1), self._torch.zeros_like(delta_batch), delta_batch
        )
        self._prev_depth_tensors[actor_indices] = depth_batch
        self._prev_min_distances[actor_indices] = current_clearances
        observation_batch = self._torch.stack(
            (
                depth_batch,
                semantic_batch.to(dtype=self._torch.float32),
                valid_batch.to(dtype=self._torch.float32),
            ),
            dim=1,
        )
        return observation_batch, delta_batch

    def _materialize_selected_observations(
        self,
        *,
        actor_ids: list[int],
        row_indices: list[int],
        step_id: int,
        depth_batch: Any,
        delta_batch: Any,
        semantic_batch: Any,
        valid_batch: Any,
        published_observations: dict[int, DistanceMatrix],
    ) -> None:
        for actor_id, row_index in zip(actor_ids, row_indices, strict=False):
            published_observations[actor_id] = self._materialize_observation(
                actor_id=actor_id,
                step_id=step_id,
                depth_2d=depth_batch[row_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                delta_2d=delta_batch[row_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                semantic_2d=semantic_batch[row_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32, copy=False),
                valid_2d=valid_batch[row_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.bool_, copy=False),
            )

    def _select_publish_rows(
        self,
        *,
        actor_indices: Any,
        publish_actor_ids: tuple[int, ...],
    ) -> tuple[list[int], list[int]]:
        if int(actor_indices.numel()) == 0 or not publish_actor_ids:
            return [], []
        publish_actor_tensor = self._torch.as_tensor(
            publish_actor_ids,
            device=actor_indices.device,
            dtype=self._torch.int64,
        )
        publish_mask = (actor_indices.unsqueeze(1) == publish_actor_tensor.unsqueeze(0)).any(dim=1)
        publish_rows_t = publish_mask.nonzero(as_tuple=False).flatten()
        if int(publish_rows_t.numel()) == 0:
            return [], []
        selected_actor_ids_t = actor_indices.index_select(0, publish_rows_t)
        return (
            publish_rows_t.detach().to(device="cpu", dtype=self._torch.int64).tolist(),
            selected_actor_ids_t.detach().to(device="cpu", dtype=self._torch.int64).tolist(),
        )

    def _materialize_step_result_rows(
        self,
        *,
        local_rows: Any,
        env_ids: Any,
        actor_indices: Any,
        rewards: Any,
        truncated_mask: Any,
    ) -> np.ndarray:
        rows = self._torch.stack(
            (
                local_rows.to(dtype=self._torch.float32),
                env_ids.to(dtype=self._torch.float32),
                self._episode_ids.index_select(0, actor_indices).to(dtype=self._torch.float32),
                truncated_mask.to(dtype=self._torch.float32),
                rewards.to(dtype=self._torch.float32),
                self._episode_returns.index_select(0, actor_indices).to(dtype=self._torch.float32),
            ),
            dim=1,
        )
        return rows.detach().cpu().numpy().astype(np.float32, copy=False)

    def _materialize_reset_result_rows(
        self,
        *,
        local_rows: Any,
        env_ids: Any,
        actor_indices: Any,
    ) -> np.ndarray:
        rows = self._torch.stack(
            (
                local_rows.to(dtype=self._torch.float32),
                env_ids.to(dtype=self._torch.float32),
                self._episode_ids.index_select(0, actor_indices).to(dtype=self._torch.float32),
            ),
            dim=1,
        )
        return rows.detach().cpu().numpy().astype(np.float32, copy=False)

    def _advance_reset_bookkeeping(self, reset_count: int) -> None:
        """Advance per-episode scene bookkeeping before batched reset state writes."""
        for _ in range(reset_count):
            is_natural = self._initial_resets_remaining <= 0
            if not is_natural:
                self._initial_resets_remaining -= 1
            self._maybe_rotate_scene(is_natural=is_natural)

    def _reset_tensor_actor_batch(
        self,
        *,
        actor_indices: Any,
        local_rows: Any,
        scratch_slot: int,
        step_id: int,
        publish_actor_set: set[int],
        observation_tensor: Any,
        populated_mask: Any,
        reward_tensor: Any,
        done_tensor: Any,
        truncated_tensor: Any,
        episode_id_tensor: Any,
        ordered_results: list[StepResult | None] | None,
        published_observations: dict[int, DistanceMatrix],
    ) -> None:
        """Reset a selected actor batch entirely on-device and seed fresh observations once."""
        if actor_indices.numel() == 0:
            return

        self._actor_positions[actor_indices] = self._spawn_positions.index_select(0, actor_indices)
        self._actor_yaws[actor_indices] = self._spawn_yaws.index_select(0, actor_indices)
        self._episode_ids[actor_indices] = self._episode_ids.index_select(0, actor_indices) + 1
        self._actor_steps[actor_indices] = 0
        self._prev_depth_tensors[actor_indices] = 0.0
        self._prev_min_distances[actor_indices] = 0.0
        self._prev_structure_band_ratios[actor_indices] = 0.0
        self._prev_forward_structure_ratios[actor_indices] = 0.0
        self._episode_returns[actor_indices] = 0.0
        self._visit_grid[actor_indices] = 0
        self._heading_visited[actor_indices] = False
        self._prev_linear_vels[actor_indices] = 0.0
        self._prev_angular_vels[actor_indices] = 0.0
        self._needs_reset_mask[actor_indices] = False

        depth_batch, semantic_batch, valid_batch, clamped_metric_batch = self._cast_actor_batch_indexed_tensors(
            actor_indices,
            scratch_slot=scratch_slot,
        )
        current_clearances = clamped_metric_batch.amin(dim=(1, 2))
        obs_batch_t, deltas_batch_t = self._consume_observation_batch(
            actor_indices=actor_indices,
            depth_batch=depth_batch,
            semantic_batch=semantic_batch,
            valid_batch=valid_batch,
            current_clearances=current_clearances,
        )
        _starvation, _proximity, structure_band_ratios_t, forward_structure_ratios_t = (
            self._compute_observation_ratios(
                clamped_metric=clamped_metric_batch,
                valid_batch=valid_batch,
            )
        )
        self._prev_structure_band_ratios[actor_indices] = structure_band_ratios_t
        self._prev_forward_structure_ratios[actor_indices] = forward_structure_ratios_t

        observation_tensor[local_rows] = obs_batch_t
        populated_mask[local_rows] = True
        reward_tensor[local_rows] = 0.0
        done_tensor[local_rows] = False
        truncated_tensor[local_rows] = False
        episode_id_tensor[local_rows] = self._episode_ids.index_select(0, actor_indices)

        result_rows = (
            self._materialize_reset_result_rows(
                local_rows=local_rows,
                env_ids=actor_indices,
                actor_indices=actor_indices,
            )
            if ordered_results is not None
            else None
        )
        publish_rows: list[int] = []
        publish_actor_ids_list: list[int] = []
        actor_ids = (
            actor_indices.detach().to(device="cpu", dtype=self._torch.int64).tolist()
            if publish_actor_set
            else []
        )
        for i, actor_id in enumerate(actor_ids):
            if actor_id in publish_actor_set:
                publish_rows.append(i)
                publish_actor_ids_list.append(actor_id)
        if result_rows is not None and ordered_results is not None:
            for row in result_rows:
                local_row = int(row[0])
                actor_id = int(row[1])
                ordered_results[local_row] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=int(row[2]),
                    done=False,
                    truncated=False,
                    reward=0.0,
                    episode_return=0.0,
                    timestamp=time.time(),
                )

        self._materialize_selected_observations(
            actor_ids=publish_actor_ids_list,
            row_indices=publish_rows,
            step_id=step_id,
            depth_batch=depth_batch,
            delta_batch=deltas_batch_t,
            semantic_batch=semantic_batch,
            valid_batch=valid_batch,
            published_observations=published_observations,
        )

    def _build_reward_component_tensor(self, ordered_components: list[np.ndarray]) -> Any | None:
        """Pack reward decomposition terms for low-volume trainer telemetry."""
        if not ordered_components:
            return None
        return self._torch.tensor(
            ordered_components,
            device=self._device,
            dtype=self._torch.float32,
        )

    def _build_result_tensors(
        self,
        ordered_results: list[StepResult],
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Pack per-actor step outcomes into canonical tensor-native rollout fields."""
        if not ordered_results:
            empty_f32 = self._torch.empty((0,), device=self._device, dtype=self._torch.float32)
            empty_bool = self._torch.empty((0,), device=self._device, dtype=self._torch.bool)
            empty_i64 = self._torch.empty((0,), device=self._device, dtype=self._torch.int64)
            return empty_f32, empty_bool, empty_bool, empty_i64, empty_i64

        reward_tensor = self._torch.tensor(
            [result.reward for result in ordered_results],
            device=self._device,
            dtype=self._torch.float32,
        )
        done_tensor = self._torch.tensor(
            [result.done for result in ordered_results],
            device=self._device,
            dtype=self._torch.bool,
        )
        truncated_tensor = self._torch.tensor(
            [result.truncated for result in ordered_results],
            device=self._device,
            dtype=self._torch.bool,
        )
        episode_id_tensor = self._torch.tensor(
            [result.episode_id for result in ordered_results],
            device=self._device,
            dtype=self._torch.int64,
        )
        env_id_tensor = self._torch.tensor(
            [result.env_id for result in ordered_results],
            device=self._device,
            dtype=self._torch.int64,
        )
        return reward_tensor, done_tensor, truncated_tensor, episode_id_tensor, env_id_tensor

    def close(self) -> None:
        self._dag_tensor = None

    @property
    def pose(self) -> RobotPose:
        return self.actor_pose(0)

    @property
    def episode_id(self) -> int:
        return int(self._episode_ids[0])

    def actor_pose(self, actor_id: int) -> RobotPose:
        pos = self._actor_positions[actor_id].detach().cpu().numpy()
        return RobotPose(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            roll=0.0,
            pitch=0.0,
            yaw=float(self._actor_yaws[actor_id].detach().cpu().item()),
            timestamp=time.time(),
        )

    def actor_episode_id(self, actor_id: int) -> int:
        return int(self._episode_ids[actor_id].detach().cpu().item())

    def perf_snapshot(self) -> SdfDagPerfSnapshot:
        """Return rolling throughput metrics for coarse telemetry and benchmarks."""
        if (
            self._perf_total_batches == 0
            or self._perf_total_actor_steps == 0
            or self._perf_total_batch_seconds <= 0.0
        ):
            return SdfDagPerfSnapshot(
                total_batches=0,
                total_actor_steps=0,
                last_actor_count=self._perf_last_actor_count,
                last_batch_step_ms=0.0,
                ema_batch_step_ms=0.0,
                avg_batch_step_ms=0.0,
                avg_actor_step_ms=0.0,
                sps=0.0,
            )

        avg_batch_seconds = self._perf_total_batch_seconds / self._perf_total_batches
        avg_actor_seconds = self._perf_total_batch_seconds / self._perf_total_actor_steps
        sps = self._perf_total_actor_steps / self._perf_total_batch_seconds
        return SdfDagPerfSnapshot(
            total_batches=self._perf_total_batches,
            total_actor_steps=self._perf_total_actor_steps,
            last_actor_count=self._perf_last_actor_count,
            last_batch_step_ms=self._perf_last_batch_seconds * 1_000.0,
            ema_batch_step_ms=self._perf_ema_batch_seconds * 1_000.0,
            avg_batch_step_ms=avg_batch_seconds * 1_000.0,
            avg_actor_step_ms=avg_actor_seconds * 1_000.0,
            sps=sps,
        )

    @staticmethod
    def _import_required_module(name: str) -> Any:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            msg = f"Required dependency '{name}' is not installed for sdfdag backend"
            raise RuntimeError(msg) from exc

    def _record_perf_sample(self, *, batch_seconds: float, actor_count: int) -> None:
        self._perf_total_batches += 1
        self._perf_total_actor_steps += actor_count
        self._perf_total_batch_seconds += batch_seconds
        self._perf_last_batch_seconds = batch_seconds
        self._perf_last_actor_count = actor_count
        if self._perf_total_batches == 1:
            self._perf_ema_batch_seconds = batch_seconds
            return
        self._perf_ema_batch_seconds = (_PERF_EMA_ALPHA * batch_seconds) + (
            (1.0 - _PERF_EMA_ALPHA) * self._perf_ema_batch_seconds
        )

    def _require_asset(self) -> GmDagAsset:
        asset = self._asset
        if asset is None:
            msg = "SdfDagBackend asset is not loaded"
            raise RuntimeError(msg)
        return asset

    def _require_dag_tensor(self) -> Any:
        dag_tensor = self._dag_tensor
        if dag_tensor is None:
            msg = "SdfDagBackend DAG tensor is not loaded"
            raise RuntimeError(msg)
        return dag_tensor

    @property
    def current_scene_name(self) -> str:
        """Return the human-readable name of the currently loaded scene."""
        return self._current_scene_name

    def _load_scene(self, scene_path: str) -> list[tuple[float, float, float]]:
        self._current_scene_name = Path(scene_path).stem
        asset = load_gmdag_asset(Path(scene_path), validate_layout=False)
        dag_tensor = self._torch.from_numpy(asset.nodes.view(np.int64)).to(
            device=self._device,
            dtype=self._torch.int64,
        )

        self._asset = asset
        self._dag_tensor = dag_tensor
        self._bbox_min = list(asset.bbox_min)
        self._bbox_max = list(asset.bbox_max)

        # Allocate visitation grid based on scene bounds
        grid_min = self._torch.tensor(
            self._bbox_min, device=self._device, dtype=self._torch.float32
        )
        grid_max = self._torch.tensor(
            self._bbox_max, device=self._device, dtype=self._torch.float32
        )
        extent = grid_max - grid_min

        # Scene-adaptive parameter scaling based on characteristic length
        char_len = (extent[0] * extent[1] * extent[2]).pow(1.0 / 3.0).item()
        adapted_grid_res = max(1.0, min(4.0, char_len / 10.0))
        adapted_grid_res_y = max(1.0, min(4.0, char_len / 10.0))
        adapted_structure_band_max = min(char_len * 0.6, 50.0)
        self._grid_res = adapted_grid_res
        self._grid_res_y = adapted_grid_res_y
        self._structure_band_max_distance = max(
            self._structure_band_min_distance + 1.0,
            adapted_structure_band_max,
        )

        grid_w = int(self._torch.ceil(extent[0] / self._grid_res).item()) + 1
        grid_h = int(self._torch.ceil(extent[2] / self._grid_res).item()) + 1
        grid_y = int(self._torch.ceil(extent[1] / self._grid_res_y).item()) + 1

        self._grid_min.copy_(grid_min)
        self._scene_bbox_min_t.copy_(grid_min)
        self._scene_bbox_max_t.copy_(grid_max)
        self._visit_grid = self._torch.zeros(
            (self._n_actors, grid_w, grid_y, grid_h),
            device=self._device,
            dtype=self._torch.int16,
        )

        _LOG.info(
            "Loaded gmdag asset %s (resolution=%d, nodes=%d, visit_grid=%dx%dx%d, "
            "char_len=%.1f, grid_res=%.2f, struct_band_max=%.1f)",
            asset.path,
            asset.resolution,
            asset.nodes.shape[0],
            grid_w,
            grid_y,
            grid_h,
            char_len,
            self._grid_res,
            self._structure_band_max_distance,
        )
        spawn_poses = self._find_spawn_poses(self._n_actors)
        for i, (x_pos, y_pos, z_pos, yaw) in enumerate(spawn_poses):
            self._spawn_positions[i].copy_(
                self._torch.tensor((x_pos, y_pos, z_pos), device=self._device)
            )
            self._spawn_yaws[i] = float(yaw)
        return [(x_pos, y_pos, z_pos) for x_pos, y_pos, z_pos, _yaw in spawn_poses]

    def _compute_reward_batch(
        self,
        *,
        actor_indices: Any,
        previous_positions: Any,  # (B, 3)
        current_positions: Any,  # (B, 3)
        collisions: Any,  # (B,) bool
        previous_clearances: Any,  # (B,)
        current_clearances: Any,  # (B,)
        starvation_ratios: Any,  # (B,)
        proximity_ratios: Any,  # (B,)
        current_structure_band_ratios: Any,  # (B,)
        current_forward_structure_ratios: Any,  # (B,)
    ) -> tuple[Any, Any]:
        """Vectorized reward engine implementation (Phase 2)."""
        grid_coords_xz = (
            ((current_positions[:, [0, 2]] - self._grid_min[[0, 2]]) / self._grid_res)
            .floor()
            .long()
        )
        grid_coords_y = (
            ((current_positions[:, 1] - self._grid_min[1]) / self._grid_res_y)
            .floor()
            .long()
        )
        grid_coords_xz[:, 0] = grid_coords_xz[:, 0].clamp(0, self._visit_grid.shape[1] - 1)
        grid_coords_y = grid_coords_y.clamp(0, self._visit_grid.shape[2] - 1)
        grid_coords_xz[:, 1] = grid_coords_xz[:, 1].clamp(0, self._visit_grid.shape[3] - 1)
        visit_counts = self._visit_grid[
            actor_indices, grid_coords_xz[:, 0], grid_coords_y, grid_coords_xz[:, 1]
        ].float()
        exploration_rewards = _EXPLORATION_REWARD / (1.0 + visit_counts)
        self._visit_grid[
            actor_indices, grid_coords_xz[:, 0], grid_coords_y, grid_coords_xz[:, 1]
        ] += 1
        # Heading novelty: reward visiting new heading sectors
        yaws = self._actor_yaws.index_select(0, actor_indices)
        heading_sectors = ((yaws % (2.0 * math.pi)) / (2.0 * math.pi) * _HEADING_SECTORS).long()
        heading_sectors = heading_sectors.clamp(0, _HEADING_SECTORS - 1)
        was_heading_visited = self._heading_visited[actor_indices, heading_sectors]
        heading_novelty = (~was_heading_visited).float() * _HEADING_NOVELTY_SCALE
        exploration_rewards = exploration_rewards + heading_novelty
        self._heading_visited[actor_indices, heading_sectors] = True
        # Frontier bonus: reward being near unvisited territory.
        # Check 6 face-adjacent neighbors in the 3D grid.
        gx = grid_coords_xz[:, 0]
        gy = grid_coords_y
        gz = grid_coords_xz[:, 1]
        g_shape = self._visit_grid.shape  # (actors, W, Y, H)
        neighbor_unvisited = self._torch.zeros(
            actor_indices.shape[0], device=self._device, dtype=self._torch.float32
        )
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx = (gx + dx).clamp(0, g_shape[1] - 1)
            ny = (gy + dy).clamp(0, g_shape[2] - 1)
            nz = (gz + dz).clamp(0, g_shape[3] - 1)
            neighbor_unvisited += (self._visit_grid[actor_indices, nx, ny, nz] == 0).float()
        # Normalize: 0-6 unvisited neighbors → 0-1 range
        frontier_bonus = (neighbor_unvisited / 6.0) * _FRONTIER_BONUS_SCALE
        exploration_rewards = exploration_rewards + frontier_bonus
        # Clearance-gate exploration: exploring into tight spaces yields less reward.
        clearance_gate = current_clearances.clamp(min=0.0, max=1.0)
        exploration_rewards = exploration_rewards * clearance_gate
        prev_struct = self._prev_structure_band_ratios[actor_indices]
        prev_fwd = self._prev_forward_structure_ratios[actor_indices]
        impl = getattr(self, "_reward_components_compiled", _reward_components_tensor)
        return impl(
            exploration_rewards,
            previous_positions,
            current_positions,
            collisions,
            previous_clearances,
            current_clearances,
            starvation_ratios,
            proximity_ratios,
            current_structure_band_ratios,
            current_forward_structure_ratios,
            prev_struct,
            prev_fwd,
            progress_reward_scale=_PROGRESS_REWARD_SCALE,
            collision_penalty=_COLLISION_PENALTY,
            obstacle_clearance_window=self._obstacle_clearance_window,
            obstacle_clearance_reward_scale=self._obstacle_clearance_reward_scale,
            starvation_ratio_threshold=self._starvation_ratio_threshold,
            starvation_penalty_scale=self._starvation_penalty_scale,
            proximity_penalty_scale=self._proximity_penalty_scale,
            structure_band_reward_scale=self._structure_band_reward_scale,
            forward_structure_reward_scale=self._forward_structure_reward_scale,
            inspection_activation_threshold=self._inspection_activation_threshold,
            inspection_reward_scale=self._inspection_reward_scale,
            starvation_dead_zone=_STARVATION_DEAD_ZONE,
        )

    def _step_kinematics_indexed(
        self,
        *,
        actor_indices: Any,
        actions_linear: Any,
        actions_angular: Any,
    ) -> None:
        """Tensor-indexed kinematic integration for the canonical action-tensor path."""
        prev_depths = self._prev_depth_tensors.index_select(0, actor_indices)
        prev_linear = self._prev_linear_vels.index_select(0, actor_indices)
        prev_angular = self._prev_angular_vels.index_select(0, actor_indices)
        positions = self._actor_positions.index_select(0, actor_indices)
        yaws = self._actor_yaws.index_select(0, actor_indices)
        impl = getattr(self, "_step_kinematics_compiled", _step_kinematics_tensor)
        updated_positions, updated_yaws, smooth_linear, smooth_angular = impl(
            prev_depths,
            actions_linear,
            actions_angular,
            prev_linear,
            prev_angular,
            positions,
            yaws,
            max_distance=self._max_distance,
            speed_fwd=self._speed_fwd,
            speed_vert=self._speed_vert,
            speed_lat=self._speed_lat,
            speed_yaw=self._speed_yaw,
            smoothing=self._smoothing,
            dt=self._dt,
            speed_limiter_distance=self._speed_limiter_distance,
        )
        self._prev_linear_vels[actor_indices] = smooth_linear
        self._prev_angular_vels[actor_indices] = smooth_angular
        # Clamp positions to scene bounding box to prevent actors escaping
        # through thin walls into void space.
        updated_positions = updated_positions.clamp(
            min=self._scene_bbox_min_t, max=self._scene_bbox_max_t,
        )
        self._actor_positions[actor_indices] = updated_positions
        self._actor_yaws[actor_indices] = updated_yaws

    def _step_kinematics_batch(
        self,
        *,
        actor_ids: tuple[int, ...],
        actions_linear: Any,  # (B, 3)
        actions_angular: Any,  # (B, 3)
    ) -> None:
        """Fully vectorized kinematic integration (Phase 3)."""
        actor_indices = self._torch.as_tensor(
            actor_ids, device=self._device, dtype=self._torch.long
        )
        prev_depths = self._prev_depth_tensors.index_select(0, actor_indices)
        prev_linear = self._prev_linear_vels.index_select(0, actor_indices)
        prev_angular = self._prev_angular_vels.index_select(0, actor_indices)
        positions = self._actor_positions.index_select(0, actor_indices)
        yaws = self._actor_yaws.index_select(0, actor_indices)
        impl = getattr(self, "_step_kinematics_compiled", _step_kinematics_tensor)
        updated_positions, updated_yaws, smooth_linear, smooth_angular = impl(
            prev_depths,
            actions_linear,
            actions_angular,
            prev_linear,
            prev_angular,
            positions,
            yaws,
            max_distance=self._max_distance,
            speed_fwd=self._speed_fwd,
            speed_vert=self._speed_vert,
            speed_lat=self._speed_lat,
            speed_yaw=self._speed_yaw,
            smoothing=self._smoothing,
            dt=self._dt,
            speed_limiter_distance=self._speed_limiter_distance,
        )
        self._prev_linear_vels[actor_indices] = smooth_linear
        self._prev_angular_vels[actor_indices] = smooth_angular
        # Clamp positions to scene bounding box to prevent actors escaping
        # through thin walls into void space.
        updated_positions = updated_positions.clamp(
            min=self._scene_bbox_min_t, max=self._scene_bbox_max_t,
        )
        self._actor_positions[actor_indices] = updated_positions
        self._actor_yaws[actor_indices] = updated_yaws

    def _build_ray_directions(self) -> Any:
        return self._torch.from_numpy(build_spherical_ray_directions(self._az_bins, self._el_bins))

    def _scratch_slot_views(
        self,
        *,
        scratch_slot: int,
        actor_count: int,
    ) -> tuple[Any, Any, Any, Any]:
        scratch_slot_count = getattr(self, "_scratch_slot_count", _SCRATCH_SLOT_COUNT)
        if scratch_slot < 0 or scratch_slot >= scratch_slot_count:
            msg = f"scratch_slot must stay within [0, {scratch_slot_count - 1}]"
            raise ValueError(msg)
        if self._ray_origins.ndim == 3:
            return (
                self._ray_origins[:actor_count],
                self._ray_dirs_world[:actor_count],
                self._out_distances[:actor_count],
                self._out_semantics[:actor_count],
            )
        return (
            self._ray_origins[scratch_slot, :actor_count],
            self._ray_dirs_world[scratch_slot, :actor_count],
            self._out_distances[scratch_slot, :actor_count],
            self._out_semantics[scratch_slot, :actor_count],
        )

    def _validate_cast_rays_inputs(
        self,
        dag_tensor: Any,
        origins: Any,
        dirs: Any,
        out_distances: Any,
        out_semantics: Any,
    ) -> None:
        expected_device = self._device.type
        tensors = (
            ("dag_tensor", dag_tensor, self._torch.int64),
            ("origins", origins, self._torch.float32),
            ("dirs", dirs, self._torch.float32),
            ("out_distances", out_distances, self._torch.float32),
            ("out_semantics", out_semantics, self._torch.int32),
        )
        for name, tensor, dtype in tensors:
            if tensor.device.type != expected_device:
                msg = f"{name} must be on {expected_device}; got {tensor.device.type}"
                raise RuntimeError(msg)
            if tensor.dtype != dtype:
                msg = f"{name} must have dtype {dtype}; got {tensor.dtype}"
                raise RuntimeError(msg)
            if not tensor.is_contiguous():
                msg = f"{name} must be contiguous"
                raise RuntimeError(msg)

        if origins.ndim != 3 or tuple(int(dim) for dim in origins.shape[2:]) != (3,):
            msg = f"origins must have shape [batch, rays, 3]; got {tuple(int(dim) for dim in origins.shape)}"
            raise RuntimeError(msg)
        if dirs.ndim != 3 or tuple(int(dim) for dim in dirs.shape[2:]) != (3,):
            msg = f"dirs must have shape [batch, rays, 3]; got {tuple(int(dim) for dim in dirs.shape)}"
            raise RuntimeError(msg)
        expected_ray_shape = (int(origins.shape[0]), int(origins.shape[1]), 3)
        if tuple(int(dim) for dim in dirs.shape) != expected_ray_shape:
            msg = f"dirs must match origins shape {expected_ray_shape}; got {tuple(int(dim) for dim in dirs.shape)}"
            raise RuntimeError(msg)
        expected_output_shape = (int(origins.shape[0]), int(origins.shape[1]))
        if tuple(int(dim) for dim in out_distances.shape) != expected_output_shape:
            msg = (
                f"out_distances must have shape {expected_output_shape}; got "
                f"{tuple(int(dim) for dim in out_distances.shape)}"
            )
            raise RuntimeError(msg)
        if tuple(int(dim) for dim in out_semantics.shape) != expected_output_shape:
            msg = (
                f"out_semantics must have shape {expected_output_shape}; got "
                f"{tuple(int(dim) for dim in out_semantics.shape)}"
            )
            raise RuntimeError(msg)

    def _cast_actor_batch(
        self,
        actor_ids: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        depth_t, semantic_t, valid_t, cm_t = self._cast_actor_batch_tensors(actor_ids)
        depth_2d = depth_t.detach().cpu().numpy().astype(np.float32, copy=False)
        semantic_2d = semantic_t.detach().cpu().numpy().astype(np.int32, copy=False)
        valid_2d = valid_t.detach().cpu().numpy().astype(np.bool_, copy=False)
        metric_2d = cm_t.detach().cpu().numpy().astype(np.float32, copy=False)
        return depth_2d, semantic_2d, valid_2d, metric_2d

    def _cast_actor_batch_tensors(
        self,
        actor_ids: tuple[int, ...],
        *,
        scratch_slot: int = 0,
    ) -> tuple[Any, Any, Any, Any]:
        actor_count = len(actor_ids)
        yaws = self._actor_yaws[list(actor_ids)]
        positions = self._actor_positions[list(actor_ids)]

        cos_yaw = self._torch.cos(yaws).unsqueeze(1)
        sin_yaw = self._torch.sin(yaws).unsqueeze(1)
        base_dirs = self._ray_dirs_local.unsqueeze(0).expand(actor_count, -1, -1)
        origins, dirs_world, out_distances, out_semantics = self._scratch_slot_views(
            scratch_slot=scratch_slot,
            actor_count=actor_count,
        )
        dirs_world[..., 0] = base_dirs[..., 0] * cos_yaw + base_dirs[..., 2] * sin_yaw
        dirs_world[..., 1] = base_dirs[..., 1]
        dirs_world[..., 2] = -base_dirs[..., 0] * sin_yaw + base_dirs[..., 2] * cos_yaw

        origins.copy_(positions.unsqueeze(1).expand(-1, self._n_rays, -1))

        dag_tensor = self._require_dag_tensor()
        self._validate_cast_rays_inputs(
            dag_tensor, origins, dirs_world, out_distances, out_semantics
        )
        self._torch_sdf.cast_rays(
            dag_tensor,
            origins,
            dirs_world,
            out_distances,
            out_semantics,
            self._config.sdf_max_steps,
            self._max_distance,
            self._bbox_min,
            self._bbox_max,
            self._require_asset().resolution,
            skip_direction_validation=True,
        )

        depth_2d, semantic_2d, valid_2d, _min_d, _sr, _pr, _sbr, _fsr, clamped_metric = (
            self._postprocess_cast_outputs(out_distances, out_semantics)
        )
        return depth_2d, semantic_2d, valid_2d, clamped_metric

    def _cast_actor_batch_indexed_tensors(
        self,
        actor_indices: Any,
        *,
        scratch_slot: int = 0,
    ) -> tuple[Any, Any, Any, Any]:
        actor_count = int(actor_indices.shape[0])
        yaws = self._actor_yaws.index_select(0, actor_indices)
        positions = self._actor_positions.index_select(0, actor_indices)

        cos_yaw = self._torch.cos(yaws).unsqueeze(1)
        sin_yaw = self._torch.sin(yaws).unsqueeze(1)
        base_dirs = self._ray_dirs_local.unsqueeze(0).expand(actor_count, -1, -1)
        origins, dirs_world, out_distances, out_semantics = self._scratch_slot_views(
            scratch_slot=scratch_slot,
            actor_count=actor_count,
        )
        dirs_world[..., 0] = base_dirs[..., 0] * cos_yaw + base_dirs[..., 2] * sin_yaw
        dirs_world[..., 1] = base_dirs[..., 1]
        dirs_world[..., 2] = -base_dirs[..., 0] * sin_yaw + base_dirs[..., 2] * cos_yaw

        origins.copy_(positions.unsqueeze(1).expand(-1, self._n_rays, -1))

        dag_tensor = self._require_dag_tensor()
        self._validate_cast_rays_inputs(
            dag_tensor, origins, dirs_world, out_distances, out_semantics
        )
        self._torch_sdf.cast_rays(
            dag_tensor,
            origins,
            dirs_world,
            out_distances,
            out_semantics,
            self._config.sdf_max_steps,
            self._max_distance,
            self._bbox_min,
            self._bbox_max,
            self._require_asset().resolution,
            skip_direction_validation=True,
        )

        depth_2d, semantic_2d, valid_2d, _min_d, _sr, _pr, _sbr, _fsr, clamped_metric = (
            self._postprocess_cast_outputs(out_distances, out_semantics)
        )
        return depth_2d, semantic_2d, valid_2d, clamped_metric

    def _coerce_batch_actor_indices(self, *, actor_count: int, actor_indices: Any | None) -> Any:
        if actor_indices is None:
            return self._torch.arange(actor_count, device=self._device, dtype=self._torch.int64)
        indices = self._torch.as_tensor(
            actor_indices, device=self._device, dtype=self._torch.int64
        )
        if indices.ndim != 1:
            msg = "actor_indices must be a 1D sequence"
            raise ValueError(msg)
        if int(indices.shape[0]) != actor_count:
            msg = f"actor_indices length {int(indices.shape[0])} does not match actor_count {actor_count}"
            raise ValueError(msg)
        if actor_count == 0:
            return indices
        if bool(((indices < 0) | (indices >= self._n_actors)).any().item()):
            msg = f"actor_indices must stay within [0, {self._n_actors - 1}]"
            raise ValueError(msg)
        if int(self._torch.unique(indices).shape[0]) != actor_count:
            msg = "actor_indices must be unique"
            raise ValueError(msg)
        return indices.contiguous()

    def _coerce_action_tensor_rows(self, action_tensor: Any) -> Any:
        """Return a contiguous float32 command matrix on the active device with shape `(actors, 4)`."""
        if hasattr(action_tensor, "detach"):
            rows = action_tensor.detach()
            if hasattr(rows, "to"):
                rows = rows.to(device=self._device, dtype=self._torch.float32)
        else:
            rows = self._torch.as_tensor(
                action_tensor, device=self._device, dtype=self._torch.float32
            )
        if rows.ndim != 2 or rows.shape[1] < 4:
            msg = "action_tensor must have shape (actors, 4)"
            raise ValueError(msg)
        rows = rows[:, :4]
        if not rows.is_contiguous():
            rows = rows.contiguous()
        return rows

    def _consume_actor_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        current_clearance: Any,
        depth_2d: Any,
        semantic_2d: Any,
        valid_2d: Any,
        materialize_depth_cpu: bool,
    ) -> tuple[Any, np.ndarray | None, Any]:
        """Update actor observation state and build the canonical trainer tensor."""
        del step_id
        previous_depth = self._prev_depth_tensors[actor_id]
        # Check if previous_depth is zero (uninitialized)
        if float(previous_depth.abs().sum()) < 1e-6:
            delta_2d = self._torch.zeros_like(depth_2d)
        else:
            delta_2d = depth_2d - previous_depth

        self._prev_depth_tensors[actor_id].copy_(depth_2d)
        self._prev_min_distances[actor_id] = current_clearance
        depth_cpu: np.ndarray | None = None
        if materialize_depth_cpu:
            depth_cpu = depth_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            # We don't need a host mirror for every step anymore if not requested
            pass

        observation_tensor = self._torch.stack(
            (
                depth_2d,
                semantic_2d.to(dtype=self._torch.float32),
                valid_2d.to(dtype=self._torch.float32),
            ),
            dim=0,
        )
        return observation_tensor, depth_cpu, delta_2d

    def _materialize_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        depth_2d: np.ndarray,
        delta_2d: np.ndarray,
        semantic_2d: np.ndarray,
        valid_2d: np.ndarray,
    ) -> DistanceMatrix:
        return materialize_distance_matrix(
            episode_id=int(self._episode_ids[actor_id]),
            env_id=actor_id,
            depth=depth_2d[np.newaxis, ...],
            delta_depth=delta_2d[np.newaxis, ...],
            semantic=semantic_2d[np.newaxis, ...],
            valid_mask=valid_2d[np.newaxis, ...],
            overhead=np.zeros((256, 256, 3), dtype=np.float32),
            robot_pose=self.actor_pose(actor_id),
            step_id=step_id,
        )

    def _build_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        depth_2d: np.ndarray,
        semantic_2d: np.ndarray,
        valid_2d: np.ndarray,
        min_distance_metric: float,
    ) -> DistanceMatrix:
        prev_depth_cpu = self._prev_depth_tensors[actor_id].detach().cpu().numpy()
        if float(np.abs(prev_depth_cpu).sum()) < 1e-6:
            delta = np.zeros_like(depth_2d)
        else:
            delta = depth_2d - prev_depth_cpu

        self._prev_depth_tensors[actor_id].copy_(
            self._torch.from_numpy(depth_2d).to(device=self._device, dtype=self._torch.float32)
        )
        self._prev_min_distances[actor_id] = min_distance_metric

        return self._materialize_observation(
            actor_id=actor_id,
            step_id=step_id,
            depth_2d=depth_2d,
            delta_2d=delta,
            semantic_2d=semantic_2d,
            valid_2d=valid_2d,
        )

    def _compute_reward(
        self,
        *,
        actor_id: int,
        previous_pose: RobotPose,
        current_pose: RobotPose,
        collision: bool,
        previous_clearance: float | None,
        current_clearance: float,
        starvation_ratio: float,
        proximity_ratio: float,
        current_structure_band_ratio: float,
        current_forward_structure_ratio: float,
    ) -> tuple[float, np.ndarray]:
        # Single actor implementation for compatibility (Phase 1/2)
        # 3D grid mapping logic
        curr_pos_t = self._torch.tensor(
            [current_pose.x, current_pose.y, current_pose.z],
            device=self._device,
            dtype=self._torch.float32,
        )
        grid_coord_x = int(
            ((curr_pos_t[0] - self._grid_min[0]) / self._grid_res).floor().clamp(
                0, self._visit_grid.shape[1] - 1
            ).item()
        )
        grid_coord_y = int(
            ((curr_pos_t[1] - self._grid_min[1]) / self._grid_res_y).floor().clamp(
                0, self._visit_grid.shape[2] - 1
            ).item()
        )
        grid_coord_z = int(
            ((curr_pos_t[2] - self._grid_min[2]) / self._grid_res).floor().clamp(
                0, self._visit_grid.shape[3] - 1
            ).item()
        )

        visit_count = int(self._visit_grid[actor_id, grid_coord_x, grid_coord_y, grid_coord_z].item())
        exploration_reward = _EXPLORATION_REWARD / (1.0 + visit_count)
        self._visit_grid[actor_id, grid_coord_x, grid_coord_y, grid_coord_z] += 1

        # Heading novelty bonus
        yaw = float(self._actor_yaws[actor_id])
        heading_sector = int((yaw % (2.0 * math.pi)) / (2.0 * math.pi) * _HEADING_SECTORS)
        heading_sector = max(0, min(_HEADING_SECTORS - 1, heading_sector))
        if not bool(self._heading_visited[actor_id, heading_sector].item()):
            exploration_reward += _HEADING_NOVELTY_SCALE
        self._heading_visited[actor_id, heading_sector] = True

        # Frontier bonus: reward adjacency to unvisited territory
        g_shape = self._visit_grid.shape
        unvisited_count = 0
        for ddx, ddy, ddz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx = max(0, min(g_shape[1] - 1, grid_coord_x + ddx))
            ny = max(0, min(g_shape[2] - 1, grid_coord_y + ddy))
            nz = max(0, min(g_shape[3] - 1, grid_coord_z + ddz))
            if int(self._visit_grid[actor_id, nx, ny, nz].item()) == 0:
                unvisited_count += 1
        exploration_reward += (unvisited_count / 6.0) * _FRONTIER_BONUS_SCALE

        dx = current_pose.x - previous_pose.x
        dy = current_pose.y - previous_pose.y
        dz = current_pose.z - previous_pose.z
        progress_reward = _PROGRESS_REWARD_SCALE * float(math.sqrt(dx * dx + dy * dy + dz * dz))
        clearance_reward = _obstacle_clearance_reward(
            previous_clearance,
            current_clearance,
            proximity_window=self._obstacle_clearance_window,
            reward_scale=self._obstacle_clearance_reward_scale,
        )
        starvation_penalty = _starvation_penalty(
            starvation_ratio,
            ratio_threshold=self._starvation_ratio_threshold,
            penalty_scale=self._starvation_penalty_scale,
        )
        proximity_penalty = _proximity_penalty(
            proximity_ratio,
            penalty_scale=self._proximity_penalty_scale,
        )
        structure_reward = _structure_band_reward(
            current_structure_band_ratio,
            reward_scale=self._structure_band_reward_scale,
        )
        forward_structure_reward = _forward_structure_reward(
            current_forward_structure_ratio,
            reward_scale=self._forward_structure_reward_scale,
        )
        inspection_reward = _inspection_reward(
            float(self._prev_structure_band_ratios[actor_id]),
            current_structure_band_ratio,
            previous_forward_structure_ratio=float(self._prev_forward_structure_ratios[actor_id]),
            current_forward_structure_ratio=current_forward_structure_ratio,
            reward_scale=self._inspection_reward_scale,
            activation_threshold=self._inspection_activation_threshold,
        )
        collision_penalty = _COLLISION_PENALTY if collision else 0.0
        reward_components = np.array(
            [
                exploration_reward,
                progress_reward,
                clearance_reward,
                starvation_penalty,
                proximity_penalty,
                structure_reward,
                forward_structure_reward,
                inspection_reward,
                collision_penalty,
            ],
            dtype=np.float32,
        )
        return float(reward_components.sum()), reward_components

    def _find_spawn_poses(self, count: int) -> list[tuple[float, float, float, float]]:
        asset = self._require_asset()
        bmin = np.array(asset.bbox_min, dtype=np.float32)
        bmax = np.array(asset.bbox_max, dtype=np.float32)
        extent = bmax - bmin
        x_values = np.linspace(
            bmin[0] + 0.2 * extent[0],
            bmax[0] - 0.2 * extent[0],
            _SPAWN_CANDIDATES_PER_AXIS,
            dtype=np.float32,
        )
        z_values = np.linspace(
            bmin[2] + 0.2 * extent[2],
            bmax[2] - 0.2 * extent[2],
            _SPAWN_CANDIDATES_PER_AXIS,
            dtype=np.float32,
        )
        y_values = np.linspace(
            bmin[1] + 0.2 * extent[1],
            bmax[1] - 0.2 * extent[1],
            _SPAWN_HEIGHT_SAMPLES,
            dtype=np.float32,
        )

        candidates = np.array(
            [[x, y, z] for x in x_values for y in y_values for z in z_values],
            dtype=np.float32,
        )
        probe_azimuth_bins = min(_SPAWN_PROBE_AZIMUTH_BINS, max(8, self._az_bins))
        probe_elevation_bins = min(_SPAWN_PROBE_ELEVATION_BINS, max(4, self._el_bins))
        probe_dirs = build_spherical_ray_directions(probe_azimuth_bins, probe_elevation_bins)
        probe_ray_count = int(probe_dirs.shape[0])
        origins = self._torch.from_numpy(
            np.repeat(candidates[:, None, :], probe_ray_count, axis=1)
        ).to(self._device)
        dirs = self._torch.from_numpy(
            np.repeat(probe_dirs[None, :, :], candidates.shape[0], axis=0)
        ).to(self._device)
        out_distances = self._torch.empty(
            (candidates.shape[0], probe_ray_count),
            device=self._device,
            dtype=self._torch.float32,
        )
        out_semantics = self._torch.empty(
            (candidates.shape[0], probe_ray_count),
            device=self._device,
            dtype=self._torch.int32,
        )
        dag_tensor = self._require_dag_tensor()
        self._validate_cast_rays_inputs(dag_tensor, origins, dirs, out_distances, out_semantics)
        self._torch_sdf.cast_rays(
            dag_tensor,
            origins,
            dirs,
            out_distances,
            out_semantics,
            max(32, min(self._config.sdf_max_steps, 128)),
            self._max_distance,
            self._bbox_min,
            self._bbox_max,
            asset.resolution,
        )
        metric_distances = out_distances.detach().cpu().numpy().astype(np.float32, copy=False)
        metric_distances = metric_distances.reshape(
            candidates.shape[0], probe_azimuth_bins, probe_elevation_bins
        )
        valid = np.isfinite(metric_distances)
        clamped_metric = np.clip(metric_distances, 0.0, self._max_distance)
        scored: list[tuple[float, float, tuple[float, float, float, float]]] = []
        for idx, candidate in enumerate(candidates):
            metric_2d = clamped_metric[idx]
            valid_2d = valid[idx]
            starvation_ratio, _prox, _struct, _fwd = _observation_profile(
                metric_2d,
                valid_2d,
                max_distance=self._max_distance,
                proximity_distance_threshold=self._proximity_distance_threshold,
                structure_band_min_distance=self._structure_band_min_distance,
                structure_band_max_distance=self._structure_band_max_distance,
            )
            score = _spawn_candidate_score(
                metric_2d,
                valid_2d,
                max_distance=self._max_distance,
                proximity_distance_threshold=self._proximity_distance_threshold,
                structure_band_min_distance=self._structure_band_min_distance,
                structure_band_max_distance=self._structure_band_max_distance,
            )
            best_yaw = _select_spawn_yaw_from_observation(
                metric_2d,
                valid_2d,
                structure_band_min_distance=self._structure_band_min_distance,
                structure_band_max_distance=self._structure_band_max_distance,
            )
            scored.append(
                (score, starvation_ratio, (float(candidate[0]), float(candidate[1]), float(candidate[2]), best_yaw))
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            center = 0.5 * (bmin + bmax)
            return [(float(center[0]), float(center[1]), float(center[2]), 0.0)]

        # Quality gate: reject candidates with excessive starvation (void)
        viable = [
            (sc, starv, sp) for sc, starv, sp in scored
            if starv < _SPAWN_MAX_STARVATION
        ]
        if len(viable) >= count:
            selected = [sp for _, _, sp in viable[:count]]
        elif viable:
            # Not enough viable candidates — fill remaining slots with the best
            selected = [sp for _, _, sp in viable]
            best_spawn = viable[0][2]
            while len(selected) < count:
                selected.append(best_spawn)
            _LOG.warning(
                "Only %d/%d spawn candidates pass quality gate (starvation < %.0f%%); "
                "duplicating best candidate",
                len(viable), count, _SPAWN_MAX_STARVATION * 100,
            )
        else:
            # All candidates are in void — use top N by score as fallback
            selected = [sp for _, _, sp in scored[:max(count, 1)]]
            best_starv = scored[0][1]
            _LOG.warning(
                "No spawn candidates pass quality gate (best starvation=%.1f%%); "
                "scene AABB may be much larger than actual geometry",
                best_starv * 100,
            )

        _LOG.info(
            "Selected %d spawn candidates (viable=%d/%d, best_score=%.2f, best_starvation=%.1f%%)",
            len(selected), len(viable), len(scored), scored[0][0], scored[0][1] * 100,
        )
        return selected
