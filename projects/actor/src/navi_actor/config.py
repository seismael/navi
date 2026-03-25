"""Configuration for the Actor service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict

TemporalCoreName = Literal["gru", "mambapy", "mamba2"]
SUPPORTED_TEMPORAL_CORES: tuple[TemporalCoreName, ...] = ("gru", "mambapy", "mamba2")

__all__: list[str] = ["ActorConfig", "SUPPORTED_TEMPORAL_CORES", "TemporalCoreName"]


def find_root_env() -> Path:
    """Search upwards for the root .env file."""
    try:
        curr = Path(__file__).resolve().parent
        for _ in range(6):
            target = curr / ".env"
            if target.exists():
                return target
            curr = curr.parent
    except Exception:  # noqa: S110
        pass
    return Path(".env")  # fallback


class ActorConfig(BaseSettings):
    """Actor service configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Robust Fallback
    sub_address: str = Field(
        default="tcp://localhost:5559",
        validation_alias="NAVI_ENV_PUB_ADDRESS",
    )
    pub_address: str = Field(
        default="tcp://*:5557",
        validation_alias="NAVI_ACTOR_PUB_ADDRESS",
    )
    step_endpoint: str = Field(
        default="tcp://localhost:5560",
        validation_alias="NAVI_ENV_REP_ADDRESS",
    )

    mode: str = "async"
    temporal_core: TemporalCoreName = Field(
        default="mamba2",
        validation_alias="NAVI_ACTOR_TEMPORAL_CORE",
    )
    azimuth_bins: int = Field(default=256, validation_alias="NAVI_AZIMUTH_BINS")
    elevation_bins: int = Field(default=48, validation_alias="NAVI_ELEVATION_BINS")
    embedding_dim: int = 128
    learning_rate: float = 3e-4
    learning_rate_final: float = 3e-5
    lr_schedule_steps: int = 0
    normalize_returns: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    rollout_length: int = 256
    minibatch_size: int = 64
    bptt_len: int = 8
    n_actors: int = 1
    max_forward: float = 1.0
    max_vertical: float = 1.0
    max_lateral: float = 1.0
    max_yaw: float = 1.0
    rnd_learning_rate: float = 3e-5
    rnd_learning_rate_final: float = 3e-6
    memory_capacity: int = 100_000
    memory_exclusion_window: int = 50
    collision_penalty: float = 0.0
    existential_tax: float = -0.02
    velocity_weight: float = 0.0
    intrinsic_coeff_init: float = 1.0
    intrinsic_coeff_final: float = 0.01
    intrinsic_anneal_steps: int = 500_000
    loop_penalty_coeff: float = 2.0
    loop_threshold: float = 0.85

    # Telemetry fan-out controls (performance)
    emit_observation_stream: bool = True
    dashboard_observation_hz: float = 10.0
    emit_training_telemetry: bool = True
    emit_update_loss_telemetry: bool = False
    emit_perf_telemetry: bool = True
    emit_internal_stats: bool = Field(
        default=True,
        validation_alias="NAVI_ACTOR_EMIT_INTERNAL_STATS",
    )
    attach_resource_snapshots: bool = Field(
        default=True,
        validation_alias="NAVI_ACTOR_ATTACH_RESOURCE_SNAPSHOTS",
    )
    print_performance_summary: bool = Field(
        default=True,
        validation_alias="NAVI_ACTOR_PRINT_PERFORMANCE_SUMMARY",
    )
    profile_cuda_events: bool = False
    reward_shaping_torch_compile: bool = Field(
        default=True,
        validation_alias="NAVI_ACTOR_REWARD_SHAPING_TORCH_COMPILE",
    )

    # Diagnostic ablations on the canonical trainer surface
    enable_episodic_memory: bool = True
    enable_reward_shaping: bool = True

    # Rollout overlap: >1 groups pipelines forward/step across CUDA streams.
    # Default 1 is optimal for small GPUs (MX150, 3 SMs). Set to 2 on larger GPUs.
    rollout_overlap_groups: int = Field(
        default=1,
        validation_alias="NAVI_ACTOR_ROLLOUT_OVERLAP_GROUPS",
    )

    # Staggered per-actor PPO: each actor independently triggers PPO updates
    # when its own rollout buffer is full, distributing PPO blocking time across
    # smaller, more frequent updates instead of one large synchronized barrier.
    stagger_ppo: bool = Field(
        default=True,
        validation_alias="NAVI_ACTOR_STAGGER_PPO",
    )
