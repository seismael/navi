"""Configuration for the Actor service."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["ActorConfig"]


@dataclass(frozen=True)
class ActorConfig:
    """Actor service configuration, loadable from TOML."""

    # ZMQ addresses
    sub_address: str = "tcp://localhost:5559"
    pub_address: str = "tcp://*:5557"
    mode: str = "async"  # "async" (PUB/SUB) or "step" (REQ/REP)
    step_endpoint: str = "tcp://localhost:5560"  # Section Manager REP address

    # Observation shape
    azimuth_bins: int = 64
    elevation_bins: int = 32

    # Cognitive architecture
    embedding_dim: int = 128
    learning_rate: float = 3e-4

    # PPO hyper-parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    rollout_length: int = 512
    minibatch_size: int = 64
    bptt_len: int = 32

    # Action scales (4-DOF)
    max_forward: float = 1.2
    max_vertical: float = 0.8
    max_lateral: float = 0.8
    max_yaw: float = 1.2

    # RND curiosity
    rnd_learning_rate: float = 1e-3

    # Episodic memory
    memory_capacity: int = 10_000
    memory_exclusion_window: int = 50

    # Reward shaping
    collision_penalty: float = -10.0
    existential_tax: float = -0.01
    velocity_weight: float = 0.1
    intrinsic_coeff_init: float = 1.0
    intrinsic_coeff_final: float = 0.01
    intrinsic_anneal_steps: int = 500_000
    loop_penalty_coeff: float = 0.5
    loop_threshold: float = 0.85
