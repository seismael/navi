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
    step_endpoint: str = "tcp://localhost:5560"  # Environment REP address

    # Observation shape (must match environment resolution)
    azimuth_bins: int = 128
    elevation_bins: int = 24

    # Cognitive architecture
    embedding_dim: int = 128
    learning_rate: float = 3e-4
    learning_rate_final: float = 3e-5

    # PPO hyper-parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.005
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    rollout_length: int = 512
    minibatch_size: int = 32
    bptt_len: int = 16

    # Multi-actor
    n_actors: int = 1

    # Steering scales (4-DOF normalised directional commands).
    # The policy outputs in [-1, 1] via Tanh; these scales define the
    # range seen by the Gaussian distribution.  1.0 = fully normalised.
    # Actual speed (m/s) is configured on the backend (drone_max_speed)
    # and further modulated by dynamic proximity scaling \u2014 the drone
    # slows near walls and accelerates in open space automatically.
    max_forward: float = 1.0    # normalised forward steering
    max_vertical: float = 1.0   # normalised vertical steering
    max_lateral: float = 1.0    # normalised lateral steering
    max_yaw: float = 1.0        # normalised yaw steering

    # RND curiosity
    rnd_learning_rate: float = 3e-5
    rnd_learning_rate_final: float = 3e-6

    # Episodic memory
    memory_capacity: int = 10_000
    memory_exclusion_window: int = 50

    # Reward shaping
    # Note: existential_tax is applied per *decision* step.  With
    # dt=0.02 each step covers 20 ms, so at 50 Hz the per-second
    # tax equals existential_tax x 50.
    collision_penalty: float = 0.0
    existential_tax: float = -0.01
    velocity_weight: float = 0.0  # disabled — speed is not a training signal
    intrinsic_coeff_init: float = 0.2
    intrinsic_coeff_final: float = 0.01
    intrinsic_anneal_steps: int = 500_000
    loop_penalty_coeff: float = 0.5
    loop_threshold: float = 0.85
