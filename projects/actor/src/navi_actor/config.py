"""Configuration for the Actor service."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["ActorConfig"]


@dataclass(frozen=True)
class ActorConfig:
    """Actor service configuration, loadable from TOML."""

    sub_address: str = "tcp://localhost:5559"
    pub_address: str = "tcp://*:5557"
    mode: str = "async"  # "async" (PUB/SUB) or "step" (REQ/REP)
    step_endpoint: str = "tcp://localhost:5560"  # Section Manager REP address
    azimuth_bins: int = 64
    elevation_bins: int = 32
    embedding_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    learning_rate: float = 3e-4
