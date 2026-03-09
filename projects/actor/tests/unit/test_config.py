"""Configuration regression tests for deterministic CLI/runtime overrides."""

from __future__ import annotations

from navi_actor.config import ActorConfig


def test_actor_config_allows_field_name_overrides_with_aliases() -> None:
    """Explicit constructor values must win over aliased environment defaults."""
    config = ActorConfig(
        sub_address="tcp://127.0.0.1:19000",
        pub_address="tcp://127.0.0.1:19557",
        step_endpoint="tcp://127.0.0.1:19002",
        azimuth_bins=128,
        elevation_bins=24,
    )

    assert config.sub_address == "tcp://127.0.0.1:19000"
    assert config.pub_address == "tcp://127.0.0.1:19557"
    assert config.step_endpoint == "tcp://127.0.0.1:19002"
    assert config.azimuth_bins == 128
    assert config.elevation_bins == 24
    assert config.emit_observation_stream is True
    assert config.emit_training_telemetry is True
    assert config.emit_perf_telemetry is True
    assert config.enable_episodic_memory is True
    assert config.enable_reward_shaping is True
