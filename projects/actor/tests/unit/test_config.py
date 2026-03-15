"""Configuration regression tests for deterministic CLI/runtime overrides."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

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
    assert config.temporal_core == "gru"
    assert config.emit_observation_stream is True
    assert config.dashboard_observation_hz == 10.0
    assert config.emit_training_telemetry is True
    assert config.emit_update_loss_telemetry is False
    assert config.emit_perf_telemetry is True
    assert config.enable_episodic_memory is True
    assert config.enable_reward_shaping is True


def test_actor_config_temporal_core_override_and_default() -> None:
    """Temporal-core selector should default to GRU and accept explicit overrides."""
    assert ActorConfig().temporal_core == "gru"
    assert ActorConfig(temporal_core="mambapy").temporal_core == "mambapy"


def test_actor_config_rejects_unsupported_temporal_core() -> None:
    """Unsupported temporal-core names should fail validation immediately."""
    with pytest.raises(ValidationError, match=r"mambapy|gru"):
        ActorConfig(temporal_core=cast("Any", "mamba-ssm"))
