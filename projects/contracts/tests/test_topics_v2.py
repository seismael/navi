"""Tests for Ghost-Matrix v2 topic constants."""

from __future__ import annotations

from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_STEP_REQUEST,
    TOPIC_STEP_RESULT,
    TOPIC_TELEMETRY_EVENT,
)


def test_topic_values_are_v2() -> None:
    assert TOPIC_DISTANCE_MATRIX == "distance_matrix_v2"
    assert TOPIC_ACTION == "action_v2"
    assert TOPIC_STEP_REQUEST == "step_request_v2"
    assert TOPIC_STEP_RESULT == "step_result_v2"
    assert TOPIC_TELEMETRY_EVENT == "telemetry_event_v2"
