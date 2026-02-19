"""ZMQ topic string constants for PUB/SUB communication."""

from __future__ import annotations

__all__: list[str] = [
    "TOPIC_ACTION",
    "TOPIC_DISTANCE_MATRIX",
    "TOPIC_STEP_REQUEST",
    "TOPIC_STEP_RESULT",
    "TOPIC_TELEMETRY_EVENT",
]

TOPIC_DISTANCE_MATRIX: str = "distance_matrix_v2"
"""Published by Simulation Layer, consumed by Brain and Gallery."""

TOPIC_ACTION: str = "action_v2"
"""Published by Brain, consumed by Simulation Layer and Gallery."""

TOPIC_STEP_REQUEST: str = "step_request_v2"
"""Sent by Brain to Simulation Layer via REQ/REP."""

TOPIC_STEP_RESULT: str = "step_result_v2"
"""Replied by Simulation Layer to Brain via REQ/REP."""

TOPIC_TELEMETRY_EVENT: str = "telemetry_event_v2"
"""Published by runtime services for asynchronous recording and replay."""
