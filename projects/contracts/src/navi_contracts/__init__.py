"""Navi Contracts — wire-format models and serialization for the Navi ecosystem."""

from __future__ import annotations

from navi_contracts.models import (
    Action,
    BatchStepRequest,
    BatchStepResult,
    DistanceMatrix,
    RobotPose,
    StepRequest,
    StepResult,
    TelemetryEvent,
)
from navi_contracts.serialization import deserialize, serialize
from navi_contracts.topics import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_STEP_REQUEST,
    TOPIC_STEP_RESULT,
    TOPIC_TELEMETRY_EVENT,
)
from navi_contracts.types import (
    DeltaDepthMatrix,
    DepthMatrix,
    EnvIdVector,
    MatrixShape,
    SemanticMatrix,
    TelemetryPayload,
    ValidMask,
    VelocityMatrix,
)

__all__: list[str] = [
    "TOPIC_ACTION",
    # Topics
    "TOPIC_DISTANCE_MATRIX",
    "TOPIC_STEP_REQUEST",
    "TOPIC_STEP_RESULT",
    "TOPIC_TELEMETRY_EVENT",
    "Action",
    "BatchStepRequest",
    "BatchStepResult",
    "DeltaDepthMatrix",
    "DepthMatrix",
    "DistanceMatrix",
    "EnvIdVector",
    # Type aliases
    "MatrixShape",
    # Models
    "RobotPose",
    "SemanticMatrix",
    "StepRequest",
    "StepResult",
    "TelemetryEvent",
    "TelemetryPayload",
    "ValidMask",
    "VelocityMatrix",
    "deserialize",
    # Serialization
    "serialize",
]
