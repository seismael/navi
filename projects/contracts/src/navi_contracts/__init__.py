"""Navi Contracts — wire-format models and serialization for the Navi ecosystem."""

from __future__ import annotations

import importlib
from typing import Any

from navi_contracts.logging import setup_logging
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
from navi_contracts.observability import (
    JsonlMetricsSink,
    RunContext,
    build_phase_metrics_payload,
    collect_process_resource_snapshot,
    get_or_create_run_context,
    get_run_id,
    write_process_manifest,
)
from navi_contracts.testing.oracle_house import (
    OracleObservation,
    canonical_house_bbox,
    house_metric_distances,
    house_observation,
    house_observation_after_forward_motion,
    house_observation_delta,
    write_square_house_obj,
)
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
    "build_phase_metrics_payload",
    "OracleObservation",
    "collect_process_resource_snapshot",
    "DeltaDepthMatrix",
    "DepthMatrix",
    "DistanceMatrix",
    "EnvIdVector",
    # Type aliases
    "MatrixShape",
    # Models
    "RobotPose",
    "RunContext",
    "SemanticMatrix",
    "StepRequest",
    "StepResult",
    "TelemetryEvent",
    "TelemetryPayload",
    "ValidMask",
    "VelocityMatrix",
    "JsonlMetricsSink",
    "canonical_house_bbox",
    "deserialize",
    "get_or_create_run_context",
    "get_run_id",
    "house_metric_distances",
    "house_observation",
    "house_observation_after_forward_motion",
    "house_observation_delta",
    "write_process_manifest",
    "write_square_house_obj",
    # Serialization (lazy — loaded on first access to avoid requiring msgpack at import time)
    "serialize",
    "setup_logging",
]

# Lazy imports for serialization (requires msgpack which may not be available
# in all consumer venvs, e.g. torch-sdf).
_LAZY_SERIALIZATION = {"serialize", "deserialize"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SERIALIZATION:
        mod = importlib.import_module("navi_contracts.serialization")
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
