"""Training sub-package for the Actor."""

from __future__ import annotations

__all__: list[str] = [
    "EvaluationMetrics",
    "EvaluationPoint",
    "OnlineSphericalTrainer",
    "OnlineTrainingMetrics",
    "TrainingLoop",
]

from navi_actor.training.loop import TrainingLoop
from navi_actor.training.online import (
    EvaluationMetrics,
    EvaluationPoint,
    OnlineSphericalTrainer,
    OnlineTrainingMetrics,
)
