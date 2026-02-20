"""Training sub-package for the Actor."""

from __future__ import annotations

__all__: list[str] = [
    "EvaluationMetrics",
    "EvaluationPoint",
    "OnlineSphericalTrainer",
    "OnlineTrainingMetrics",
    "PpoTrainer",
    "PpoTrainingMetrics",
    "TrainingLoop",
]

from navi_actor.training.loop import TrainingLoop
from navi_actor.training.online import (
    EvaluationMetrics,
    EvaluationPoint,
    OnlineSphericalTrainer,
    OnlineTrainingMetrics,
)
from navi_actor.training.ppo_trainer import PpoTrainer, PpoTrainingMetrics
