"""Inference sub-package for the Actor."""

from __future__ import annotations

__all__: list[str] = [
    "InferenceRunner",
    "InferenceMetrics",
]

from navi_actor.inference.runner import InferenceMetrics, InferenceRunner
