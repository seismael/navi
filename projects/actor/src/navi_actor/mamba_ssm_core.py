"""Compatibility shim for the canonical fused Mamba-2 temporal core."""

from __future__ import annotations

import logging

from navi_actor.mamba_core import Mamba2TemporalCore as Mamba2SsmTemporalCore

__all__: list[str] = ["Mamba2SsmTemporalCore"]

_LOGGER = logging.getLogger(__name__)