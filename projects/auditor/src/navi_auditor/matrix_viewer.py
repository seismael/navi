"""Matrix viewer adapter for Ghost-Matrix dashboard operations."""

from __future__ import annotations

import logging

from navi_auditor.dashboard.app import run_dashboard

__all__: list[str] = ["MatrixViewer"]

_LOGGER = logging.getLogger(__name__)


class MatrixViewer:
    """Thin wrapper around the PyQtGraph dashboard for Gallery API surface."""

    def __init__(
        self,
        matrix_sub: str,
        actor_sub: str = "",
        step_endpoint: str | None = None,
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
        scene_path: str | None = None,
    ) -> None:
        self._matrix_sub = matrix_sub
        self._actor_sub = actor_sub
        self._step_endpoint = step_endpoint or ""
        self._hz = hz
        self._linear_speed = linear_speed
        self._yaw_rate = yaw_rate
        self._scene_path = scene_path

    def run(self) -> None:
        """Launch the PyQtGraph RL Dashboard."""
        _LOGGER.info("Starting Ghost-Matrix Dashboard (hz=%0.1f, matrix_sub=%s)", 
                     self._hz, self._matrix_sub)
        run_dashboard(
            matrix_sub=self._matrix_sub,
            actor_sub=self._actor_sub,
            step_endpoint=self._step_endpoint,
            hz=self._hz,
            linear_speed=self._linear_speed,
            yaw_rate=self._yaw_rate,
            scene_path=self._scene_path,
        )
