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
        matrix_sub: str = "",
        actor_sub: str = "",
        step_endpoint: str | None = None,
        actor_id: int = 0,
        enable_actor_selector: bool = True,
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
        max_distance_m: float = 30.0,
        scene_path: str | None = None,
    ) -> None:
        self._matrix_sub = matrix_sub
        self._actor_sub = actor_sub
        self._step_endpoint = step_endpoint or ""
        self._actor_id = actor_id
        self._enable_actor_selector = enable_actor_selector
        self._hz = hz
        self._linear_speed = linear_speed
        self._yaw_rate = yaw_rate
        self._max_distance_m = max_distance_m
        self._scene_path = scene_path

    def run(self) -> None:
        """Launch the PyQtGraph RL Dashboard."""
        _LOGGER.info(
            "Starting Ghost-Matrix Dashboard (hz=%0.1f, matrix_sub=%s, actor_sub=%s)",
            self._hz,
            self._matrix_sub or "<disabled>",
            self._actor_sub or "<disabled>",
        )
        run_dashboard(
            matrix_sub=self._matrix_sub,
            actor_sub=self._actor_sub,
            step_endpoint=self._step_endpoint,
            actor_id=self._actor_id,
            enable_actor_selector=self._enable_actor_selector,
            hz=self._hz,
            linear_speed=self._linear_speed,
            yaw_rate=self._yaw_rate,
            max_distance_m=self._max_distance_m,
            scene_path=self._scene_path,
        )
