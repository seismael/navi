"""Matrix viewer adapter for Ghost-Matrix dashboard operations."""

from __future__ import annotations

from navi_auditor.dashboard.app import run_dashboard

__all__: list[str] = ["MatrixViewer"]


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
    ) -> None:
        self._matrix_sub = matrix_sub
        self._actor_sub = actor_sub
        self._step_endpoint = step_endpoint or ""
        self._hz = hz
        self._linear_speed = linear_speed
        self._yaw_rate = yaw_rate

    def run(self) -> None:
        """Launch the PyQtGraph RL Dashboard."""
        run_dashboard(
            matrix_sub=self._matrix_sub,
            actor_sub=self._actor_sub,
            step_endpoint=self._step_endpoint,
            hz=self._hz,
            linear_speed=self._linear_speed,
            yaw_rate=self._yaw_rate,
        )
