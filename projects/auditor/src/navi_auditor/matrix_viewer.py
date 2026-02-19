"""Matrix viewer adapter for Ghost-Matrix dashboard operations."""

from __future__ import annotations

from navi_auditor.recorder import LiveDashboard

__all__: list[str] = ["MatrixViewer"]


class MatrixViewer:
    """Thin wrapper around LiveDashboard for explicit Gallery API surface."""

    def __init__(
        self,
        matrix_sub: str,
        step_endpoint: str | None = None,
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
    ) -> None:
        self._dashboard = LiveDashboard(
            matrix_sub=matrix_sub,
            step_endpoint=step_endpoint,
            tick_hz=hz,
            linear_speed=linear_speed,
            yaw_rate=yaw_rate,
        )

    def run(self) -> None:
        """Run interactive matrix viewer."""
        self._dashboard.run()
