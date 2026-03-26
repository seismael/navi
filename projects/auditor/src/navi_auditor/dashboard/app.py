"""Ghost-Matrix RL Dashboard — actor 0 observer view.

The dashboard is intentionally visual-only: it renders one live actor 0
depth view with mode/status indication and a count of discovered actors.
"""

from __future__ import annotations

import time

import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from navi_auditor.dashboard.panels import (
    ImagePanel,
    StatusBar,
)
from navi_auditor.dashboard.renderers import (
    render_first_person,
)
from navi_auditor.dashboard.status_line import build_status_metrics_line
from navi_auditor.stream_engine import StreamEngine, StreamState

__all__: list[str] = ["GhostMatrixDashboard"]


class GhostMatrixDashboard(QtWidgets.QMainWindow):
    """High-performance real-time actor-0 visualiser.

    Parameters
    ----------
    matrix_sub
        Optional ZMQ SUB address for Environment PUB (distance_matrix_v2).
        Leave empty in passive actor-only training mode.
    actor_sub
        ZMQ SUB address for Actor/Trainer PUB (action_v2, telemetry).
    step_endpoint
        Optional ZMQ REQ address for manual teleop stepping.
    hz
        Target refresh rate in Hertz.
    linear_speed
        Maximum linear velocity for manual teleop.
    yaw_rate
        Maximum yaw rate for manual teleop.
    """

    def __init__(
        self,
        matrix_sub: str = "",
        actor_sub: str = "",
        step_endpoint: str = "",
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
        max_distance_m: float = 30.0,
        scene_path: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Ghost-Matrix RL Auditor")
        self.resize(1920, 1080)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        # Stream engine — always watching actor 0
        self._engine = StreamEngine(
            matrix_sub=matrix_sub,
            actor_sub=actor_sub,
            step_endpoint=step_endpoint,
            selected_actor_id=0,
        )

        # Teleop state
        self._linear_speed = linear_speed
        self._yaw_rate_max = yaw_rate
        self._manual_mode = False
        self._fwd = 0.0
        self._yaw = 0.0
        self._scene_path = scene_path
        self._max_distance_m = float(max_distance_m)

        # ── build UI ─────────────────────────────────────────────────
        self._status_bar = StatusBar()
        self._actor_panel = ImagePanel()
        self._build_layout()

        # ── tick timer ───────────────────────────────────────────────
        self._tick_ms = max(1, int(1000.0 / max(1.0, hz)))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._tick_ms)

    # ── layout construction ──────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the main window layout: single header row + full-bleed actor panel."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Single header row (mode + actor count + metrics)
        main_layout.addWidget(self._status_bar)

        # Actor panel — no margins, fills all remaining space
        main_layout.addWidget(self._actor_panel, stretch=1)

    def _shared_metrics_state(self) -> StreamState | None:
        """Return one actor state carrying shared coarse trainer metrics."""
        for actor_state in self._engine.actor_states.values():
            if (
                len(actor_state.perf_sps_history) > 0
                or len(actor_state.env_perf_sps_history) > 0
                or len(actor_state.ppo_reward_ema_history) > 0
                or len(actor_state.reward_history) > 0
            ):
                return actor_state
        return None

    # ── tick / render loop ───────────────────────────────────────────

    def _tick(self) -> None:
        """Called every timer interval — poll ZMQ, update all panels."""
        # Ingest capped ZMQ burst (Standard: UI Throughput)
        _msgs_processed = self._engine.poll(max_messages=100)

        # Update actor count display
        n_actors = self._engine.n_actors
        self._status_bar.set_actor_count(n_actors)

        state = self._engine.actor_states.get(0)
        if state is None:
            self._status_bar.set_mode("WAITING")
            self._status_bar.set_metrics_text(build_status_metrics_line(None))
            return

        # Determine mode (Standard: Mode Detection)
        # Check ALL actor states — training telemetry is only published for the
        # selected stream actor, so non-selected actors would never accumulate
        # reward history even though they are part of the same training session.
        has_training_data = any(
            len(s.reward_history) > 0 or len(s.ppo_reward_ema_history) > 0
            for s in self._engine.actor_states.values()
        )
        has_inference_data = state.latest_features is not None

        if self._manual_mode:
            mode = "MANUAL"
        elif has_training_data:
            mode = "TRAINING"
        elif has_inference_data:
            mode = "INFERENCE"
        else:
            mode = "OBSERVER"
        self._status_bar.set_mode(mode)
        self._status_bar.set_metrics_text(
            build_status_metrics_line(
                state,
                now=time.time(),
                fallback_state=self._shared_metrics_state(),
            ),
        )

        if state.latest_matrix is not None:
            self._update_actor_panel(self._actor_panel, state.latest_matrix)

        self._handle_teleop()

    def _update_actor_panel(self, panel: ImagePanel, dm: object) -> None:
        """Update one actor panel with the corrected forward perspective view."""
        from navi_contracts import DistanceMatrix

        assert isinstance(dm, DistanceMatrix)
        # Render at the panel's current pixel size for full-bleed display
        size = panel.size()
        render_w = max(320, size.width())
        render_h = max(200, size.height())
        actor_img, _center_m = render_first_person(
            dm.depth[0],
            dm.semantic[0],
            dm.valid_mask[0],
            render_w,
            render_h,
            pitch=float(dm.robot_pose.pitch),
        )
        panel.set_image(actor_img)

    # ── teleop / keyboard control ────────────────────────────────────

    def _handle_teleop(self) -> None:
        """Send step request if in manual mode with active input."""
        if not self._manual_mode:
            return
        has_input = abs(self._fwd) > 0.01 or abs(self._yaw) > 0.01
        if has_input and self._engine.has_step_socket:
            self._engine.send_step_request(self._fwd, self._yaw)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:  # type: ignore[override]  # noqa: N802
        """Handle key press for teleop and dashboard control."""
        if event is None:
            return
        key = event.key()

        # ESC / Q → close
        if key in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Q):
            self.close()
            return

        # Tab → toggle manual mode
        if key == QtCore.Qt.Key.Key_Tab:
            if self._engine.has_step_socket:
                self._manual_mode = not self._manual_mode
            return

        # Movement
        if key in (QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_Up):
            self._fwd = self._linear_speed
        elif key in (QtCore.Qt.Key.Key_S, QtCore.Qt.Key.Key_Down):
            self._fwd = -self._linear_speed
        if key in (QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_Left):
            self._yaw = self._yaw_rate_max
        elif key in (QtCore.Qt.Key.Key_D, QtCore.Qt.Key.Key_Right):
            self._yaw = -self._yaw_rate_max

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent | None) -> None:  # type: ignore[override]  # noqa: N802
        """Reset velocity on key release."""
        if event is None:
            return
        key = event.key()
        if key in (
            QtCore.Qt.Key.Key_W,
            QtCore.Qt.Key.Key_S,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
        ):
            self._fwd = 0.0
        if key in (
            QtCore.Qt.Key.Key_A,
            QtCore.Qt.Key.Key_D,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ):
            self._yaw = 0.0
        super().keyReleaseEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # type: ignore[override]  # noqa: N802
        """Clean up ZMQ on window close."""
        self._timer.stop()
        self._engine.close()
        if event is not None:
            event.accept()
        super().closeEvent(event)


def run_dashboard(
    matrix_sub: str = "",
    actor_sub: str = "",
    step_endpoint: str = "",
    hz: float = 30.0,
    linear_speed: float = 1.5,
    yaw_rate: float = 1.5,
    max_distance_m: float = 30.0,
    scene_path: str | None = None,
) -> None:
    """Launch the Ghost-Matrix RL Dashboard as a standalone application."""
    app = pg.mkQApp("Ghost-Matrix RL Auditor")

    dashboard = GhostMatrixDashboard(
        matrix_sub=matrix_sub,
        actor_sub=actor_sub,
        step_endpoint=step_endpoint,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        max_distance_m=max_distance_m,
        scene_path=scene_path,
    )
    dashboard.show()
    app.exec()
