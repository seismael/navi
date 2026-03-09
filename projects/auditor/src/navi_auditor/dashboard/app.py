"""Ghost-Matrix RL Dashboard — single selected actor view.

The dashboard is intentionally visual-only: it renders one live actor
depth view with mode/status indication. By default actor 0 is shown for
scalability. Optional selector mode can be enabled to switch actor.
"""

from __future__ import annotations

import time

import cv2
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from navi_auditor.dashboard.panels import (
    ImagePanel,
    StatusBar,
)
from navi_auditor.dashboard.renderers import (
    add_orientation_guides,
    depth_to_viridis,
)
from navi_auditor.dashboard.status_line import build_status_metrics_line
from navi_auditor.stream_engine import StreamEngine

__all__: list[str] = ["GhostMatrixDashboard"]

# ── FOV slice fraction ───────────────────────────────────────────────
_FOV_FRACTION: float = 120.0 / 360.0


class GhostMatrixDashboard(QtWidgets.QMainWindow):
    """High-performance real-time selected-actor visualiser.

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
        actor_id: int = 0,
        enable_actor_selector: bool = False,
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
        scene_path: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Ghost-Matrix RL Auditor")
        self.resize(1920, 1080)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        self._known_actors: list[int] = []
        self._selected_actor: int = actor_id
        self._enable_actor_selector = enable_actor_selector

        # Stream engine
        self._engine = StreamEngine(
            matrix_sub=matrix_sub,
            actor_sub=actor_sub,
            step_endpoint=step_endpoint,
            selected_actor_id=None if enable_actor_selector else actor_id,
        )

        # Teleop state
        self._linear_speed = linear_speed
        self._yaw_rate_max = yaw_rate
        self._manual_mode = False
        self._fwd = 0.0
        self._yaw = 0.0
        self._scene_path = scene_path

        # ── build UI ─────────────────────────────────────────────────
        self._status_bar = StatusBar()
        self._actor_panel = ImagePanel(title="LIVE ACTOR")
        self._build_layout()

        # ── tick timer ───────────────────────────────────────────────
        self._tick_ms = max(1, int(1000.0 / max(1.0, hz)))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._tick_ms)

    # ── layout construction ──────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the main window layout with one selected actor view."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top status bar
        main_layout.addWidget(self._status_bar)

        if self._enable_actor_selector:
            self._actor_selector = QtWidgets.QComboBox()
            self._actor_selector.setStyleSheet(
                "QComboBox { background: #1a1a2e; color: #fff; border: 1px solid #333; "
                "padding: 5px; border-radius: 4px; min-width: 120px; font-weight: bold; } "
                "QComboBox::drop-down { border: none; } "
                "QComboBox QAbstractItemView { background: #1a1a2e; color: #fff; selection-background-color: #2e86de; }"
            )
            self._actor_selector.currentIndexChanged.connect(self._on_actor_selector_changed)

            selector_container = QtWidgets.QWidget()
            selector_layout = QtWidgets.QHBoxLayout(selector_container)
            selector_layout.setContentsMargins(10, 5, 10, 5)
            selector_layout.addWidget(QtWidgets.QLabel("ACTOR:"))
            selector_layout.addWidget(self._actor_selector)
            selector_layout.addStretch()
            main_layout.addWidget(selector_container)

        main_layout.addWidget(self._actor_panel, stretch=1)

    def _refresh_selector(self) -> None:
        """Refresh actor selector options from discovered stream actors."""
        if not self._enable_actor_selector:
            return
        discovered = sorted(self._engine.actor_states.keys())
        if discovered != self._known_actors:
            self._known_actors = discovered
            self._actor_selector.blockSignals(True)
            self._actor_selector.clear()
            for actor_id in discovered:
                self._actor_selector.addItem(f"Actor {actor_id}", actor_id)
            idx = self._actor_selector.findData(self._selected_actor)
            if idx >= 0:
                self._actor_selector.setCurrentIndex(idx)
            self._actor_selector.blockSignals(False)

    def _on_actor_selector_changed(self, _index: int) -> None:
        """Switch selected actor from UI selector."""
        if not self._enable_actor_selector:
            return
        actor_id = self._actor_selector.currentData()
        if actor_id is None:
            return
        self._selected_actor = int(actor_id)
        self._engine.set_selected_actor(self._selected_actor)

    # ── tick / render loop ───────────────────────────────────────────

    def _tick(self) -> None:
        """Called every timer interval — poll ZMQ, update all panels."""
        # Ingest capped ZMQ burst (Standard: UI Throughput)
        _msgs_processed = self._engine.poll(max_messages=100)

        self._refresh_selector()
        state = self._engine.actor_states.get(self._selected_actor)
        if state is None:
            self._status_bar.set_mode("WAITING")
            self._status_bar.set_metrics_text(build_status_metrics_line(None))
            return

        # Determine mode (Standard: Mode Detection)
        has_training_data = (len(state.reward_history) > 0 or len(state.ppo_reward_ema_history) > 0)
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
            build_status_metrics_line(state, now=time.time()),
        )

        if state.latest_matrix is not None:
            self._update_actor_panel(self._actor_panel, state.latest_matrix)

        self._handle_teleop()

    def _update_actor_panel(self, panel: ImagePanel, dm: object) -> None:
        """Update one actor panel with depth-derived viridis view."""
        import numpy as np

        from navi_contracts import DistanceMatrix

        assert isinstance(dm, DistanceMatrix)
        # Roll so bin 0 (Forward) is in the middle of the azimuth range
        depth_2d = np.roll(dm.depth[0], shift=dm.depth[0].shape[0] // 2, axis=0)
        valid_2d = np.roll(dm.valid_mask[0], shift=dm.valid_mask[0].shape[0] // 2, axis=0)
        az_bins = depth_2d.shape[0]

        fov_bins = max(1, int(az_bins * _FOV_FRACTION))
        centre_bin = az_bins // 2
        fov_lo = centre_bin - fov_bins // 2
        fov_hi = fov_lo + fov_bins
        fov_depth = depth_2d[fov_lo:fov_hi, :]
        fov_valid = valid_2d[fov_lo:fov_hi, :]

        # Raw depth (Viridis) — transpose to put Azimuth on X and Elevation on Y
        viridis_img = depth_to_viridis(fov_depth.T, fov_valid.T)
        viridis_resized = cv2.resize(
            viridis_img, (480, 360), interpolation=cv2.INTER_NEAREST,
        )
        add_orientation_guides(viridis_resized)
        panel.set_image(viridis_resized)

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
            QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_S,
            QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down,
        ):
            self._fwd = 0.0
        if key in (
            QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_D,
            QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right,
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
    actor_id: int = 0,
    enable_actor_selector: bool = False,
    hz: float = 30.0,
    linear_speed: float = 1.5,
    yaw_rate: float = 1.5,
    scene_path: str | None = None,
) -> None:
    """Launch the Ghost-Matrix RL Dashboard as a standalone application."""
    app = pg.mkQApp("Ghost-Matrix RL Auditor")

    dashboard = GhostMatrixDashboard(
        matrix_sub=matrix_sub,
        actor_sub=actor_sub,
        step_endpoint=step_endpoint,
        actor_id=actor_id,
        enable_actor_selector=enable_actor_selector,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        scene_path=scene_path,
    )
    dashboard.show()
    app.exec()
