"""Ghost-Matrix RL Dashboard — PyQtGraph/Qt6 main application.

GPU-accelerated, dockable-panel layout with real-time RL training
curves, action-intention overlays, depth-mapped colormaps, semantic
legend, and Tab-toggle manual teleop.

Replaces the legacy OpenCV ``LiveDashboard`` with a professional
robotics observability suite.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from navi_auditor.dashboard.panels import (
    ImagePanel,
    RollingPlot,
    StatusBar,
)
from navi_auditor.dashboard.renderers import (
    VIEW_RANGE_M,
    compute_nav_metrics,
    depth_to_viridis,
    draw_semantic_legend,
    render_bev_occupancy,
    render_first_person,
    render_forward_polar,
    zoom_overhead,
)
from navi_auditor.stream_engine import StreamEngine

__all__: list[str] = ["GhostMatrixDashboard"]

# ── FOV slice fraction ───────────────────────────────────────────────
_FOV_FRACTION: float = 120.0 / 360.0


class GhostMatrixDashboard(QtWidgets.QMainWindow):
    """High-performance real-time RL training visualiser.

    Parameters
    ----------
    matrix_sub
        ZMQ SUB address for Section-Manager PUB (distance_matrix_v2).
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
        matrix_sub: str,
        actor_sub: str = "",
        step_endpoint: str = "",
        hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Ghost-Matrix RL Auditor")
        self.resize(1920, 1080)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        # Stream engine
        self._engine = StreamEngine(
            matrix_sub=matrix_sub,
            actor_sub=actor_sub,
            step_endpoint=step_endpoint,
        )

        # Teleop state
        self._linear_speed = linear_speed
        self._yaw_rate_max = yaw_rate
        self._manual_mode = False
        self._fwd = 0.0
        self._yaw = 0.0
        self._overhead_zoom = 1.0
        self._scan_history: list[np.ndarray] = []

        # ── build UI ─────────────────────────────────────────────────
        self._status_bar = StatusBar()
        self._build_layout()
        self._wire_plots()

        # ── tick timer ───────────────────────────────────────────────
        self._tick_ms = max(1, int(1000.0 / max(1.0, hz)))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._tick_ms)

    # ── layout construction ──────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the main window layout with dockable panels."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top status bar
        main_layout.addWidget(self._status_bar)

        # Body: splitter with left (viewport) and right (panels + plots)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(body, stretch=1)

        # Left: First-Person Viewport (~70%)
        self._fp_panel = ImagePanel(title="FIRST PERSON")
        self._fp_arrow = self._fp_panel.add_action_arrow()
        body.addWidget(self._fp_panel)

        # Right: vertical splitter with spatial panels + training plots
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        body.addWidget(right_splitter)

        # Right-top: tabbed spatial panels
        self._spatial_tabs = QtWidgets.QTabWidget()
        self._spatial_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333; background: #0d0d1a; }"
            "QTabBar::tab { background: #1a1a2e; color: #ccc; padding: 4px 12px; }"
            "QTabBar::tab:selected { background: #2a2a4e; color: #fff; }"
        )

        self._polar_panel = ImagePanel(title="FORWARD POLAR")
        self._overhead_panel = ImagePanel(title="OVERHEAD + HUD")
        self._overhead_arrow = self._overhead_panel.add_action_arrow()
        self._bev_panel = ImagePanel(title="BIRD'S EYE 360")
        self._depth_panel = ImagePanel(title="RAW DEPTH (Viridis)")

        self._spatial_tabs.addTab(self._polar_panel, "Polar")
        self._spatial_tabs.addTab(self._overhead_panel, "Overhead")
        self._spatial_tabs.addTab(self._bev_panel, "Bird's Eye")
        self._spatial_tabs.addTab(self._depth_panel, "Raw Depth")
        right_splitter.addWidget(self._spatial_tabs)

        # Right-bottom: training plots in a 2x2 grid
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QGridLayout(plot_container)
        plot_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.setSpacing(2)

        self._plot_reward = RollingPlot(title="Step Reward", color="#2e86de")
        self._plot_collision = RollingPlot(title="Collision", color="#e74c3c")
        self._plot_novelty = RollingPlot(title="Novelty", color="#27ae60")
        self._plot_coverage = RollingPlot(title="Eval Coverage", color="#8e44ad")

        plot_layout.addWidget(self._plot_reward, 0, 0)
        plot_layout.addWidget(self._plot_collision, 0, 1)
        plot_layout.addWidget(self._plot_novelty, 1, 0)
        plot_layout.addWidget(self._plot_coverage, 1, 1)
        right_splitter.addWidget(plot_container)

        # Set default split ratios
        body.setStretchFactor(0, 7)
        body.setStretchFactor(1, 3)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

    def _wire_plots(self) -> None:
        """Connect rolling plots to stream engine ring buffers."""
        state = self._engine.state
        self._plot_reward.set_data_from_deque(state.reward_history)
        self._plot_collision.set_data_from_deque(state.collision_history)
        self._plot_novelty.set_data_from_deque(state.novelty_history)
        self._plot_coverage.set_data_from_deque(state.eval_coverage_history)

    # ── tick / render loop ───────────────────────────────────────────

    def _tick(self) -> None:
        """Called every timer interval — poll ZMQ, update all panels."""
        self._engine.poll()
        state = self._engine.state

        # Determine mode
        has_training_data = len(state.reward_history) > 0
        if self._manual_mode:
            mode = "MANUAL"
        elif has_training_data:
            mode = "TRAINING"
        else:
            mode = "OBSERVER"
        self._status_bar.set_mode(mode)

        dm = state.latest_matrix
        action = state.latest_action

        if dm is not None:
            self._update_viewport(dm)
            self._update_spatial_panels(dm)
            self._update_status(dm, action)

        self._update_action_arrow(dm, action)
        self._refresh_plots()
        self._handle_teleop()

    def _update_viewport(self, dm: object) -> None:
        """Render first-person view and push to viewport panel."""
        from navi_contracts import DistanceMatrix

        assert isinstance(dm, DistanceMatrix)
        depth_2d = dm.depth[0]
        semantic_2d = dm.semantic[0]
        valid_2d = dm.valid_mask[0]
        az_bins = depth_2d.shape[0]

        fov_bins = max(1, int(az_bins * _FOV_FRACTION))
        centre_bin = az_bins // 2
        fov_lo = centre_bin - fov_bins // 2
        fov_hi = fov_lo + fov_bins

        fov_depth = depth_2d[fov_lo:fov_hi, :]
        fov_semantic = semantic_2d[fov_lo:fov_hi, :]
        fov_valid = valid_2d[fov_lo:fov_hi, :]

        fp_img, _center_m = render_first_person(
            fov_depth, fov_semantic, fov_valid, 960, 720,
        )
        draw_semantic_legend(fp_img, 14, 14)
        self._fp_panel.set_image(fp_img)

    def _update_spatial_panels(self, dm: object) -> None:
        """Update side spatial panels: polar, overhead, bev, raw depth."""
        from navi_contracts import DistanceMatrix

        assert isinstance(dm, DistanceMatrix)
        depth_2d = dm.depth[0]
        valid_2d = dm.valid_mask[0]
        az_bins = depth_2d.shape[0]

        fov_bins = max(1, int(az_bins * _FOV_FRACTION))
        centre_bin = az_bins // 2
        fov_lo = centre_bin - fov_bins // 2
        fov_hi = fov_lo + fov_bins
        fov_depth = depth_2d[fov_lo:fov_hi, :]
        fov_valid = valid_2d[fov_lo:fov_hi, :]

        # Polar scan
        polar_img = render_forward_polar(
            fov_depth, fov_valid, 320, 320,
            scan_history=self._scan_history,
        )
        self._polar_panel.set_image(polar_img)

        # Overhead minimap
        overhead_bgr = dm.overhead.astype(np.uint8, copy=False)
        if self._overhead_zoom > 1.01:
            overhead_bgr = zoom_overhead(overhead_bgr, self._overhead_zoom)
        overhead_bgr = cv2.convertScaleAbs(overhead_bgr, alpha=1.1, beta=45)
        self._overhead_panel.set_image(overhead_bgr)

        # Bird's Eye
        pose_list = list(self._engine.state.pose_history)
        bev_img = render_bev_occupancy(
            depth_2d, valid_2d, 320, 320,
            pose_history=pose_list,
        )
        self._bev_panel.set_image(bev_img)

        # Raw depth (Viridis)
        viridis_img = depth_to_viridis(fov_depth, fov_valid)
        viridis_resized = cv2.resize(
            viridis_img, (320, 320), interpolation=cv2.INTER_NEAREST,
        )
        self._depth_panel.set_image(viridis_resized)

    def _update_status(self, dm: object, action: object) -> None:
        """Update status bar from latest data."""
        from navi_contracts import Action, DistanceMatrix

        assert isinstance(dm, DistanceMatrix)
        p = dm.robot_pose
        self._status_bar.set_step(dm.step_id)
        self._status_bar.set_pose(p.x, p.y, p.z, p.yaw)

        depth_2d = dm.depth[0]
        valid_2d = dm.valid_mask[0]
        az_bins = depth_2d.shape[0]
        fov_bins = max(1, int(az_bins * _FOV_FRACTION))
        centre_bin = az_bins // 2
        fov_lo = centre_bin - fov_bins // 2
        fov_hi = fov_lo + fov_bins
        fov_depth = depth_2d[fov_lo:fov_hi, :]
        fov_valid = valid_2d[fov_lo:fov_hi, :]

        fwd, _left, _right = compute_nav_metrics(fov_depth, fov_valid)
        self._status_bar.set_nearest_obstacle(fwd * VIEW_RANGE_M)

        stale = max(0.0, time.time() - self._engine.state.last_rx_time)
        self._status_bar.set_stream_health(stale)

        if isinstance(action, Action):
            lin = float(action.linear_velocity[0, 0])
            yaw = float(action.angular_velocity[0, 2])
            self._status_bar.set_velocity(lin, yaw)

    def _update_action_arrow(self, dm: object, action: object) -> None:
        """Update the action intention arrow overlays."""
        from navi_contracts import Action

        if not isinstance(action, Action):
            self._fp_arrow.setVisible(False)
            self._overhead_arrow.setVisible(False)
            return

        lin = float(action.linear_velocity[0, 0])
        yaw = float(action.angular_velocity[0, 2])

        # Use latest advantage for colour if available
        state = self._engine.state
        advantage = float(state.advantage_history[-1]) if state.advantage_history else 0.0

        self._fp_arrow.setVisible(True)
        self._fp_arrow.update_from_action(
            linear_speed=lin,
            yaw_rate=yaw,
            advantage=advantage,
            img_width=960,
            img_height=720,
        )

        self._overhead_arrow.setVisible(True)
        self._overhead_arrow.update_from_action(
            linear_speed=lin,
            yaw_rate=yaw,
            advantage=advantage,
            img_width=320,
            img_height=320,
        )

    def _refresh_plots(self) -> None:
        """Redraw all rolling training plots."""
        self._plot_reward.refresh()
        self._plot_collision.refresh()
        self._plot_novelty.refresh()
        self._plot_coverage.refresh()

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

        # Zoom
        if key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self._overhead_zoom = min(6.0, self._overhead_zoom + 0.5)
            return
        if key == QtCore.Qt.Key.Key_Minus:
            self._overhead_zoom = max(1.0, self._overhead_zoom - 0.5)
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
    matrix_sub: str,
    actor_sub: str = "",
    step_endpoint: str = "",
    hz: float = 30.0,
    linear_speed: float = 1.5,
    yaw_rate: float = 1.5,
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
    )
    dashboard.show()
    app.exec()
