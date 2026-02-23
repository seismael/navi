"""Ghost-Matrix RL Dashboard — PyQtGraph/Qt6 main application.

GPU-accelerated, dockable-panel layout with real-time RL training
curves, depth-mapped colormaps, and live occupancy map.

Provides a two-panel spatial view (Live Map + Raw Depth) on the left
and a dense two-column chart grid of training metrics on the right.
"""

from __future__ import annotations

import time

import cv2
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from navi_auditor.dashboard.occupancy_view import OccupancyMap
from navi_auditor.dashboard.panels import (
    ImagePanel,
    RollingPlot,
    StatusBar,
)
from navi_auditor.dashboard.renderers import (
    VIEW_RANGE_M,
    add_orientation_guides,
    compute_nav_metrics,
    depth_to_viridis,
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
        ZMQ SUB address for Environment PUB (distance_matrix_v2).
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
        scene_path: str | None = None,
        n_actors: int = 1,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Ghost-Matrix RL Auditor")
        self.resize(1920, 1080)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        self._n_actors = max(1, n_actors)
        self._active_actor: int = 0

        # Stream engine
        self._engine = StreamEngine(
            matrix_sub=matrix_sub,
            actor_sub=actor_sub,
            step_endpoint=step_endpoint,
            n_actors=self._n_actors,
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
        self._build_layout()
        self._wire_plots()

        # ── tick timer ───────────────────────────────────────────────
        self._tick_ms = max(1, int(1000.0 / max(1.0, hz)))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._tick_ms)

    # ── layout construction ──────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the main window layout: spatial views left, 2-col charts right."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top status bar
        main_layout.addWidget(self._status_bar)

        # Actor tab bar (only visible when n_actors > 1)
        if self._n_actors > 1:
            self._actor_tab_bar = QtWidgets.QTabBar()
            self._actor_tab_bar.setStyleSheet(
                "QTabBar::tab { background: #1a1a2e; color: #aaa; "
                "padding: 6px 18px; margin: 1px; border-radius: 4px; } "
                "QTabBar::tab:selected { background: #2e86de; color: #fff; }"
            )
            for i in range(self._n_actors):
                self._actor_tab_bar.addTab(f"Actor {i}")
            self._actor_tab_bar.currentChanged.connect(self._on_actor_tab_changed)
            main_layout.addWidget(self._actor_tab_bar)

        # Body: splitter with left (spatial views) and right (chart grid)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(body, stretch=1)

        # ── Left: stacked spatial views (~30%) ───────────────────────
        view_container = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_container)
        view_layout.setContentsMargins(2, 2, 2, 2)
        view_layout.setSpacing(2)

        # Live 2D Occupancy Map
        self._occ_map = OccupancyMap(max_distance=15.0)
        self._env_panel = ImagePanel(title="LIVE MAP")
        view_layout.addWidget(self._env_panel, stretch=1)

        # Raw Depth (Viridis)
        self._depth_panel = ImagePanel(title="RAW DEPTH (Viridis)")
        view_layout.addWidget(self._depth_panel, stretch=1)

        body.addWidget(view_container)

        # ── Right: 2-column training metric chart grid (~70%) ────────
        chart_scroll = QtWidgets.QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        chart_scroll.setStyleSheet(
            "QScrollArea { border: none; background: #0d0d1a; }"
        )

        chart_container = QtWidgets.QWidget()
        chart_grid = QtWidgets.QGridLayout(chart_container)
        chart_grid.setContentsMargins(2, 2, 2, 2)
        chart_grid.setSpacing(3)

        # Create all plots
        self._plot_reward_ema = RollingPlot(
            title="Reward EMA", color="#2e86de",
        )
        self._plot_reward = RollingPlot(title="Step Reward", color="#2e86de")
        self._plot_episode_return = RollingPlot(
            title="Episode Return", color="#8e44ad",
        )
        self._plot_episode_len = RollingPlot(
            title="Episode Length", color="#1abc9c",
        )
        self._plot_collision = RollingPlot(
            title="Done / Collision", color="#e74c3c",
        )
        self._plot_forward = RollingPlot(
            title="Forward Velocity", color="#27ae60",
        )
        self._plot_yaw = RollingPlot(title="Yaw Rate", color="#f39c12")
        self._plot_front_depth = RollingPlot(
            title="Front Depth (min)", color="#3498db",
        )
        self._plot_mean_depth = RollingPlot(title="Mean Depth", color="#e67e22")
        self._plot_near_fraction = RollingPlot(
            title="Near-Object Fraction", color="#e74c3c",
        )
        self._plot_policy_loss = RollingPlot(
            title="Policy Loss", color="#9b59b6",
        )
        self._plot_value_loss = RollingPlot(
            title="Value Loss", color="#e67e22",
        )
        self._plot_entropy = RollingPlot(
            title="Entropy", color="#1abc9c",
        )
        self._plot_kl = RollingPlot(
            title="Approx KL", color="#e74c3c",
        )
        self._plot_clip_fraction = RollingPlot(
            title="Clip Fraction", color="#f39c12",
        )
        self._plot_rnd_loss = RollingPlot(
            title="RND Loss", color="#3498db",
        )
        self._plot_intrinsic = RollingPlot(
            title="Intrinsic Reward", color="#2ecc71",
        )
        self._plot_loop_sim = RollingPlot(
            title="Loop Similarity", color="#e74c3c",
        )
        self._plot_beta = RollingPlot(
            title="Beta (intrinsic coeff)", color="#9b59b6",
        )

        # Arrange in 2-column grid — most important charts at top
        # Row 0: headline metrics
        # Row 1-4: PPO training diagnostics
        # Row 5+: sensor / navigation metrics
        chart_pairs: list[tuple[RollingPlot, RollingPlot]] = [
            (self._plot_reward_ema, self._plot_episode_return),
            (self._plot_policy_loss, self._plot_value_loss),
            (self._plot_entropy, self._plot_kl),
            (self._plot_clip_fraction, self._plot_rnd_loss),
            (self._plot_reward, self._plot_collision),
            (self._plot_episode_len, self._plot_forward),
            (self._plot_yaw, self._plot_front_depth),
            (self._plot_mean_depth, self._plot_near_fraction),
            (self._plot_intrinsic, self._plot_loop_sim),
            (self._plot_beta, self._plot_beta),  # placeholder, handled below
        ]

        chart_h = 110
        for row, (left, right) in enumerate(chart_pairs):
            if row == len(chart_pairs) - 1:
                # Last row: beta spans both columns
                self._plot_beta.setMinimumHeight(chart_h)
                self._plot_beta.setMaximumHeight(chart_h * 2)
                chart_grid.addWidget(self._plot_beta, row, 0, 1, 2)
                break
            left.setMinimumHeight(chart_h)
            left.setMaximumHeight(chart_h * 2)
            right.setMinimumHeight(chart_h)
            right.setMaximumHeight(chart_h * 2)
            chart_grid.addWidget(left, row, 0)
            chart_grid.addWidget(right, row, 1)

        chart_grid.setRowStretch(len(chart_pairs), 1)
        chart_scroll.setWidget(chart_container)
        body.addWidget(chart_scroll)

        # Split ratios: ~30% spatial views, ~70% charts
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 7)

    def _wire_plots(self) -> None:
        """Connect rolling plots to the active actor's stream state."""
        state = self._engine.actor_states[self._active_actor]
        self._plot_reward.set_data_from_deque(state.reward_history)
        self._plot_episode_return.set_data_from_deque(state.episode_return_history)
        self._plot_collision.set_data_from_deque(state.collision_history)
        self._plot_episode_len.set_data_from_deque(state.episode_length_history)
        self._plot_forward.set_data_from_deque(state.forward_cmd_history)
        self._plot_yaw.set_data_from_deque(state.yaw_cmd_history)
        self._plot_front_depth.set_data_from_deque(state.front_depth_history)
        self._plot_mean_depth.set_data_from_deque(state.mean_depth_history)
        self._plot_near_fraction.set_data_from_deque(state.near_fraction_history)
        # PPO-specific metrics
        self._plot_reward_ema.set_data_from_deque(state.ppo_reward_ema_history)
        self._plot_policy_loss.set_data_from_deque(state.ppo_policy_loss_history)
        self._plot_value_loss.set_data_from_deque(state.ppo_value_loss_history)
        self._plot_entropy.set_data_from_deque(state.ppo_entropy_history)
        self._plot_kl.set_data_from_deque(state.ppo_kl_history)
        self._plot_clip_fraction.set_data_from_deque(
            state.ppo_clip_fraction_history,
        )
        self._plot_rnd_loss.set_data_from_deque(state.ppo_rnd_loss_history)
        self._plot_intrinsic.set_data_from_deque(
            state.ppo_intrinsic_reward_history,
        )
        self._plot_loop_sim.set_data_from_deque(
            state.ppo_loop_similarity_history,
        )
        self._plot_beta.set_data_from_deque(state.ppo_beta_history)

    # ── actor tab switching ────────────────────────────────────────────

    def _on_actor_tab_changed(self, index: int) -> None:
        """Switch the displayed actor when a tab is clicked."""
        self._active_actor = max(0, min(index, self._n_actors - 1))
        self._wire_plots()

    # ── tick / render loop ───────────────────────────────────────────

    def _tick(self) -> None:
        """Called every timer interval — poll ZMQ, update all panels."""
        self._engine.poll()
        state = self._engine.actor_states[self._active_actor]

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
            self._update_spatial_panels(dm)
            self._update_status(dm, action)

        self._refresh_plots()
        self._handle_teleop()

    def _update_spatial_panels(self, dm: object) -> None:
        """Update spatial view panels: live occupancy map + raw depth."""
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

        # Live occupancy map — accumulate + render
        p = dm.robot_pose
        self._occ_map.update(
            depth_2d, valid_2d, p.x, p.z, p.yaw, dm.episode_id,
        )
        occ_img = self._occ_map.render(480, 480)
        self._env_panel.set_image(occ_img)

        # Raw depth (Viridis) — transpose to fix 180° flip
        viridis_img = depth_to_viridis(fov_depth.T, fov_valid.T)
        viridis_resized = cv2.resize(
            viridis_img, (480, 360), interpolation=cv2.INTER_NEAREST,
        )
        add_orientation_guides(viridis_resized)
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

    def _refresh_plots(self) -> None:
        """Redraw all rolling training plots."""
        self._plot_reward.refresh()
        self._plot_episode_return.refresh()
        self._plot_collision.refresh()
        self._plot_episode_len.refresh()
        self._plot_forward.refresh()
        self._plot_yaw.refresh()
        self._plot_front_depth.refresh()
        self._plot_mean_depth.refresh()
        self._plot_near_fraction.refresh()
        # PPO-specific
        self._plot_reward_ema.refresh()
        self._plot_policy_loss.refresh()
        self._plot_value_loss.refresh()
        self._plot_entropy.refresh()
        self._plot_kl.refresh()
        self._plot_clip_fraction.refresh()
        self._plot_rnd_loss.refresh()
        self._plot_intrinsic.refresh()
        self._plot_loop_sim.refresh()
        self._plot_beta.refresh()

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
    matrix_sub: str,
    actor_sub: str = "",
    step_endpoint: str = "",
    hz: float = 30.0,
    linear_speed: float = 1.5,
    yaw_rate: float = 1.5,
    scene_path: str | None = None,
    n_actors: int = 1,
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
        scene_path=scene_path,
        n_actors=n_actors,
    )
    dashboard.show()
    app.exec()
