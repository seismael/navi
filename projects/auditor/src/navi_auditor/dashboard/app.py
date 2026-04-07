"""Ghost-Matrix RL Dashboard — actor 0 observer view.

The dashboard is intentionally visual-only: it renders one live actor 0
depth view with mode/status indication and a count of discovered actors.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from navi_auditor.dashboard.panels import (
    ImagePanel,
    StatusBar,
)
from navi_auditor.dashboard.renderers import (
    depth_to_observer_palette,
    render_first_person,
)
from navi_auditor.dashboard.status_line import build_status_metrics_line
from navi_auditor.demonstration_recorder import DemonstrationRecorder
from navi_auditor.stream_engine import StreamEngine, StreamState

__all__: list[str] = ["GhostMatrixDashboard"]

_LOG = logging.getLogger(__name__)


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
        start_manual: bool = False,
        enable_recording: bool = False,
        drone_max_speed: float = 5.0,
        drone_climb_rate: float = 2.0,
        drone_strafe_speed: float = 3.0,
        drone_yaw_rate: float = 3.0,
        max_steps: int = 0,
    ) -> None:
        super().__init__()
        title = "Ghost-Matrix RL Auditor"
        if start_manual:
            title = "Ghost-Matrix Explorer — WASD to navigate, ESC to quit"
        self.setWindowTitle(title)
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
        self._manual_mode = start_manual and self._engine.has_step_socket
        self._fwd = 0.0
        self._vert = 0.0
        self._yaw = 0.0
        self._scene_path = scene_path
        self._max_distance_m = float(max_distance_m)

        # ── demonstration recorder ───────────────────────────────────
        self._recorder: DemonstrationRecorder | None = None
        if enable_recording:
            self._recorder = DemonstrationRecorder(
                drone_max_speed=drone_max_speed,
                drone_climb_rate=drone_climb_rate,
                drone_strafe_speed=drone_strafe_speed,
                drone_yaw_rate=drone_yaw_rate,
                scene_name=scene_path or "",
            )
            # Auto-start recording so the user can just fly
            self._recorder.start()

        self._max_steps = max_steps

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
        # Ingest ZMQ messages (Standard: UI Throughput)
        _msgs_processed = self._engine.poll(max_messages=200)

        # Update actor count and scene name displays
        n_actors = self._engine.n_actors
        self._status_bar.set_actor_count(n_actors)

        state = self._engine.actor_states.get(0)

        # Scene name from any actor state that has it
        scene_name = ""
        for s in self._engine.actor_states.values():
            if s.current_scene_name:
                scene_name = s.current_scene_name
                break
        self._status_bar.set_scene_name(scene_name)
        if state is None:
            self._status_bar.set_mode("WAITING")
            self._status_bar.set_metrics_text(
                build_status_metrics_line(None, mode="WAITING"),
            )
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
            base_mode = "MANUAL"
            mode = "MANUAL"
            if self._recorder is not None and self._recorder.is_recording:
                steps = self._recorder.step_count
                if self._max_steps > 0:
                    mode = f"MANUAL ● REC ({steps}/{self._max_steps} steps)"
                else:
                    mode = f"MANUAL ● REC ({steps} steps)"
        elif has_inference_data:
            base_mode = "INFERENCE"
            mode = "INFERENCE"
        elif has_training_data:
            base_mode = "TRAINING"
            mode = "TRAINING"
        else:
            base_mode = "OBSERVER"
            mode = "OBSERVER"
        self._status_bar.set_mode(mode)
        self._status_bar.set_metrics_text(
            build_status_metrics_line(
                state,
                now=time.time(),
                fallback_state=self._shared_metrics_state(),
                mode=base_mode,
            ),
        )

        # Only re-render the actor panel when a genuinely new observation arrived.
        # This avoids wasting CPU on re-rendering stale frames and ensures the
        # displayed view always represents the latest training step.
        if self._engine.observation_updated and state.latest_matrix is not None:
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
        has_input = abs(self._fwd) > 0.01 or abs(self._vert) > 0.01 or abs(self._yaw) > 0.01
        if has_input and self._engine.has_step_socket:
            self._engine.send_step_request(self._fwd, self._yaw, self._vert)
            # Capture demonstration data if recording
            if (
                self._recorder is not None
                and self._recorder.is_recording
            ):
                state = self._engine.actor_states.get(0)
                if state is not None and state.latest_matrix is not None:
                    self._recorder.capture(
                        state.latest_matrix,
                        linear_velocity=self._fwd,
                        yaw_rate=self._yaw,
                        vertical_velocity=self._vert,
                    )
                    if (
                        self._max_steps > 0
                        and self._recorder.step_count >= self._max_steps
                    ):
                        _LOG.info(
                            "Reached max steps (%d) — auto-closing.",
                            self._max_steps,
                        )
                        self.close()
                        return

    def _toggle_recording(self) -> None:
        """Toggle demonstration recording on/off."""
        if self._recorder is None:
            self._status_bar.set_mode("MANUAL (recording not enabled — use --record)")
            return
        if self._recorder.is_recording:
            self._recorder.stop()
            if self._recorder.step_count > 0:
                try:
                    save_path = self._recorder.save()
                    _LOG.info("Demonstration saved to %s", save_path)
                    self._status_bar.set_mode(f"MANUAL — saved {save_path.name}")
                except Exception:
                    _LOG.exception("Failed to save demonstration")
                    self._status_bar.set_mode("MANUAL — save FAILED")
            else:
                self._status_bar.set_mode("MANUAL — empty recording discarded")
        else:
            self._recorder.start()
            self._status_bar.set_mode("MANUAL ● REC")

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

        # Vertical (Space/R = up, Shift/F = down)
        if key in (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_R):
            self._vert = self._linear_speed
        elif key in (QtCore.Qt.Key.Key_Shift, QtCore.Qt.Key.Key_F):
            self._vert = -self._linear_speed

        # F12 — diagnostic snapshot
        if key == QtCore.Qt.Key.Key_F12:
            self._capture_snapshot()
            return

        # B — toggle demonstration recording
        if key == QtCore.Qt.Key.Key_B:
            self._toggle_recording()
            return

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
        if key in (
            QtCore.Qt.Key.Key_Space,
            QtCore.Qt.Key.Key_R,
            QtCore.Qt.Key.Key_Shift,
            QtCore.Qt.Key.Key_F,
        ):
            self._vert = 0.0
        super().keyReleaseEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # type: ignore[override]  # noqa: N802
        """Clean up ZMQ on window close."""
        # Auto-save any in-progress recording
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop()
            if self._recorder.step_count > 0:
                try:
                    save_path = self._recorder.save()
                    _LOG.info("Auto-saved demonstration on close: %s", save_path)
                except Exception:
                    _LOG.exception("Failed to auto-save demonstration on close")
        self._timer.stop()
        self._engine.close()
        if event is not None:
            event.accept()
        super().closeEvent(event)

    # ── diagnostic snapshot (F12) ───────────────────────────────────────

    def _capture_snapshot(self) -> None:
        """Dump a comprehensive diagnostic snapshot of actor 0 to disk."""
        from navi_contracts import DistanceMatrix

        state = self._engine.actor_states.get(0)
        if state is None or state.latest_matrix is None:
            self._status_bar.flash_message("Snapshot failed: no observation data")
            return

        dm = state.latest_matrix
        assert isinstance(dm, DistanceMatrix)

        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("artifacts") / "dashboard-captures" / f"snapshot_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_depth = np.asarray(dm.depth[0], dtype=np.float32)
        raw_valid = np.asarray(dm.valid_mask[0], dtype=bool)
        raw_semantic = np.asarray(dm.semantic[0], dtype=np.int32)
        raw_delta = np.asarray(dm.delta_depth[0], dtype=np.float32)

        # ── 1. Raw arrays ───────────────────────────────────────────
        np.savez_compressed(
            out_dir / "distance_matrix_arrays.npz",
            depth=raw_depth,
            delta_depth=raw_delta,
            valid_mask=raw_valid,
            semantic=raw_semantic,
        )

        # ── 2. Rendered perspective (current viewport) ───────────
        perspective_img, _center_m = render_first_person(
            raw_depth, raw_semantic, raw_valid, 960, 720,
            pitch=float(dm.robot_pose.pitch),
        )
        cv2.imwrite(str(out_dir / "perspective.png"), perspective_img)

        # ── 3. Full 360° panoramic depth heatmap ─────────────────
        pano_depth = raw_depth.T.astype(np.float32)   # (el, az)
        pano_valid = raw_valid.T
        pano_img = depth_to_observer_palette(pano_depth, pano_valid, fog_of_war=True)
        pano_img = cv2.resize(pano_img, (960, 240), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "panorama_full.png"), pano_img)

        # ── 4. Valid mask visualisation ───────────────────────────
        valid_vis: np.ndarray = (raw_valid.T.astype(np.uint8) * np.uint8(255))
        valid_vis = cv2.resize(valid_vis, (960, 240), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "panorama_valid.png"), valid_vis)

        # ── 5. Robot pose ────────────────────────────────────────
        pose = dm.robot_pose
        pose_dict = {
            "x": pose.x, "y": pose.y, "z": pose.z,
            "roll": pose.roll, "pitch": pose.pitch, "yaw": pose.yaw,
            "timestamp": pose.timestamp,
        }
        (out_dir / "robot_pose.json").write_text(
            json.dumps(pose_dict, indent=2), encoding="utf-8",
        )

        # ── 6. Observation metadata ─────────────────────────────
        obs_meta = {
            "episode_id": int(dm.episode_id),
            "step_id": int(dm.step_id),
            "env_ids": [int(x) for x in dm.env_ids],
            "matrix_shape": [int(raw_depth.shape[0]), int(raw_depth.shape[1])],
            "timestamp": float(dm.timestamp),
        }
        (out_dir / "observation_metadata.json").write_text(
            json.dumps(obs_meta, indent=2), encoding="utf-8",
        )

        # ── 7. Comprehensive actor state dump ───────────────────
        actor_dump = _build_actor_state_dump(state, dm, raw_depth, raw_valid)
        (out_dir / "actor_state.json").write_text(
            json.dumps(actor_dump, indent=2), encoding="utf-8",
        )

        self._status_bar.flash_message(f"Snapshot saved \u2192 {out_dir}")
        _LOG.info("Dashboard snapshot saved to %s", out_dir)


# ── snapshot helpers (module-level to keep class lean) ───────────────


def _deque_tail(d: deque[float], n: int = 200) -> list[float]:
    """Return at most the last *n* values from a ring buffer."""
    items = list(d)
    return items[-n:]


def _build_actor_state_dump(
    state: StreamState,
    dm: object,
    raw_depth: np.ndarray,
    raw_valid: np.ndarray,
) -> dict[str, object]:
    """Build a JSON-serialisable dict capturing the full actor 0 diagnostic state."""
    from navi_contracts import Action, DistanceMatrix

    assert isinstance(dm, DistanceMatrix)

    valid_depths = raw_depth[raw_valid] if np.any(raw_valid) else np.array([], dtype=np.float32)
    max_dist_approx = float(raw_depth.max()) if raw_depth.size > 0 else 1.0
    starvation_ratio = float(np.mean(raw_depth >= max_dist_approx - 1e-3)) if raw_depth.size > 0 else 0.0
    proximity_ratio = float(np.mean(raw_valid & (raw_depth <= 0.15))) if raw_depth.size > 0 else 0.0

    action_info: dict[str, object] = {}
    if state.latest_action is not None and isinstance(state.latest_action, Action):
        act = state.latest_action
        action_info = {
            "linear_velocity": act.linear_velocity[0].tolist() if len(act.linear_velocity) > 0 else [],
            "angular_velocity": act.angular_velocity[0].tolist() if len(act.angular_velocity) > 0 else [],
            "policy_id": act.policy_id,
            "step_id": int(act.step_id),
        }

    return {
        "capture_utc": datetime.now(tz=UTC).isoformat(),
        "scene_name": state.current_scene_name,
        "action": action_info,
        "depth_statistics": {
            "valid_ratio": float(np.mean(raw_valid.astype(np.float32))),
            "starvation_ratio": starvation_ratio,
            "proximity_ratio": proximity_ratio,
            "valid_min": float(valid_depths.min()) if valid_depths.size > 0 else None,
            "valid_max": float(valid_depths.max()) if valid_depths.size > 0 else None,
            "valid_mean": float(valid_depths.mean()) if valid_depths.size > 0 else None,
            "valid_median": float(np.median(valid_depths)) if valid_depths.size > 0 else None,
            "valid_std": float(valid_depths.std()) if valid_depths.size > 0 else None,
            "total_bins": int(raw_depth.size),
            "valid_bins": int(np.sum(raw_valid)),
        },
        "histories": {
            "reward": _deque_tail(state.reward_history),
            "collision": _deque_tail(state.collision_history),
            "forward_cmd": _deque_tail(state.forward_cmd_history),
            "yaw_cmd": _deque_tail(state.yaw_cmd_history),
            "front_depth": _deque_tail(state.front_depth_history),
            "mean_depth": _deque_tail(state.mean_depth_history),
            "near_fraction": _deque_tail(state.near_fraction_history),
        },
        "ppo_histories": {
            "reward_ema": _deque_tail(state.ppo_reward_ema_history),
            "policy_loss": _deque_tail(state.ppo_policy_loss_history),
            "value_loss": _deque_tail(state.ppo_value_loss_history),
            "entropy": _deque_tail(state.ppo_entropy_history),
            "kl": _deque_tail(state.ppo_kl_history),
            "clip_fraction": _deque_tail(state.ppo_clip_fraction_history),
            "total_loss": _deque_tail(state.ppo_total_loss_history),
            "lr": _deque_tail(state.ppo_lr_history),
            "raw_reward": _deque_tail(state.ppo_raw_reward_history),
            "shaped_reward": _deque_tail(state.ppo_shaped_reward_history),
            "done": _deque_tail(state.ppo_done_history),
        },
        "perf_histories": {
            "sps": _deque_tail(state.perf_sps_history),
            "forward_ms": _deque_tail(state.perf_forward_ms_history),
            "batch_step_ms": _deque_tail(state.perf_batch_step_ms_history),
            "opt_ms": _deque_tail(state.perf_opt_ms_history),
            "tick_ms": _deque_tail(state.perf_tick_ms_history),
        },
    }


def run_dashboard(
    matrix_sub: str = "",
    actor_sub: str = "",
    step_endpoint: str = "",
    hz: float = 30.0,
    linear_speed: float = 1.5,
    yaw_rate: float = 1.5,
    max_distance_m: float = 30.0,
    scene_path: str | None = None,
    start_manual: bool = False,
    enable_recording: bool = False,
    drone_max_speed: float = 5.0,
    drone_climb_rate: float = 2.0,
    drone_strafe_speed: float = 3.0,
    drone_yaw_rate: float = 3.0,
    max_steps: int = 0,
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
        start_manual=start_manual,
        enable_recording=enable_recording,
        drone_max_speed=drone_max_speed,
        drone_climb_rate=drone_climb_rate,
        drone_strafe_speed=drone_strafe_speed,
        drone_yaw_rate=drone_yaw_rate,
        max_steps=max_steps,
    )
    dashboard.show()
    app.exec()
