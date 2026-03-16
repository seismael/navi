"""Individual PyQtGraph panel widgets for the Ghost-Matrix dashboard.

Each class wraps a single logical area of the dashboard — a viewport,
a plot, or a status indicator — using ``pyqtgraph`` and ``PyQt6``
primitives.  They are composed together by ``app.GhostMatrixDashboard``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

__all__: list[str] = [
    "ActionArrow",
    "ImagePanel",
    "RollingPlot",
    "StatusBar",
]


# ── ImagePanel — displays a BGR NumPy frame via pyqtgraph ────────────


class ImagePanel(pg.GraphicsLayoutWidget):
    """Panel that displays a BGR uint8 NumPy image via ``ImageItem``.

    Optionally overlays an action-intention arrow at the viewport centre.
    """

    def __init__(self, title: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._title = title

        self._view = self.addViewBox(row=0, col=0)
        self._view.setAspectLocked(False)
        self._view.invertY(True)
        self._view.setMenuEnabled(False)

        self._img_item = pg.ImageItem()
        self._img_item.setAutoDownsample(False)
        self._view.addItem(self._img_item)

        # Action intention arrow overlay
        self._arrow: ActionArrow | None = None

        if title:
            label = pg.LabelItem(title, color="w", size="9pt")
            self.addItem(label, row=1, col=0)

    def set_image(self, bgr: np.ndarray) -> None:
        """Update displayed image from BGR uint8 array (H, W, 3)."""
        # pyqtgraph ImageItem expects (W, H, 3) RGB
        rgb = bgr[:, :, ::-1]  # BGR → RGB
        transposed = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        self._img_item.setImage(transposed, autoLevels=False, levels=(0, 255))
        self._view.setRange(
            xRange=[0, transposed.shape[0]],
            yRange=[0, transposed.shape[1]],
            padding=0,
        )

    def add_action_arrow(self) -> ActionArrow:
        """Create and attach a centred action arrow overlay."""
        self._arrow = ActionArrow()
        self._view.addItem(self._arrow)
        return self._arrow

    def get_arrow(self) -> ActionArrow | None:
        """Return the attached arrow, if any."""
        return self._arrow


# ── ActionArrow — direction + confidence overlay ─────────────────────


class ActionArrow(pg.ArrowItem):
    """Visual indicator of the actor's current action intention.

    Length ∝ linear velocity, angle ∝ yaw rate.
    Colour encodes advantage: green = good move, red = bad move.
    """

    def __init__(self) -> None:
        super().__init__(
            angle=90,
            tipAngle=30,
            headLen=16,
            tailLen=50,
            tailWidth=5,
            pen=pg.mkPen("w", width=1),
            brush=pg.mkBrush(0, 220, 120, 200),
        )
        self._base_tail = 50

    def update_from_action(
        self,
        linear_speed: float,
        yaw_rate: float,
        advantage: float = 0.0,
        img_width: int = 640,
        img_height: int = 480,
    ) -> None:
        """Reposition, rotate, and recolour the arrow from live data."""
        # Position at viewport centre
        self.setPos(img_width / 2.0, img_height / 2.0)

        # Angle: 90 = pointing up; add yaw offset (degrees)
        yaw_deg = float(np.clip(yaw_rate, -2.0, 2.0)) * -45.0
        self.setStyle(angle=90.0 + yaw_deg)

        # Length proportional to speed
        speed_norm = float(np.clip(abs(linear_speed), 0.0, 2.0)) / 2.0
        tail = int(self._base_tail * (0.3 + 0.7 * speed_norm))
        self.setStyle(tailLen=tail)

        # Colour from advantage: green (positive) → red (negative)
        adv_clamped = float(np.clip(advantage, -1.5, 1.5))
        t = (adv_clamped + 1.5) / 3.0  # 0..1
        r = int(255 * (1.0 - t))
        g = int(255 * t)
        b = 40
        self.setBrush(pg.mkBrush(r, g, b, 200))


# ── RollingPlot — time-series with ring buffer ───────────────────────


class RollingPlot(pg.PlotWidget):
    """Live-updating time-series plot backed by a ring buffer.

    Shows the most recent ``maxlen`` data points with auto-scaling Y
    axis and a stable X range.
    """

    def __init__(
        self,
        title: str = "",
        color: str = "#2e86de",
        maxlen: int = 2000,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent, title=title)

        self.setBackground("#1a1a2e")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

        self._pen = pg.mkPen(color=color, width=2)
        self._curve = self.plot(pen=self._pen)
        self._maxlen = maxlen
        self._data: deque[float] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        """Add a single data point."""
        self._data.append(value)

    def set_data_from_deque(self, data: deque[float]) -> None:
        """Replace internal buffer with an external deque reference."""
        self._data = data

    def refresh(self) -> None:
        """Redraw the curve from current buffer contents."""
        if len(self._data) < 2:
            return
        y = np.array(self._data, dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        self._curve.setData(x, y)


# ── StatusBar — top-bar state indicators ─────────────────────────────


class StatusBar(QtWidgets.QFrame):
    """Minimal status bar with mode indicator and compact metrics."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(
            "QFrame { background: #0d0d1a; border-bottom: 1px solid #333; }"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(16)

        # Mode indicator
        self._mode_label = QtWidgets.QLabel("[ OBSERVER ]")
        self._mode_label.setStyleSheet(
            "color: #aaa; font-weight: 700; font-size: 14px; padding: 2px 8px;"
        )
        layout.addWidget(self._mode_label)

        self._metrics_label = QtWidgets.QLabel("stall=-- | rollout_sps=-- | env_sps=-- | per_actor_sps=-- | ema=-- | ep=0 | step=--")
        self._metrics_label.setStyleSheet(
            "color: #d5dde8; font-weight: 700; font-size: 14px; padding: 2px 4px;"
        )
        layout.addWidget(self._metrics_label)

        layout.addStretch()

    def set_mode(self, mode: str) -> None:
        """Update mode indicator text and colour."""
        styles: dict[str, str] = {
            "OBSERVER": "color: #aaa; font-weight: 700; font-size: 14px; padding: 2px 8px;",
            "MANUAL": "color: #f0ad4e; font-weight: 700; font-size: 14px; padding: 2px 8px; background: #332200;",
            "TRAINING": "color: #5bc0de; font-weight: 700; font-size: 14px; padding: 2px 8px; background: #002233;",
        }
        self._mode_label.setText(f"[ {mode} ]")
        self._mode_label.setStyleSheet(
            styles.get(mode, styles["OBSERVER"])
        )

    def set_metrics_text(self, text: str) -> None:
        """Update compact telemetry details rendered beside mode."""
        self._metrics_label.setText(text)
