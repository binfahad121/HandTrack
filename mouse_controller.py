"""
Virtual Mouse Controller module for HandTrack.
Provides smooth cursor movement, edge-triggered clicking, and drag gestures
with fail-safe protections and active interaction box mapping.
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("HandTrack.Mouse")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


class VirtualMouseController:
    """
    Translates hand tracking landmarks into operating system mouse events.
    Includes Exponential Moving Average (EMA) smoothing, edge-triggered clicks,
    and safe state clamping.
    """

    def __init__(
        self,
        sensitivity: float = 1.3,
        smoothing_factor: float = 0.65,
        pinch_threshold: float = 0.05,
        active_box_scale: float = 0.70,
    ):
        self.sensitivity = max(0.2, min(4.0, sensitivity))
        self.smoothing_factor = max(0.05, min(0.95, smoothing_factor))
        self.pinch_threshold = pinch_threshold
        self.active_box_scale = active_box_scale

        self.is_enabled = False
        self.is_dragging = False
        self._prev_pinched = False
        self._pinch_frame_count = 0
        self._drag_activation_frames = 12

        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Screen dimensions
        self.screen_width = 1920
        self.screen_height = 1080
        self._pyautogui = None
        self._init_pyautogui()

    def _init_pyautogui(self) -> None:
        """Initialize PyAutoGUI safely."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.001
            size = pyautogui.size()
            self.screen_width, self.screen_height = size.width, size.height
            self._pyautogui = pyautogui
            logger.info("PyAutoGUI initialized with screen resolution %dx%d", self.screen_width, self.screen_height)
        except Exception as exc:
            logger.warning("PyAutoGUI screen detection unavailable (headless or permissions missing): %s", exc)
            self._pyautogui = None

    def toggle_enable(self) -> bool:
        """Toggle mouse control on or off."""
        self.is_enabled = not self.is_enabled
        if not self.is_enabled:
            self.release_all()
        logger.info("Virtual Mouse %s", "ENABLED [ARMED]" if self.is_enabled else "DISABLED")
        return self.is_enabled

    def adjust_sensitivity(self, delta: float) -> float:
        """Adjust cursor sensitivity."""
        self.sensitivity = round(max(0.2, min(3.5, self.sensitivity + delta)), 2)
        logger.info("Cursor sensitivity adjusted to %.2f", self.sensitivity)
        return self.sensitivity

    def release_all(self) -> None:
        """Safety release of any pressed mouse buttons and state reset."""
        if self.is_dragging and self._pyautogui:
            try:
                self._pyautogui.mouseUp()
            except Exception:
                pass
        self.is_dragging = False
        self._prev_pinched = False
        self._pinch_frame_count = 0
        self._smooth_x = None
        self._smooth_y = None

    def map_to_screen(
        self,
        norm_x: float,
        norm_y: float,
    ) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates [0.0, 1.0] to screen pixel coordinates
        using an inner active interaction box to ensure comfortable physical reach.
        """
        # Active box margins
        margin_x = (1.0 - self.active_box_scale) / 2.0
        margin_y = (1.0 - self.active_box_scale) / 2.0

        # Remap within active box
        rel_x = (norm_x - margin_x) / self.active_box_scale
        rel_y = (norm_y - margin_y) / self.active_box_scale

        # Apply sensitivity centered around middle (0.5)
        centered_x = (rel_x - 0.5) * self.sensitivity + 0.5
        centered_y = (rel_y - 0.5) * self.sensitivity + 0.5

        # Clamp to [0.0, 1.0]
        clamped_x = max(0.0, min(1.0, centered_x))
        clamped_y = max(0.0, min(1.0, centered_y))

        target_screen_x = int(clamped_x * (self.screen_width - 1))
        target_screen_y = int(clamped_y * (self.screen_height - 1))

        return target_screen_x, target_screen_y

    def smooth_coordinates(self, target_x: int, target_y: int) -> Tuple[int, int]:
        """
        Applies Exponential Moving Average (EMA) to prevent cursor tremble.
        """
        if self._smooth_x is None or self._smooth_y is None:
            self._smooth_x = float(target_x)
            self._smooth_y = float(target_y)
        else:
            alpha = 1.0 - self.smoothing_factor
            self._smooth_x = alpha * target_x + (1.0 - alpha) * self._smooth_x
            self._smooth_y = alpha * target_y + (1.0 - alpha) * self._smooth_y

        return int(round(self._smooth_x)), int(round(self._smooth_y))

    def update(
        self,
        index_tip: Tuple[float, float, float],
        thumb_tip: Tuple[float, float, float],
        wrist: Tuple[float, float, float],
        middle_mcp: Tuple[float, float, float],
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Processes index tip and thumb tip coordinates to update mouse state.
        Returns:
            (screen_cursor_pos, action_feedback_message)
        """
        # Calculate scale-invariant pinch distance
        palm_scale = max(0.001, np.linalg.norm(np.array(wrist) - np.array(middle_mcp)))
        pinch_dist = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip)) / palm_scale
        is_pinching = pinch_dist < (self.pinch_threshold / 0.15)  # Scale normalized

        # Calculate target cursor coordinates
        raw_x, raw_y = self.map_to_screen(index_tip[0], index_tip[1])
        cur_x, cur_y = self.smooth_coordinates(raw_x, raw_y)

        action_msg = "Idle"

        if not self.is_enabled:
            return (cur_x, cur_y), "Mouse Disabled (Press 'M' to enable)"

        if self._pyautogui is None:
            return (cur_x, cur_y), "PyAutoGUI Unavailable"

        try:
            # Move cursor
            self._pyautogui.moveTo(cur_x, cur_y)

            # Edge-triggered Click & Drag Logic
            if is_pinching:
                self._pinch_frame_count += 1
                if not self._prev_pinched:
                    # Pinch onset -> single click
                    self._pyautogui.click()
                    action_msg = "Left Click"
                elif self._pinch_frame_count >= self._drag_activation_frames and not self.is_dragging:
                    # Sustained pinch -> start drag
                    self._pyautogui.mouseDown()
                    self.is_dragging = True
                    action_msg = "Dragging Started"
                elif self.is_dragging:
                    action_msg = "Dragging..."
            else:
                if self.is_dragging:
                    # Released pinch -> finish drag
                    self._pyautogui.mouseUp()
                    self.is_dragging = False
                    action_msg = "Drop / Released"
                self._pinch_frame_count = 0
                action_msg = "Hovering"

            self._prev_pinched = is_pinching

        except Exception as exc:
            logger.warning("Mouse action caught exception: %s", exc)
            self.release_all()
            action_msg = "Failsafe Triggered"

        return (cur_x, cur_y), action_msg
