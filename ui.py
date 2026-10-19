"""
UI rendering module for HandTrack.
Draws a futuristic cybernetic HUD overlay using OpenCV primitives:
rounded header bars, telemetry side-panels, status badges, and help modals.
"""

from collections import deque
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from config import AppMode, HandTrackConfig, PALETTE_COLORS
from gesture_controller import GestureType


def draw_rounded_rectangle(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
) -> None:
    """
    Renders a rectangle with rounded corners using OpenCV primitives.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return

    r = min(radius, w // 2, h // 2)

    if thickness == -1:
        # Filled rounded rectangle
        # Center rectangles
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        # 4 Corner circles
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        # Outlined rounded rectangle
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


class UIRenderer:
    """
    Renders futuristic heads-up display overlays on video frames.
    Includes FPS tracking, toast messages, mode telemetry, and help cards.
    """

    def __init__(self, config: HandTrackConfig):
        self.config = config
        self._fps_history = deque(maxlen=30)
        self._last_time = time.monotonic()
        self.current_fps = 0.0

        # Notification toasts: list of (message, expiration_timestamp)
        self._toast_message: Optional[str] = None
        self._toast_expires: float = 0.0

    def update_fps(self) -> float:
        """Calculates smoothed frame rate."""
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
            self.current_fps = sum(self._fps_history) / len(self._fps_history)
        return self.current_fps

    def show_toast(self, message: str, duration_sec: float = 2.5) -> None:
        """Triggers a transient status toast notification."""
        self._toast_message = message
        self._toast_expires = time.monotonic() + duration_sec

    def render_hud(
        self,
        frame: np.ndarray,
        mode: AppMode,
        num_hands: int,
        gesture: GestureType,
        mouse_info: Tuple[bool, str, float],
        canvas_info: Tuple[str, Tuple[int, int, int], int, bool],
        finger_states: Optional[Dict[str, bool]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Draws the complete HUD overlay on the provided frame in place.
        """
        self.update_fps()
        h, w, _ = frame.shape

        # 1. Top Header Banner
        self._draw_header(frame, mode, w)

        # 2. Left Telemetry Panel
        self._draw_telemetry_panel(
            frame,
            mode=mode,
            num_hands=num_hands,
            gesture=gesture,
            mouse_info=mouse_info,
            canvas_info=canvas_info,
            finger_states=finger_states,
            telemetry=telemetry,
        )

        # 3. Mode-specific guides (e.g. mouse interaction boundary)
        if mode == AppMode.VIRTUAL_MOUSE:
            self._draw_mouse_boundary(frame, w, h)

        # 4. Bottom Controls Shortcut Bar
        self._draw_bottom_bar(frame, w, h)

        # 5. Toast Notification (if active)
        self._draw_toast(frame, w, h)

        # 6. Full Help Modal (if toggled)
        if self.config.show_help:
            self._draw_help_modal(frame, w, h)

    def _draw_header(self, frame: np.ndarray, mode: AppMode, width: int) -> None:
        """Draws top title bar, mode badge, and FPS badge."""
        # Header glass panel
        header_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, header_h), (width, header_h), (60, 60, 75), 1)

        # Title: HANDTRACK with glowing cyan dot
        cv2.circle(frame, (25, 28), 6, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (25, 28), 9, (255, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "HANDTRACK",
            (42, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mode Badge Pill (Center)
        badge_text = mode.label
        badge_color = mode.badge_color
        center_x = width // 2
        pill_w = 260
        x1 = center_x - pill_w // 2
        x2 = center_x + pill_w // 2
        draw_rounded_rectangle(frame, (x1, 10), (x2, 46), (35, 35, 45), radius=8, thickness=-1)
        draw_rounded_rectangle(frame, (x1, 10), (x2, 46), badge_color, radius=8, thickness=2)
        cv2.putText(
            frame,
            badge_text,
            (x1 + 18, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # FPS & Cam Status (Right)
        fps_text = f"FPS: {int(self.current_fps):02d}"
        cv2.putText(
            frame,
            fps_text,
            (width - 150, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 255, 120),
            1,
            cv2.LINE_AA,
        )
        # Small green online dot
        cv2.circle(frame, (width - 40, 28), 5, (50, 255, 50), -1, cv2.LINE_AA)

    def _draw_telemetry_panel(
        self,
        frame: np.ndarray,
        mode: AppMode,
        num_hands: int,
        gesture: GestureType,
        mouse_info: Tuple[bool, str, float],
        canvas_info: Tuple[str, Tuple[int, int, int], int, bool],
        finger_states: Optional[Dict[str, bool]],
        telemetry: Optional[Dict[str, Any]],
    ) -> None:
        """Renders information cards on the left side of the frame."""
        panel_x = 20
        panel_y = 70
        panel_w = 270
        panel_h = 320

        # Semi-transparent backing panel
        overlay = frame.copy()
        draw_rounded_rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (22, 22, 28), 12, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        draw_rounded_rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (65, 65, 80), 12, 1)

        # Telemetry Card Header
        cv2.putText(frame, "TELEMETRY & SENSORS", (panel_x + 16, panel_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (panel_x + 16, panel_y + 35), (panel_x + panel_w - 16, panel_y + 35), (55, 55, 70), 1)

        cur_y = panel_y + 60

        # Detected Hands count
        hands_color = (50, 255, 50) if num_hands > 0 else (120, 120, 130)
        cv2.putText(frame, "Detected Hands:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{num_hands} ACTIVE", (panel_x + 155, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hands_color, 1, cv2.LINE_AA)
        cur_y += 30

        # Recognized Gesture Pill
        cv2.putText(frame, "Gesture:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
        cur_y += 20
        # Pill container
        g_name = gesture.value if num_hands > 0 else "No Hand Detected"
        g_color = (255, 200, 0) if num_hands > 0 else (100, 100, 110)
        draw_rounded_rectangle(frame, (panel_x + 16, cur_y - 14), (panel_x + panel_w - 16, cur_y + 14), (38, 38, 48), 6, -1)
        draw_rounded_rectangle(frame, (panel_x + 16, cur_y - 14), (panel_x + panel_w - 16, cur_y + 14), g_color, 6, 1)
        cv2.putText(frame, g_name, (panel_x + 24, cur_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cur_y += 40

        # Mode Specific Telemetry Rows
        if mode == AppMode.VIRTUAL_MOUSE:
            m_enabled, m_action, m_sens = mouse_info
            # Mouse status with armed safety indicator
            cv2.putText(frame, "Mouse Control:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
            status_text = "ARMED [ON]" if m_enabled else "OFF (Press M)"
            status_color = (50, 255, 50) if m_enabled else (120, 120, 130)
            cv2.putText(frame, status_text, (panel_x + 135, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
            cur_y += 28

            cv2.putText(frame, f"Sensitivity: {m_sens:.1f}x", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
            cur_y += 26
            cv2.putText(frame, f"Action: {m_action}", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

        elif mode == AppMode.AIR_DRAW:
            color_name, bgr_color, stroke_count, is_eraser = canvas_info
            cv2.putText(frame, "Active Tool:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
            tool_name = "ERASER (E)" if is_eraser else color_name
            tool_col = (180, 180, 255) if is_eraser else bgr_color
            cv2.putText(frame, tool_name, (panel_x + 115, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tool_col, 1, cv2.LINE_AA)
            cur_y += 28

            # Color preview circle
            cv2.putText(frame, "Palette Swatch:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
            cv2.circle(frame, (panel_x + 160, cur_y - 5), 8, bgr_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (panel_x + 160, cur_y - 5), 10, (255, 255, 255), 1, cv2.LINE_AA)
            cur_y += 28
            cv2.putText(frame, f"Drawn Strokes: {stroke_count}", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        elif mode == AppMode.GESTURE_RECOGNITION:
            if finger_states:
                cv2.putText(frame, "Finger Extension State:", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA)
                cur_y += 22
                for f_name, is_ext in finger_states.items():
                    f_col = (50, 255, 50) if is_ext else (100, 100, 115)
                    state_lbl = "EXT" if is_ext else "CURL"
                    cv2.putText(frame, f"{f_name.capitalize()}: {state_lbl}", (panel_x + 20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, f_col, 1, cv2.LINE_AA)
                    cur_y += 18
            else:
                cv2.putText(frame, "Show hand to classify", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 160), 1, cv2.LINE_AA)

        elif mode == AppMode.LIVE_TRACKING:
            if telemetry and "index_tip_norm" in telemetry:
                ix, iy, iz = telemetry["index_tip_norm"]
                cv2.putText(frame, "Index Tip (Norm):", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA)
                cur_y += 22
                cv2.putText(frame, f"X: {ix:.3f}  Y: {iy:.3f}", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
                cur_y += 20
                cv2.putText(frame, f"Depth (Z): {iz:.3f}", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Awaiting hand presence...", (panel_x + 16, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 140), 1, cv2.LINE_AA)

    def _draw_mouse_boundary(self, frame: np.ndarray, width: int, height: int) -> None:
        """Visualizes the active interaction zone for cursor mapping."""
        scale = self.config.active_box_scale
        bw = int(width * scale)
        bh = int(height * scale)
        x1 = (width - bw) // 2
        y1 = (height - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh

        # Subtle dashed-look cybernetic box
        box_color = (0, 180, 255) if self.config.mouse_enabled else (70, 70, 85)
        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), box_color, radius=12, thickness=1)
        cv2.putText(
            frame,
            "VIRTUAL SCREEN BOUNDARY",
            (x1 + 14, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            box_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_bottom_bar(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draws persistent keyboard shortcuts guide along bottom."""
        bar_h = 36
        y1 = height - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (width, height), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, y1), (width, y1), (60, 60, 75), 1)

        shortcuts = (
            "[1-4] Modes | [M] Mouse | [C] Clear | [Z] Undo | [E] Eraser | "
            "[S] Save | [[ / ]] Sens | [R] Reset | [H] Help | [Q/ESC] Quit"
        )
        cv2.putText(
            frame,
            shortcuts,
            (24, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )

    def _draw_toast(self, frame: np.ndarray, width: int, height: int) -> None:
        """Displays transient banner notification."""
        if not self._toast_message or time.monotonic() > self._toast_expires:
            return

        toast_text = self._toast_message
        (tw, th), _ = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 20
        pw = tw + pad * 2
        ph = th + 24
        x1 = (width - pw) // 2
        y1 = height - 90
        x2 = x1 + pw
        y2 = y1 + ph

        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), (25, 25, 32), radius=8, thickness=-1)
        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), radius=8, thickness=2)
        cv2.putText(
            frame,
            toast_text,
            (x1 + pad, y1 + th + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_help_modal(self, frame: np.ndarray, width: int, height: int) -> None:
        """Renders comprehensive help dialog card over the screen."""
        card_w = min(740, width - 60)
        card_h = min(520, height - 80)
        x1 = (width - card_w) // 2
        y1 = (height - card_h) // 2
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Dim entire background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Draw card panel
        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), (28, 28, 36), radius=16, thickness=-1)
        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), radius=16, thickness=2)

        # Modal Title
        cv2.putText(
            frame,
            "HANDTRACK — KEYBOARD SHORTCUTS & USER GUIDE",
            (x1 + 30, y1 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(frame, (x1 + 30, y1 + 60), (x2 - 30, y1 + 60), (70, 70, 85), 1)

        help_sections = [
            ("OPERATING MODES", [
                ("1", "Live Tracking: 21 landmarks skeleton and normalized 3D coordinates."),
                ("2", "Virtual Mouse: Control OS cursor via index fingertip with smoothing."),
                ("3", "AirDraw: Draw on digital canvas using index pointing gesture."),
                ("4", "Gesture Recognition: Detect Open Palm, Fist, Pointing, Peace, Pinch."),
            ]),
            ("INTERACTIVE ACTIONS", [
                ("M", "Toggle Virtual Mouse control ON / OFF (Safety switch)."),
                ("Pinch", "Thumb + Index pinch triggers a single Left Click or Click-and-Drag."),
                ("C / Z", "Clear Canvas (C) or Undo most recent stroke (Z)."),
                ("E", "Toggle Eraser mode in AirDraw."),
                ("S", "Save current canvas artwork as high-res PNG to disk."),
                ("[ / ]", "Decrease / Increase cursor motion sensitivity."),
                ("R / H", "Reset settings to defaults (R) | Toggle this Help card (H)."),
                ("Q / ESC", "Cleanly release webcam and exit application."),
            ]),
        ]

        row_y = y1 + 95
        for section_title, items in help_sections:
            cv2.putText(frame, section_title, (x1 + 30, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 1, cv2.LINE_AA)
            row_y += 26
            for key, desc in items:
                cv2.putText(frame, f"[{key}]", (x1 + 40, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 120), 1, cv2.LINE_AA)
                cv2.putText(frame, desc, (x1 + 130, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 240), 1, cv2.LINE_AA)
                row_y += 24
            row_y += 12

        cv2.putText(
            frame,
            "Press [H] or any key to close this help window",
            (x1 + 30, y2 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 175),
            1,
            cv2.LINE_AA,
        )
