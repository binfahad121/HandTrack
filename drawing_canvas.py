"""
Air Drawing Canvas module for HandTrack.
Manages multi-color stroke paths, eraser mode, undo stack history,
compositing onto camera frames, and saving PNG artworks.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import PALETTE_COLORS, ERASER_COLOR

logger = logging.getLogger("HandTrack.Canvas")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class DrawingStroke:
    """Represents a continuous stroke on the canvas."""
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 255, 0)
    thickness: int = 5
    is_eraser: bool = False


class AirDrawingCanvas:
    """
    Interactive drawing canvas overlaid on webcam frames.
    Maintains stroke history for undo, selectable palettes, and PNG export.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        brush_thickness: int = 5,
        eraser_thickness: int = 35,
        output_dir: str = "saved_drawings",
    ):
        self.width = width
        self.height = height
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness
        self.output_dir = output_dir

        self.palette = PALETTE_COLORS
        self.active_color_index = 0
        self.is_eraser_active = False

        # Canvas surface: 3-channel uint8 image
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Stroke history for undo stack
        self.strokes: List[DrawingStroke] = []
        self._current_stroke: Optional[DrawingStroke] = None
        self._prev_point: Optional[Tuple[int, int]] = None

        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def current_color(self) -> Tuple[int, int, int]:
        """Returns the currently active drawing color (BGR)."""
        if self.is_eraser_active:
            return ERASER_COLOR
        return self.palette[self.active_color_index][1]

    @property
    def current_color_name(self) -> str:
        """Returns the display name of the current color or Eraser."""
        if self.is_eraser_active:
            return "Eraser Mode"
        return self.palette[self.active_color_index][0]

    def cycle_color(self) -> str:
        """Cycles to the next palette color and disables eraser."""
        self.is_eraser_active = False
        self.active_color_index = (self.active_color_index + 1) % len(self.palette)
        logger.info("Selected color: %s", self.current_color_name)
        return self.current_color_name

    def set_color_index(self, index: int) -> None:
        """Select a color by index (0 to len-1)."""
        self.is_eraser_active = False
        self.active_color_index = index % len(self.palette)

    def toggle_eraser(self) -> bool:
        """Toggle eraser on or off."""
        self.is_eraser_active = not self.is_eraser_active
        logger.info("Eraser %s", "ENABLED" if self.is_eraser_active else "DISABLED")
        return self.is_eraser_active

    def start_stroke(self, start_pt: Tuple[int, int]) -> None:
        """Starts a new continuous drawing stroke."""
        stroke = DrawingStroke(
            points=[start_pt],
            color=self.current_color,
            thickness=self.eraser_thickness if self.is_eraser_active else self.brush_thickness,
            is_eraser=self.is_eraser_active,
        )
        self._current_stroke = stroke
        self.strokes.append(stroke)
        self._prev_point = start_pt

    def add_point(self, pt: Tuple[int, int]) -> None:
        """
        Adds a point to the current stroke and renders the line segment.
        Rejects extreme jumps to avoid unwanted lines when tracking flickers.
        """
        if self._prev_point is None or self._current_stroke is None:
            self.start_stroke(pt)
            return

        # Distance threshold: if finger jumped more than 150px in 1 frame, split stroke
        dist = np.hypot(pt[0] - self._prev_point[0], pt[1] - self._prev_point[1])
        if dist > 150:
            self.end_stroke()
            self.start_stroke(pt)
            return

        # Draw segment onto canvas
        color = self._current_stroke.color
        thick = self._current_stroke.thickness
        cv2.line(self.canvas, self._prev_point, pt, color, thick, cv2.LINE_AA)

        self._current_stroke.points.append(pt)
        self._prev_point = pt

    def end_stroke(self) -> None:
        """Concludes the current stroke when finger lifts or gesture changes."""
        self._current_stroke = None
        self._prev_point = None

    def undo(self) -> bool:
        """
        Undoes the most recent stroke and redraws the canvas.
        Returns True if a stroke was removed, False if already empty.
        """
        if not self.strokes:
            logger.info("Undo requested but stroke history is empty.")
            return False

        self.strokes.pop()
        self._redraw_all_strokes()
        logger.info("Undo performed. %d strokes remaining.", len(self.strokes))
        return True

    def clear(self) -> None:
        """Clears the entire canvas and resets stroke history."""
        self.strokes.clear()
        self._current_stroke = None
        self._prev_point = None
        self.canvas[:] = 0
        logger.info("Canvas cleared.")

    def _redraw_all_strokes(self) -> None:
        """Reconstructs the canvas from the current stroke history."""
        self.canvas[:] = 0
        for stroke in self.strokes:
            pts = stroke.points
            if len(pts) < 2:
                if len(pts) == 1:
                    cv2.circle(self.canvas, pts[0], stroke.thickness // 2, stroke.color, -1, cv2.LINE_AA)
                continue
            for i in range(1, len(pts)):
                cv2.line(self.canvas, pts[i - 1], pts[i], stroke.color, stroke.thickness, cv2.LINE_AA)

    def blend_with_frame(
        self,
        frame_bgr: np.ndarray,
        pointer_pt: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Overlays drawing strokes onto the webcam frame using pixel masking.
        Optionally renders a pointer reticle at the active fingertip position.
        """
        # Create mask of drawn areas (non-black pixels)
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Blend: background where mask is black, canvas where mask is white
        bg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        blended = cv2.add(bg, fg)

        # Draw active pointer reticle
        if pointer_pt is not None:
            px, py = pointer_pt
            if self.is_eraser_active:
                cv2.circle(blended, (px, py), self.eraser_thickness // 2, (180, 180, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(blended, (px, py), self.brush_thickness + 2, self.current_color, -1, cv2.LINE_AA)
                cv2.circle(blended, (px, py), self.brush_thickness + 5, (255, 255, 255), 1, cv2.LINE_AA)

        return blended

    def save_drawing(self, background_frame: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Saves the current drawing as a PNG file.
        If background_frame is provided, saves the composite; otherwise saves drawing on black.
        Returns (success_status, saved_filepath).
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"handtrack_art_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)

            if background_frame is not None:
                img_to_save = self.blend_with_frame(background_frame)
            else:
                img_to_save = self.canvas

            cv2.imwrite(filepath, img_to_save)
            logger.info("Drawing saved successfully to: %s", filepath)
            return True, filepath
        except Exception as exc:
            logger.error("Failed to save drawing: %s", exc)
            return False, str(exc)
