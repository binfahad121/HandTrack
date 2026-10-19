"""
Configuration module for HandTrack.
Defines application modes, visual styling constants, and default parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import os

class AppMode(Enum):
    """Supported operating modes for HandTrack."""
    LIVE_TRACKING = 1
    VIRTUAL_MOUSE = 2
    AIR_DRAW = 3
    GESTURE_RECOGNITION = 4

    @property
    def label(self) -> str:
        labels = {
            AppMode.LIVE_TRACKING: "MODE 1: LIVE TRACKING",
            AppMode.VIRTUAL_MOUSE: "MODE 2: VIRTUAL MOUSE",
            AppMode.AIR_DRAW: "MODE 3: AIR DRAW",
            AppMode.GESTURE_RECOGNITION: "MODE 4: GESTURE RECOGNITION",
        }
        return labels[self]

    @property
    def badge_color(self) -> Tuple[int, int, int]:
        # BGR format for OpenCV
        colors = {
            AppMode.LIVE_TRACKING: (255, 180, 0),    # Vibrant Cyan/Sky Blue
            AppMode.VIRTUAL_MOUSE: (0, 220, 255),    # Vibrant Yellow/Amber
            AppMode.AIR_DRAW: (100, 255, 100),       # Neon Green
            AppMode.GESTURE_RECOGNITION: (255, 105, 180), # Neon Pink/Purple
        }
        return colors[self]


# Predefined neon color palette for AirDraw (BGR format)
PALETTE_COLORS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Neon Cyan", (255, 255, 0)),
    ("Electric Green", (50, 255, 50)),
    ("Solar Yellow", (0, 230, 255)),
    ("Hot Pink", (180, 105, 255)),
    ("Flame Orange", (0, 140, 255)),
]

ERASER_COLOR: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class HandTrackConfig:
    """Configurable application parameters with reset capability."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps_limit: int = 60

    # MediaPipe settings
    model_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "hand_landmarker.task",
    )
    max_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Virtual Mouse settings
    mouse_enabled: bool = False
    cursor_sensitivity: float = 1.3
    smoothing_factor: float = 0.65  # Higher = smoother EMA filter
    pinch_threshold: float = 0.05   # Normalized distance threshold for pinch
    drag_hold_threshold: float = 0.04
    active_box_scale: float = 0.7   # Central interaction zone for comfortable arm reach

    # AirDraw settings
    brush_thickness: int = 5
    eraser_thickness: int = 35
    output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saved_drawings",
    )

    # UI Theme styling (BGR)
    color_bg_dark: Tuple[int, int, int] = (20, 20, 24)
    color_panel_bg: Tuple[int, int, int] = (32, 32, 40)
    color_panel_border: Tuple[int, int, int] = (70, 70, 85)
    color_accent_cyan: Tuple[int, int, int] = (255, 220, 0)
    color_accent_green: Tuple[int, int, int] = (100, 255, 120)
    color_accent_red: Tuple[int, int, int] = (60, 60, 255)
    color_accent_gray: Tuple[int, int, int] = (120, 120, 130)
    color_text_white: Tuple[int, int, int] = (245, 245, 245)
    color_text_dim: Tuple[int, int, int] = (170, 170, 180)

    # HUD view state
    show_help: bool = False
    show_landmarks: bool = True

    def reset(self) -> None:
        """Reset configurable runtime values back to factory defaults."""
        self.cursor_sensitivity = 1.3
        self.smoothing_factor = 0.65
        self.pinch_threshold = 0.05
        self.max_hands = 2
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.mouse_enabled = False
        self.show_help = False
        self.show_landmarks = True
