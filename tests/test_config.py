"""
Unit tests for HandTrack configuration and AppMode definitions.
"""

from config import AppMode, HandTrackConfig, PALETTE_COLORS, ERASER_COLOR


def test_app_mode_properties():
    """Verify AppMode labels and badge colors."""
    assert AppMode.LIVE_TRACKING.label == "MODE 1: LIVE TRACKING"
    assert AppMode.VIRTUAL_MOUSE.label == "MODE 2: VIRTUAL MOUSE"
    assert AppMode.AIR_DRAW.label == "MODE 3: AIR DRAW"
    assert AppMode.GESTURE_RECOGNITION.label == "MODE 4: GESTURE RECOGNITION"

    for mode in AppMode:
        color = mode.badge_color
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_palette_colors():
    """Verify predefined color palette and eraser definitions."""
    assert len(PALETTE_COLORS) >= 5
    for name, color in PALETTE_COLORS:
        assert isinstance(name, str)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    assert ERASER_COLOR == (0, 0, 0)


def test_config_defaults_and_reset():
    """Verify default config parameters and reset method."""
    cfg = HandTrackConfig()
    assert cfg.cursor_sensitivity == 1.3
    assert cfg.smoothing_factor == 0.65
    assert cfg.pinch_threshold == 0.05
    assert cfg.mouse_enabled is False
    assert cfg.show_help is False

    # Modify parameters
    cfg.cursor_sensitivity = 2.5
    cfg.smoothing_factor = 0.9
    cfg.pinch_threshold = 0.08
    cfg.mouse_enabled = True
    cfg.show_help = True

    # Reset
    cfg.reset()
    assert cfg.cursor_sensitivity == 1.3
    assert cfg.smoothing_factor == 0.65
    assert cfg.pinch_threshold == 0.05
    assert cfg.mouse_enabled is False
    assert cfg.show_help is False
