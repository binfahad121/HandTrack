"""
Unit tests for VirtualMouseController.
Verifies screen mapping, EMA smoothing, sensitivity bounds, and safety states.
"""

from mouse_controller import VirtualMouseController


def test_mouse_initial_safety_state():
    """Verify virtual mouse starts disabled by default."""
    ctrl = VirtualMouseController()
    assert ctrl.is_enabled is False

    # Toggling enables it
    enabled = ctrl.toggle_enable()
    assert enabled is True
    assert ctrl.is_enabled is True

    # Toggling again disables it
    enabled = ctrl.toggle_enable()
    assert enabled is False
    assert ctrl.is_enabled is False


def test_screen_coordinate_mapping():
    """Verify coordinate mapping stays within screen boundaries."""
    ctrl = VirtualMouseController(active_box_scale=0.7)
    ctrl.screen_width = 1920
    ctrl.screen_height = 1080

    # Center of camera frame maps to center of screen
    sx, sy = ctrl.map_to_screen(0.5, 0.5)
    assert abs(sx - 1920 // 2) <= 10
    assert abs(sy - 1080 // 2) <= 10

    # Extremes are safely clamped to [0, screen_dimension - 1]
    sx_min, sy_min = ctrl.map_to_screen(0.0, 0.0)
    assert sx_min >= 0 and sy_min >= 0

    sx_max, sy_max = ctrl.map_to_screen(1.0, 1.0)
    assert sx_max <= 1919 and sy_max <= 1079


def test_coordinate_smoothing():
    """Verify Exponential Moving Average (EMA) dampens jumps."""
    ctrl = VirtualMouseController(smoothing_factor=0.5)

    # Initial position
    s1_x, s1_y = ctrl.smooth_coordinates(100, 100)
    assert (s1_x, s1_y) == (100, 100)

    # Subsequent large jump should be smoothed towards target
    s2_x, s2_y = ctrl.smooth_coordinates(200, 200)
    # alpha = 0.5, so 0.5 * 200 + 0.5 * 100 = 150
    assert s2_x == 150
    assert s2_y == 150


def test_sensitivity_adjustment():
    """Verify sensitivity clamping bounds."""
    ctrl = VirtualMouseController(sensitivity=1.0)
    assert ctrl.sensitivity == 1.0

    ctrl.adjust_sensitivity(0.5)
    assert ctrl.sensitivity == 1.5

    # Should clamp at upper limit 3.5
    ctrl.adjust_sensitivity(5.0)
    assert ctrl.sensitivity == 3.5

    # Should clamp at lower limit 0.2
    ctrl.adjust_sensitivity(-10.0)
    assert ctrl.sensitivity == 0.2
