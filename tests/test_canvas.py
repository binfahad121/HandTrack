"""
Unit tests for AirDrawingCanvas operations:
strokes, eraser, undo stack, clearing, blending, and PNG file export.
"""

import os
import shutil
import tempfile
import cv2
import numpy as np

from drawing_canvas import AirDrawingCanvas


def test_canvas_initialization():
    """Verify clean canvas initial state."""
    canvas = AirDrawingCanvas(width=640, height=480)
    assert canvas.canvas.shape == (480, 640, 3)
    assert np.all(canvas.canvas == 0)
    assert len(canvas.strokes) == 0
    assert not canvas.is_eraser_active


def test_drawing_and_undo():
    """Verify adding stroke points and undo behavior."""
    canvas = AirDrawingCanvas(width=640, height=480)

    # Stroke 1
    canvas.start_stroke((100, 100))
    canvas.add_point((110, 110))
    canvas.add_point((120, 120))
    canvas.end_stroke()
    assert len(canvas.strokes) == 1
    assert np.any(canvas.canvas > 0)

    # Stroke 2
    canvas.start_stroke((200, 200))
    canvas.add_point((210, 210))
    canvas.end_stroke()
    assert len(canvas.strokes) == 2

    # Undo Stroke 2
    undone = canvas.undo()
    assert undone is True
    assert len(canvas.strokes) == 1

    # Undo Stroke 1
    undone = canvas.undo()
    assert undone is True
    assert len(canvas.strokes) == 0
    assert np.all(canvas.canvas == 0)

    # Undo on empty canvas
    assert canvas.undo() is False


def test_clear_canvas():
    """Verify clearing canvas resets history and resets pixel data to black."""
    canvas = AirDrawingCanvas(width=640, height=480)
    canvas.start_stroke((50, 50))
    canvas.add_point((60, 60))
    canvas.end_stroke()
    assert len(canvas.strokes) == 1

    canvas.clear()
    assert len(canvas.strokes) == 0
    assert np.all(canvas.canvas == 0)


def test_eraser_toggle():
    """Verify toggling eraser mode."""
    canvas = AirDrawingCanvas(width=640, height=480)
    assert canvas.is_eraser_active is False
    assert "Eraser" not in canvas.current_color_name

    canvas.toggle_eraser()
    assert canvas.is_eraser_active is True
    assert "Eraser" in canvas.current_color_name
    assert canvas.current_color == (0, 0, 0)

    # Cycling color turns off eraser
    canvas.cycle_color()
    assert canvas.is_eraser_active is False


def test_save_drawing_to_disk():
    """Verify PNG export creates a valid image file."""
    temp_dir = tempfile.mkdtemp()
    try:
        canvas = AirDrawingCanvas(width=640, height=480, output_dir=temp_dir)
        canvas.start_stroke((50, 50))
        canvas.add_point((150, 150))
        canvas.end_stroke()

        success, filepath = canvas.save_drawing()
        assert success is True
        assert os.path.exists(filepath)

        # Check saved image can be loaded by OpenCV
        loaded = cv2.imread(filepath)
        assert loaded is not None
        assert loaded.shape == (480, 640, 3)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
