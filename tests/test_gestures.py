"""
Unit tests for gesture recognition using synthetic hand landmarks.
Verifies Open Palm, Fist, Index Pointing, Two Fingers, and Pinch.
"""

from typing import List, Tuple
from gesture_controller import GestureClassifier, GestureType


def build_synthetic_hand(
    thumb_extended: bool = True,
    index_extended: bool = True,
    middle_extended: bool = True,
    ring_extended: bool = True,
    pinky_extended: bool = True,
    is_pinch: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Constructs a synthetic 21-landmark hand model with configurable finger extensions.
    Wrist is positioned at (0.50, 0.80, 0.0).
    """
    wrist = (0.50, 0.80, 0.0)
    landmarks = [wrist] * 21

    # Base palm MCPs
    landmarks[0] = wrist
    landmarks[5] = (0.45, 0.60, 0.0)   # Index MCP
    landmarks[9] = (0.50, 0.58, 0.0)   # Middle MCP
    landmarks[13] = (0.55, 0.60, 0.0)  # Ring MCP
    landmarks[17] = (0.60, 0.65, 0.0)  # Pinky MCP

    # Helper for four fingers
    def set_finger(base_idx: int, x: float, extended: bool) -> None:
        mcp_idx = base_idx
        pip_idx = base_idx + 1
        dip_idx = base_idx + 2
        tip_idx = base_idx + 3
        if extended:
            landmarks[pip_idx] = (x, 0.48, 0.0)
            landmarks[dip_idx] = (x, 0.38, 0.0)
            landmarks[tip_idx] = (x, 0.28, 0.0)
        else:
            # Curled into palm
            landmarks[pip_idx] = (x, 0.52, 0.0)
            landmarks[dip_idx] = (x, 0.60, 0.0)
            landmarks[tip_idx] = (x, 0.64, 0.0)

    # Set 4 fingers
    set_finger(5, 0.45, index_extended)
    set_finger(9, 0.50, middle_extended)
    set_finger(13, 0.55, ring_extended)
    set_finger(17, 0.60, pinky_extended)

    # Set thumb
    if thumb_extended:
        landmarks[1] = (0.44, 0.74, 0.0)
        landmarks[2] = (0.38, 0.68, 0.0)
        landmarks[3] = (0.32, 0.62, 0.0)
        landmarks[4] = (0.26, 0.56, 0.0)  # Extended out to side
    else:
        # Tucked against palm near pinky MCP
        landmarks[1] = (0.47, 0.75, 0.0)
        landmarks[2] = (0.48, 0.70, 0.0)
        landmarks[3] = (0.50, 0.68, 0.0)
        landmarks[4] = (0.52, 0.66, 0.0)

    # Override for pinch
    if is_pinch:
        pinch_pt = (0.42, 0.45, 0.0)
        landmarks[4] = (pinch_pt[0] - 0.01, pinch_pt[1], 0.0)
        landmarks[8] = (pinch_pt[0] + 0.01, pinch_pt[1], 0.0)

    return landmarks


def test_classify_open_palm():
    """All 5 fingers extended should classify as OPEN_PALM."""
    classifier = GestureClassifier()
    hand = build_synthetic_hand(True, True, True, True, True)
    assert classifier.classify_raw(hand) == GestureType.OPEN_PALM


def test_classify_fist():
    """All fingers curled inward should classify as FIST."""
    classifier = GestureClassifier()
    hand = build_synthetic_hand(False, False, False, False, False)
    assert classifier.classify_raw(hand) == GestureType.FIST


def test_classify_index_pointing():
    """Index extended, all others curled should classify as INDEX_POINTING."""
    classifier = GestureClassifier()
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=True,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
    )
    assert classifier.classify_raw(hand) == GestureType.INDEX_POINTING


def test_classify_two_fingers():
    """Index and middle extended, others curled should classify as TWO_FINGERS."""
    classifier = GestureClassifier()
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=True,
        middle_extended=True,
        ring_extended=False,
        pinky_extended=False,
    )
    assert classifier.classify_raw(hand) == GestureType.TWO_FINGERS


def test_classify_pinch():
    """Thumb and index fingertips brought together should classify as THUMB_INDEX_PINCH."""
    classifier = GestureClassifier()
    hand = build_synthetic_hand(is_pinch=True)
    assert classifier.classify_raw(hand) == GestureType.THUMB_INDEX_PINCH


def test_classify_missing_or_empty_landmarks():
    """Empty or partial landmarks should yield UNKNOWN without crashing."""
    classifier = GestureClassifier()
    assert classifier.classify_raw([]) == GestureType.UNKNOWN
    assert classifier.classify_raw([(0.0, 0.0, 0.0)] * 10) == GestureType.UNKNOWN


def test_temporal_smoothing():
    """Classifier should smooth brief single-frame noise."""
    classifier = GestureClassifier(history_size=5)
    fist = build_synthetic_hand(False, False, False, False, False)
    palm = build_synthetic_hand(True, True, True, True, True)

    # Establish stable fist
    for _ in range(5):
        classifier.classify(fist)
    assert classifier.classify(fist) == GestureType.FIST

    # Single noisy frame should not immediately switch state
    classifier.classify(palm)
    assert classifier._current_gesture == GestureType.FIST
