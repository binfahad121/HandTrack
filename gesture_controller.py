"""
Gesture classification and recognition module for HandTrack.
Analyzes 21 3D hand landmarks using scale-invariant Euclidean geometry,
joint angle ratios, and hysteresis state filtering.
"""

from collections import deque
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple


class GestureType(Enum):
    """Supported hand gestures."""
    OPEN_PALM = "Open Palm"
    FIST = "Fist"
    INDEX_POINTING = "Index Pointing"
    TWO_FINGERS = "Two Fingers (V/Peace)"
    THUMB_INDEX_PINCH = "Thumb-Index Pinch"
    UNKNOWN = "Unknown / Adjusting"


def euclidean_distance_3d(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> float:
    """Computes Euclidean distance between two 3D points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def euclidean_distance_2d(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Computes Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class GestureClassifier:
    """
    Geometric hand gesture classifier.
    Calculates finger extension states relative to palm scale (wrist-to-MCP distance)
    to achieve distance and scale invariance.
    """

    def __init__(self, history_size: int = 5, pinch_threshold: float = 0.055):
        self.history_size = history_size
        self.pinch_threshold = pinch_threshold
        # Rolling buffer for gesture stabilization
        self._history: deque = deque(maxlen=history_size)
        self._current_gesture = GestureType.UNKNOWN

    def reset(self) -> None:
        """Clear gesture history buffer."""
        self._history.clear()
        self._current_gesture = GestureType.UNKNOWN

    def get_finger_states(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> Dict[str, bool]:
        """
        Determines whether each finger (thumb, index, middle, ring, pinky) is extended.
        Uses distance from wrist (landmark 0) compared to intermediate joints (PIP/MCP).
        """
        if len(landmarks) < 21:
            return {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

        wrist = landmarks[0]
        # Palm scale reference: wrist (0) to middle MCP (9)
        palm_scale = max(0.001, euclidean_distance_3d(wrist, landmarks[9]))

        # Helper to check if finger is extended
        def is_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            d_tip = euclidean_distance_3d(tip, wrist)
            d_pip = euclidean_distance_3d(pip, wrist)
            d_mcp = euclidean_distance_3d(mcp, wrist)
            # Tip must be substantially farther from wrist than PIP and MCP
            return (d_tip > d_pip * 1.12) and (d_tip > d_mcp * 1.30)

        index_ext = is_extended(8, 6, 5)
        middle_ext = is_extended(12, 10, 9)
        ring_ext = is_extended(16, 14, 13)
        pinky_ext = is_extended(20, 18, 17)

        # Thumb extension: check distance from thumb tip (4) to pinky MCP (17)
        # When thumb is tucked into palm, it is close to 17. When open, it reaches far.
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]
        thumb_reach = euclidean_distance_3d(thumb_tip, pinky_mcp) / palm_scale
        thumb_ip_reach = euclidean_distance_3d(thumb_ip, pinky_mcp) / palm_scale
        thumb_ext = (thumb_reach > 1.25) or (thumb_reach > thumb_ip_reach * 1.15)

        return {
            "thumb": thumb_ext,
            "index": index_ext,
            "middle": middle_ext,
            "ring": ring_ext,
            "pinky": pinky_ext,
        }

    def compute_pinch_distance(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> float:
        """
        Computes normalized distance between thumb tip (4) and index tip (8).
        Normalized relative to palm scale (wrist to middle MCP).
        """
        if len(landmarks) < 21:
            return 1.0

        wrist = landmarks[0]
        palm_scale = max(0.001, euclidean_distance_3d(wrist, landmarks[9]))
        pinch_dist = euclidean_distance_3d(landmarks[4], landmarks[8])
        # Return distance normalized by palm scale
        return pinch_dist / palm_scale

    def classify_raw(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> GestureType:
        """
        Performs instant classification of hand landmarks without temporal filtering.
        """
        if not landmarks or len(landmarks) < 21:
            return GestureType.UNKNOWN

        norm_pinch = self.compute_pinch_distance(landmarks)
        finger_states = self.get_finger_states(landmarks)

        # Threshold scaled: standard pinch is < 0.35 of palm scale
        is_pinching = norm_pinch < 0.35

        num_extended = sum(finger_states[f] for f in ["index", "middle", "ring", "pinky"])

        # 1. Fist: all 4 non-thumb fingers are curled inward and thumb is tucked
        if num_extended == 0 and not finger_states["thumb"]:
            return GestureType.FIST

        # 2. Check Thumb-Index Pinch (when not in a closed fist)
        if is_pinching:
            return GestureType.THUMB_INDEX_PINCH

        # 3. Open Palm: all 4 non-thumb fingers extended, and thumb usually extended
        if num_extended == 4 and finger_states["thumb"]:
            return GestureType.OPEN_PALM

        # 4. Index Pointing: index extended, middle, ring, pinky curled
        if finger_states["index"] and not finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return GestureType.INDEX_POINTING

        # 5. Two Fingers (V / Peace): index & middle extended, ring & pinky curled
        if finger_states["index"] and finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return GestureType.TWO_FINGERS

        # 6. Fallback Open Palm if 4 fingers extended even if thumb is slightly tucked
        if num_extended == 4:
            return GestureType.OPEN_PALM

        return GestureType.UNKNOWN

    def classify(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> GestureType:
        """
        Classifies gestures with temporal majority voting to eliminate frame jitter.
        """
        raw_gesture = self.classify_raw(landmarks)
        self._history.append(raw_gesture)

        # Count frequencies in recent history
        counts: Dict[GestureType, int] = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1

        # Most frequent gesture in window
        majority_gesture = max(counts.items(), key=lambda item: item[1])[0]

        # Require at least 60% consensus to switch away from current stable gesture
        required_votes = max(1, int(len(self._history) * 0.6))
        if counts[majority_gesture] >= required_votes:
            self._current_gesture = majority_gesture

        return self._current_gesture
