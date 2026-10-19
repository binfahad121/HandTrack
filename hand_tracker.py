"""
Hand Tracker module using Google MediaPipe Tasks Hand Landmarker.
Handles model downloading, initialization, video frame inference with monotonic timestamps,
and sleek skeleton rendering on OpenCV frames.
"""

import os
import time
import urllib.request
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Set up logger
logger = logging.getLogger("HandTrack.Tracker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
logger.propagate = False

# Official MediaPipe Hand Landmarker model URL
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# 21 Hand Landmark Connections
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base cross-connections
    (5, 9), (9, 13), (13, 17),
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class HandDetection:
    """Encapsulates tracking data for a single detected hand."""
    def __init__(
        self,
        handedness: str,
        score: float,
        landmarks: List[Tuple[float, float, float]],
        pixel_landmarks: List[Tuple[int, int]],
    ):
        self.handedness = handedness
        self.score = score
        self.landmarks = landmarks              # Normalized (x, y, z)
        self.pixel_landmarks = pixel_landmarks  # Pixel coordinates (x, y)


def download_model_if_missing(model_path: str) -> bool:
    """
    Downloads the official MediaPipe Hand Landmarker task file if missing or invalid.
    Verifies that the downloaded file is non-empty and accessible.
    """
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size > 1_000_000:  # Valid model is ~8-10MB
            logger.info("Found existing verified model at %s (%.2f MB)", model_path, size / (1024 * 1024))
            return True
        else:
            logger.warning("Existing model file is corrupt or incomplete (%d bytes). Re-downloading...", size)
            try:
                os.remove(model_path)
            except OSError:
                pass

    try:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    except OSError as err:
        logger.error("Could not create model directory for %s: %s", model_path, err)
        return False

    logger.info("Downloading official MediaPipe Hand Landmarker model from Google Cloud Storage...")
    logger.info("URL: %s", MODEL_URL)
    logger.info("Destination: %s", model_path)

    try:
        def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100.0)
                if int(percent) % 25 == 0:
                    logger.debug("Download progress: %.1f%%", percent)

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_report_progress)
        
        # Verify downloaded file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file was not created at {model_path}")
        
        file_size = os.path.getsize(model_path)
        if file_size < 1_000_000:
            raise ValueError(f"Downloaded model is too small ({file_size} bytes). Download may have failed.")
        
        logger.info("Successfully downloaded and verified hand_landmarker.task (%.2f MB)", file_size / (1024 * 1024))
        return True

    except Exception as exc:
        logger.error("Failed to download MediaPipe model: %s", exc)
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError:
                pass
        return False


class HandTracker:
    """
    Wrapper around MediaPipe Tasks Hand Landmarker.
    Supports video mode inference with guaranteed strictly increasing monotonic timestamps.
    """
    def __init__(
        self,
        model_path: str,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.landmarker = None
        self._last_timestamp_ms = -1
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initializes the MediaPipe Tasks HandLandmarker."""
        if not download_model_if_missing(self.model_path):
            logger.error("MediaPipe Hand Landmarker model asset is unavailable.")
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("MediaPipe HandLandmarker initialized successfully in VIDEO mode.")
            return True
        except Exception as exc:
            logger.error("Error creating MediaPipe HandLandmarker: %s", exc)
            self._is_initialized = False
            return False

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandDetection]:
        """
        Processes a BGR image frame and returns a list of detected hands.
        Computes monotonic millisecond timestamps to satisfy MediaPipe requirements.
        """
        if not self._is_initialized or self.landmarker is None:
            return []

        import mediapipe as mp

        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ensure strictly increasing timestamp in milliseconds
        curr_timestamp_ms = int(time.monotonic() * 1000)
        if curr_timestamp_ms <= self._last_timestamp_ms:
            curr_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = curr_timestamp_ms

        try:
            result = self.landmarker.detect_for_video(mp_image, curr_timestamp_ms)
        except Exception as err:
            logger.warning("Hand landmarker inference error: %s", err)
            return []

        detections: List[HandDetection] = []
        if not result.hand_landmarks:
            return detections

        for i, landmarks_proto in enumerate(result.hand_landmarks):
            handedness_label = "Unknown"
            score = 1.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
                handedness_label = category.category_name
                score = category.score

            landmarks = []
            pixel_landmarks = []
            for lm in landmarks_proto:
                landmarks.append((lm.x, lm.y, lm.z))
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pixel_landmarks.append((px, py))

            detections.append(
                HandDetection(
                    handedness=handedness_label,
                    score=score,
                    landmarks=landmarks,
                    pixel_landmarks=pixel_landmarks,
                )
            )

        return detections

    def draw_skeleton(
        self,
        frame: np.ndarray,
        detections: List[HandDetection],
        line_color: Tuple[int, int, int] = (255, 200, 0),     # Cyan
        joint_color: Tuple[int, int, int] = (255, 255, 255),  # White
        tip_color: Tuple[int, int, int] = (50, 255, 50),      # Neon Green
    ) -> None:
        """
        Renders a futuristic cybernetic hand skeleton on the frame.
        Distinguishes fingertip nodes with larger glowing circles.
        """
        for hand in detections:
            pts = hand.pixel_landmarks
            if len(pts) < 21:
                continue

            # Draw bone connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = pts[start_idx]
                p2 = pts[end_idx]
                cv2.line(frame, p1, p2, line_color, 2, cv2.LINE_AA)

            # Draw joints and fingertips
            for idx, pt in enumerate(pts):
                if idx in FINGERTIP_INDICES:
                    # Outer halo
                    cv2.circle(frame, pt, 7, tip_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 5, line_color, 1, cv2.LINE_AA)

            # Hand label badge above wrist (landmark 0)
            wrist_x, wrist_y = pts[0]
            label = f"{hand.handedness} ({int(hand.score * 100)}%)"
            badge_y = max(20, wrist_y + 25)
            cv2.putText(
                frame,
                label,
                (wrist_x - 40, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (wrist_x - 40, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        """Safely release the MediaPipe Tasks landmarker."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
            self._is_initialized = False
            logger.info("HandTracker closed cleanly.")
