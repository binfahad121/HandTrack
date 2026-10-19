"""
HandTrack: Real-Time Hand Tracking & Gesture Control Application.
Main entry point orchestrating camera capture, MediaPipe inference,
gesture classification, virtual mouse control, air drawing, and cybernetic HUD rendering.
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

# Pre-configure matplotlib temp/config directory in workspace
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import cv2
import numpy as np

from config import AppMode, HandTrackConfig
from drawing_canvas import AirDrawingCanvas
from gesture_controller import GestureClassifier, GestureType
from hand_tracker import HandTracker, download_model_if_missing
from mouse_controller import VirtualMouseController
from ui import UIRenderer

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
)
logger = logging.getLogger("HandTrack.Main")


class HandTrackApp:
    """
    Main application coordinator managing runtime state, pipelines, and UI loop.
    """

    def __init__(self, config: Optional[HandTrackConfig] = None):
        self.config = config or HandTrackConfig()
        self.current_mode = AppMode.LIVE_TRACKING

        # Initialize sub-systems
        self.tracker = HandTracker(
            model_path=self.config.model_path,
            max_hands=self.config.max_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.gesture_classifier = GestureClassifier(pinch_threshold=self.config.pinch_threshold)
        self.mouse_controller = VirtualMouseController(
            sensitivity=self.config.cursor_sensitivity,
            smoothing_factor=self.config.smoothing_factor,
            pinch_threshold=self.config.pinch_threshold,
            active_box_scale=self.config.active_box_scale,
        )
        self.canvas = AirDrawingCanvas(
            width=self.config.frame_width,
            height=self.config.frame_height,
            brush_thickness=self.config.brush_thickness,
            eraser_thickness=self.config.eraser_thickness,
            output_dir=self.config.output_dir,
        )
        self.ui = UIRenderer(self.config)

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_running = False

    def initialize_camera(self) -> bool:
        """
        Attempts to open the configured camera index or tests available alternatives.
        """
        logger.info("Initializing camera capture (Index %d)...", self.config.camera_index)
        cap = cv2.VideoCapture(self.config.camera_index)

        # On macOS, if primary camera doesn't open, try fallback index 1
        if not cap.isOpened() and self.config.camera_index == 0:
            logger.warning("Camera index 0 not available. Trying camera index 1...")
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            logger.error("Could not access webcam. Ensure camera permissions are granted.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.cap = cap
        logger.info("Camera initialized successfully.")
        return True

    def setup(self) -> bool:
        """Verifies assets and initializes tracking model."""
        if not self.tracker.initialize():
            logger.error("HandTracker initialization failed. Check model path and requirements.")
            return False
        return True

    def handle_keyboard(self, key_code: int) -> bool:
        """
        Processes keyboard shortcuts. Returns False when application should exit.
        """
        if key_code in [27, ord('q'), ord('Q')]:  # ESC or Q
            logger.info("Exit command received. Shutting down...")
            return False

        # Mode switching
        if key_code == ord('1'):
            self.current_mode = AppMode.LIVE_TRACKING
            self.ui.show_toast("Switched to Mode 1: Live Tracking")
        elif key_code == ord('2'):
            self.current_mode = AppMode.VIRTUAL_MOUSE
            self.ui.show_toast("Switched to Mode 2: Virtual Mouse")
        elif key_code == ord('3'):
            self.current_mode = AppMode.AIR_DRAW
            self.ui.show_toast("Switched to Mode 3: Air Draw")
        elif key_code == ord('4'):
            self.current_mode = AppMode.GESTURE_RECOGNITION
            self.ui.show_toast("Switched to Mode 4: Gesture Recognition")

        # Mouse toggle
        elif key_code in [ord('m'), ord('M')]:
            is_enabled = self.mouse_controller.toggle_enable()
            self.config.mouse_enabled = is_enabled
            self.ui.show_toast(f"Virtual Mouse: {'ARMED [ON]' if is_enabled else 'DISABLED'}")

        # AirDraw actions
        elif key_code in [ord('c'), ord('C')]:
            self.canvas.clear()
            self.ui.show_toast("Drawing Canvas Cleared")
        elif key_code in [ord('z'), ord('Z')]:
            if self.canvas.undo():
                self.ui.show_toast("Undo stroke")
        elif key_code in [ord('e'), ord('E')]:
            is_eraser = self.canvas.toggle_eraser()
            self.ui.show_toast(f"Tool: {'Eraser' if is_eraser else self.canvas.current_color_name}")
        elif key_code in [ord('s'), ord('S')]:
            success, path = self.canvas.save_drawing()
            if success:
                self.ui.show_toast(f"Saved: {os.path.basename(path)}")
            else:
                self.ui.show_toast("Save Failed!")

        # Sensitivity adjustment
        elif key_code == ord('['):
            new_sens = self.mouse_controller.adjust_sensitivity(-0.1)
            self.config.cursor_sensitivity = new_sens
            self.ui.show_toast(f"Sensitivity: {new_sens:.1f}x")
        elif key_code == ord(']'):
            new_sens = self.mouse_controller.adjust_sensitivity(0.1)
            self.config.cursor_sensitivity = new_sens
            self.ui.show_toast(f"Sensitivity: {new_sens:.1f}x")

        # Reset & Help
        elif key_code in [ord('r'), ord('R')]:
            self.config.reset()
            self.mouse_controller.sensitivity = self.config.cursor_sensitivity
            self.mouse_controller.release_all()
            self.ui.show_toast("Settings Reset to Defaults")
        elif key_code in [ord('h'), ord('H')]:
            self.config.show_help = not self.config.show_help

        return True

    def process_single_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, GestureType]:
        """
        Processes a single input frame through tracking, active mode handler, and HUD.
        Mirrors frame horizontally for intuitive selfie view.
        """
        # 1. Flip horizontally for natural mirror selfie view
        frame = cv2.flip(frame, 1)

        # 2. Hand tracking
        detections = self.tracker.process_frame(frame)
        num_hands = len(detections)

        primary_gesture = GestureType.UNKNOWN
        finger_states = None
        telemetry = {}
        mouse_action = "Idle"
        pointer_pt = None

        if num_hands > 0:
            hand = detections[0]  # Primary active hand
            landmarks = hand.landmarks
            pixel_pts = hand.pixel_landmarks

            # Gesture classification
            primary_gesture = self.gesture_classifier.classify(landmarks)
            finger_states = self.gesture_classifier.get_finger_states(landmarks)

            # Telemetry for Mode 1
            if len(landmarks) > 8:
                telemetry["index_tip_norm"] = landmarks[8]
                telemetry["wrist_norm"] = landmarks[0]

            # Mode 2: Virtual Mouse
            if self.current_mode == AppMode.VIRTUAL_MOUSE and len(landmarks) >= 21:
                cursor_pos, mouse_action = self.mouse_controller.update(
                    index_tip=landmarks[8],
                    thumb_tip=landmarks[4],
                    wrist=landmarks[0],
                    middle_mcp=landmarks[9],
                )

            # Mode 3: AirDraw Pointer & Stroke
            if len(pixel_pts) >= 9:
                pointer_pt = pixel_pts[8]  # Index fingertip
                if self.current_mode == AppMode.AIR_DRAW:
                    if primary_gesture == GestureType.INDEX_POINTING:
                        self.canvas.add_point(pointer_pt)
                    else:
                        self.canvas.end_stroke()
        else:
            self.gesture_classifier.reset()
            self.canvas.end_stroke()
            if self.current_mode == AppMode.VIRTUAL_MOUSE:
                self.mouse_controller.release_all()

        # 3. Mode-specific rendering
        if self.current_mode == AppMode.AIR_DRAW:
            # Alpha blend drawing canvas
            frame = self.canvas.blend_with_frame(frame, pointer_pt=pointer_pt)

        if self.config.show_landmarks and num_hands > 0:
            self.tracker.draw_skeleton(frame, detections)

        # 4. Render HUD Overlay
        mouse_info = (
            self.mouse_controller.is_enabled,
            mouse_action,
            self.mouse_controller.sensitivity,
        )
        canvas_info = (
            self.canvas.current_color_name,
            self.canvas.current_color,
            len(self.canvas.strokes),
            self.canvas.is_eraser_active,
        )

        self.ui.render_hud(
            frame=frame,
            mode=self.current_mode,
            num_hands=num_hands,
            gesture=primary_gesture,
            mouse_info=mouse_info,
            canvas_info=canvas_info,
            finger_states=finger_states,
            telemetry=telemetry,
        )

        return frame, primary_gesture

    def run(self) -> None:
        """Main application camera processing loop."""
        if not self.setup():
            logger.error("Failed to setup HandTrack. Exiting.")
            return

        if not self.initialize_camera():
            logger.error("Camera unavailable. You may test logic using '--headless-test'.")
            return

        self._is_running = True
        window_name = "HandTrack - Real-Time Hand Tracking & Gesture Control"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.config.frame_width, self.config.frame_height)

        logger.info("Application running. Press 'H' for help, 'Q' to quit.")

        try:
            while self._is_running:
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    logger.warning("Failed to receive frame from camera.")
                    time.sleep(0.02)
                    continue

                display_frame, _ = self.process_single_frame(raw_frame)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self.handle_keyboard(key):
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.cleanup()

    def run_headless_test(self, num_frames: int = 15) -> bool:
        """
        Executes an automated end-to-end smoke test using synthetic frames.
        Validates pipeline robustness without requiring physical webcam permissions.
        """
        logger.info("Starting automated headless pipeline verification (%d frames)...", num_frames)
        if not self.setup():
            logger.error("Headless test failed: setup error.")
            return False

        h, w = self.config.frame_height, self.config.frame_width

        # Test across all 4 modes
        modes = [
            AppMode.LIVE_TRACKING,
            AppMode.VIRTUAL_MOUSE,
            AppMode.AIR_DRAW,
            AppMode.GESTURE_RECOGNITION,
        ]

        for i in range(num_frames):
            mode = modes[i % len(modes)]
            self.current_mode = mode

            # Generate synthetic test frame (dark gradient with noise)
            test_frame = np.full((h, w, 3), 30, dtype=np.uint8)
            cv2.circle(test_frame, (w // 2, h // 2), 40, (120, 80, 50), -1)

            # Exercise canvas drawing during AirDraw mode
            if mode == AppMode.AIR_DRAW:
                self.canvas.add_point((300 + i * 10, 300 + i * 5))

            processed_frame, _ = self.process_single_frame(test_frame)

            # Verify output frame integrity
            assert processed_frame.shape == (h, w, 3), f"Invalid output frame shape: {processed_frame.shape}"
            assert processed_frame.dtype == np.uint8, f"Invalid frame dtype: {processed_frame.dtype}"

        # Test saving canvas
        save_success, save_path = self.canvas.save_drawing()
        assert save_success, f"Failed to save drawing in headless test: {save_path}"
        assert os.path.exists(save_path), f"Saved drawing does not exist: {save_path}"

        logger.info("Headless pipeline test PASSED successfully!")
        self.cleanup()
        return True

    def cleanup(self) -> None:
        """Cleanly releases all hardware and system resources."""
        logger.info("Releasing resources...")
        self._is_running = False
        if self.mouse_controller:
            self.mouse_controller.release_all()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.tracker is not None:
            self.tracker.close()
        cv2.destroyAllWindows()
        logger.info("HandTrack shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HandTrack: Real-Time Hand Tracking & Gesture Control")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height (default: 720)")
    parser.add_argument("--headless-test", action="store_true", help="Run automated headless pipeline test and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app_config = HandTrackConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
    )

    app = HandTrackApp(app_config)

    if args.headless_test:
        success = app.run_headless_test()
        sys.exit(0 if success else 1)
    else:
        app.run()
