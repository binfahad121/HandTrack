"""
Unit tests for HandTracker initialization and missing model error handling.
"""

from hand_tracker import HandTracker, download_model_if_missing


def test_tracker_missing_model_file_handling():
    """Verify that an invalid or missing model path fails gracefully without crashing."""
    invalid_path = "/nonexistent/directory/invalid_model.task"
    tracker = HandTracker(model_path=invalid_path)
    
    # Initialize should return False and log useful diagnostic error
    success = tracker.initialize()
    assert success is False
    assert tracker._is_initialized is False

    # Processing frames on uninitialized tracker should return empty list safely
    dummy_frame = None
    results = tracker.process_frame(dummy_frame)
    assert results == []

    # Safe release should not raise
    tracker.close()
