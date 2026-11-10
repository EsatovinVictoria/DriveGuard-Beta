"""Unit tests for Face Detection module.
Verifies Haar cascade face detection approximation, failure handling, and provenance flags.
"""

import numpy as np
from app.face_detection import FaceDetector, detect_face


def test_face_detection_failure_handling():
    # Pure black image (no face)
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = detect_face(black_img)

    assert res["face_detected"] is False
    assert res["face_box"] is None
    assert res["method_status"] == "APPROXIMATION"
    assert "No face detected" in res["message"]


def test_face_detector_empty_input():
    empty_img = np.array([], dtype=np.uint8)
    res = detect_face(empty_img)
    assert res["face_detected"] is False
    assert res["cropped_face"] is None


def test_face_detector_structure():
    detector = FaceDetector()
    assert not detector.classifier.empty()
