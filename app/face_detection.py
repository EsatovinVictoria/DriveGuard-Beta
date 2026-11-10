"""DriveGuard Face Detection Module.
Provides interface for face localization within driver frames.

RESEARCH NOTE:
Bakheet & Al-Hamadi (Brain Sciences 2021, Section 2.2) utilize an:
"improved cascaded adaBoost classifier based on an extended set of Haar-like features"
(edge, line, center-surrounded, and special diagonal line features).

Because the original trained cascade weights from the author are not publicly distributed,
this module implements an APPROXIMATION using OpenCV's Haar Cascade classifier
(haarcascade_frontalface_default.xml / haarcascade_frontalface_alt2.xml).

Status: APPROXIMATION (Explicitly documented; not claimed as the exact research model).
"""

import os
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from app.config import (
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE,
    FACE_DETECTION_SCALE_FACTOR,
)


class FaceDetector:
    """Haar Cascade face detector operating as a modular approximation."""

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        
        self.cascade_path = cascade_path
        self.classifier = cv2.CascadeClassifier(self.cascade_path)
        if self.classifier.empty():
            # Try alternate cascade if default failed
            alt_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
            if os.path.exists(alt_path):
                self.cascade_path = alt_path
                self.classifier = cv2.CascadeClassifier(alt_path)

    def detect_face(
        self,
        image: np.ndarray,
        scale_factor: float = FACE_DETECTION_SCALE_FACTOR,
        min_neighbors: int = FACE_DETECTION_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = FACE_DETECTION_MIN_SIZE,
    ) -> Dict[str, Any]:
        """Detect primary driver face in the preprocessed or raw frame.

        Returns dictionary formatted for downstream pipeline stages:
        - face_detected: bool
        - face_box: dict with x, y, width, height (or None)
        - method: str
        - method_status: 'APPROXIMATION'
        - cropped_face: numpy array (or None)
        - annotated_image: copy of image with visualization overlay
        """
        if image is None or image.size == 0:
            return {
                "face_detected": False,
                "face_box": None,
                "method": "APPROXIMATION_HAAR_CASCADE",
                "method_status": "APPROXIMATION",
                "message": "Input image is empty or None",
                "cropped_face": None,
                "annotated_image": None,
            }

        # Convert to grayscale for Haar cascade
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        annotated = image.copy()

        if len(faces) == 0:
            return {
                "face_detected": False,
                "face_box": None,
                "method": "APPROXIMATION_HAAR_CASCADE",
                "method_status": "APPROXIMATION",
                "message": "No face detected in frame",
                "cropped_face": None,
                "annotated_image": annotated,
            }

        # Select the largest face by area (primary driver)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = [int(v) for v in faces_sorted[0]]

        # Ensure coordinates are within image boundaries
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        cropped = image[y : y + h, x : x + w].copy()

        # Draw green bounding box on annotated image
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Face (Haar Approx)",
            (x, max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        return {
            "face_detected": True,
            "face_box": {"x": x, "y": y, "width": w, "height": h},
            "method": "APPROXIMATION_HAAR_CASCADE",
            "method_status": "APPROXIMATION",
            "message": "Face successfully detected",
            "cropped_face": cropped,
            "annotated_image": annotated,
        }


# Global instance for convenient functional usage
_default_detector: Optional[FaceDetector] = None

def detect_face(image: np.ndarray) -> Dict[str, Any]:
    """Functional convenience entrypoint for face detection."""
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector()
    return _default_detector.detect_face(image)
