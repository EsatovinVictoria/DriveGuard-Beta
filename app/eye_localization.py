"""DriveGuard Eye Localization Module.
Localizes the eye-pair region within the detected driver face.

METHODOLOGICAL NOTE:
Bakheet & Al-Hamadi (Brain Sciences 2021, Section 2.2) state:
"For detecting an eye-pair region, we use an improved active shape model (ASM) algorithm
based on statistical learning models ... The ASM is trained on a set of specific points
representing facial feature contours ... Principal component analysis (PCA) is then
performed ... an iterative scheme using a cost function can be adopted to match models iteratively."

CRITICAL RESEARCH GAP IN PAPER:
The authors did NOT publish:
1. The facial landmark schema or point count.
2. The manual landmark annotation training set.
3. The PCA eigenvector matrices or variance thresholds.
4. The exact cost function formulation and termination threshold.
5. Any pre-trained ASM model files.

SUPPORTED MODES:
1. "APPROXIMATION_HAAR": Uses OpenCV Haar eye cascade / anthropometric upper-face
   geometric heuristic to localize the eye pair. Explicitly labeled APPROXIMATION.
2. "MANUAL_ROI": Accepts user-selected bounding box from the web interface canvas,
   enabling experimentation with Improved HOG without algorithmic eye detector bias.
3. "ASM_MODEL": Placeholder interface that returns status NOT_IMPLEMENTED pending
   model weights / landmark dataset availability.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import (
    EYE_CASCADE_FILENAME,
    EYE_FALLBACK_TO_UPPER_FACE,
    EYE_LOCALIZATION_MODE,
)


class EyeLocalizer:
    """Modular eye localizer with multi-mode fallback and research provenance labeling."""

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = os.path.join(cv2.data.haarcascades, EYE_CASCADE_FILENAME)
        
        self.cascade_path = cascade_path
        self.eye_classifier = cv2.CascadeClassifier(self.cascade_path)
        if self.eye_classifier.empty():
            alt_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
            if os.path.exists(alt_path):
                self.cascade_path = alt_path
                self.eye_classifier = cv2.CascadeClassifier(alt_path)

    def locate_eyes(
        self,
        face_image: np.ndarray,
        mode: Optional[str] = None,
        manual_box: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Locate eye region within a cropped face image.

        Args:
            face_image: Cropped face image (BGR uint8).
            mode: 'APPROXIMATION_HAAR', 'MANUAL_ROI', or 'ASM_MODEL'.
            manual_box: Optional dict with 'x', 'y', 'width', 'height' for MANUAL_ROI.

        Returns:
            Dict containing:
            - eye_detected: bool
            - eye_box: dict (x, y, width, height) relative to face_image
            - method: str
            - method_status: 'APPROXIMATION', 'APPROVED_IMPROVISATION', or 'NOT_IMPLEMENTED'
            - message: str
            - cropped_eyes: np.ndarray or None
            - annotated_face: np.ndarray
        """
        if face_image is None or face_image.size == 0:
            return {
                "eye_detected": False,
                "eye_box": None,
                "method": "NONE",
                "method_status": "ERROR",
                "message": "Input face image is empty or None",
                "cropped_eyes": None,
                "annotated_face": None,
            }

        active_mode = mode or EYE_LOCALIZATION_MODE
        h_face, w_face = face_image.shape[:2]
        annotated = face_image.copy()

        # -------------------------------------------------------------------
        # MODE 1: MANUAL ROI (Interactive user selection from Web Canvas)
        # -------------------------------------------------------------------
        if active_mode == "MANUAL_ROI" and manual_box is not None:
            bx = max(0, min(manual_box.get("x", 0), w_face - 1))
            by = max(0, min(manual_box.get("y", 0), h_face - 1))
            bw = max(1, min(manual_box.get("width", 10), w_face - bx))
            bh = max(1, min(manual_box.get("height", 10), h_face - by))

            cropped = face_image[by : by + bh, bx : bx + bw].copy()
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 105, 180), 2)
            cv2.putText(
                annotated,
                "Eye Pair (Manual ROI)",
                (bx, max(12, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 105, 180),
                1,
                cv2.LINE_AA,
            )
            return {
                "eye_detected": True,
                "eye_box": {"x": bx, "y": by, "width": bw, "height": bh},
                "method": "MANUAL_ROI_SELECTION",
                "method_status": "APPROVED_IMPROVISATION",
                "message": "Eye region defined via manual canvas selection",
                "cropped_eyes": cropped,
                "annotated_face": annotated,
            }

        # -------------------------------------------------------------------
        # MODE 2: ASM_MODEL (Placeholder when model weights are missing)
        # -------------------------------------------------------------------
        if active_mode == "ASM_MODEL":
            return {
                "eye_detected": False,
                "eye_box": None,
                "method": "IMPROVED_ACTIVE_SHAPE_MODEL",
                "method_status": "NOT_IMPLEMENTED",
                "message": (
                    "Improved ASM requires proprietary landmark weights and PCA matrices "
                    "not provided in the Bakheet & Al-Hamadi paper. Please use APPROXIMATION_HAAR "
                    "or MANUAL_ROI mode."
                ),
                "cropped_eyes": None,
                "annotated_face": annotated,
            }

        # -------------------------------------------------------------------
        # MODE 3: APPROXIMATION_HAAR (Haar cascade + anthropometric eye-pair)
        # -------------------------------------------------------------------
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image.copy()

        # Restrict ocular search strictly to the eye band (15% to 55% of face height)
        # This guarantees nostrils and mouth are completely excluded from detection.
        y_ocular_start = int(h_face * 0.15)
        y_ocular_end = int(h_face * 0.55)
        roi_gray = gray[y_ocular_start:y_ocular_end, :]

        eyes = self.eye_classifier.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(int(w_face * 0.10), int(h_face * 0.08)),
        )

        eye_box: Optional[Dict[str, int]] = None
        face_mid_x = w_face / 2.0

        if len(eyes) >= 2:
            # Sort detected eyes horizontally
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            leftmost = eyes_sorted[0]
            rightmost = eyes_sorted[-1]

            # Merge into a comprehensive eye-pair box spanning both eyes (Paper Figure 4 & 7a)
            ex1 = max(0, leftmost[0] - int(leftmost[2] * 0.25))
            ex2 = min(w_face, rightmost[0] + rightmost[2] + int(rightmost[2] * 0.25))

            # Vertical extent covering eyebrows to lower eye contour
            ey1 = max(0, y_ocular_start + min(leftmost[1], rightmost[1]) - int(leftmost[3] * 0.35))
            ey2 = min(h_face, y_ocular_start + max(leftmost[1] + leftmost[3], rightmost[1] + rightmost[3]) + int(rightmost[3] * 0.35))

            ew = ex2 - ex1
            eh = ey2 - ey1
            if ew > 20 and eh > 12:
                eye_box = {"x": int(ex1), "y": int(ey1), "width": int(ew), "height": int(eh)}

        elif len(eyes) == 1:
            # Single eye found (other eye is occluded, closed, or obscured by glasses).
            # Use facial bilateral symmetry to construct the complete eye-pair box!
            e = eyes[0]
            eye_center_x = e[0] + e[2] / 2.0
            dist_to_midline = abs(eye_center_x - face_mid_x)

            # Ensure minimum separation to cover both eyes
            half_span = max(dist_to_midline + e[2] * 0.6, w_face * 0.36)
            ex1 = max(0, int(face_mid_x - half_span))
            ex2 = min(w_face, int(face_mid_x + half_span))

            ey1 = max(0, int(y_ocular_start + e[1] - e[3] * 0.35))
            ey2 = min(h_face, int(y_ocular_start + e[1] + e[3] * 1.35))

            ew = ex2 - ex1
            eh = ey2 - ey1
            if ew > 20 and eh > 12:
                eye_box = {"x": int(ex1), "y": int(ey1), "width": int(ew), "height": int(eh)}

        # Fallback to anthropometric eye-pair band if cascade missed both eyes
        # (e.g. dark sunglasses, closed eyes, extreme head tilt)
        if eye_box is None and EYE_FALLBACK_TO_UPPER_FACE:
            # Standard facial proportions: ocular strip spans 20% to 52% height, 10% to 90% width
            fx = int(w_face * 0.10)
            fy = int(h_face * 0.20)
            fw = int(w_face * 0.80)
            fh = int(h_face * 0.32)
            eye_box = {"x": fx, "y": fy, "width": fw, "height": fh}

        if eye_box is not None:
            bx, by, bw, bh = eye_box["x"], eye_box["y"], eye_box["width"], eye_box["height"]
            cropped = face_image[by : by + bh, bx : bx + bw].copy()

            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                "Eye-Pair (Haar/Anthro Approx)",
                (bx, max(12, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            return {
                "eye_detected": True,
                "eye_box": eye_box,
                "method": "APPROXIMATION_HAAR_AND_ANTHROPOMETRIC",
                "method_status": "APPROXIMATION",
                "message": "Eye-pair region localized via Haar/anthropometric approximation",
                "cropped_eyes": cropped,
                "annotated_face": annotated,
            }

        return {
            "eye_detected": False,
            "eye_box": None,
            "method": "APPROXIMATION_HAAR",
            "method_status": "APPROXIMATION",
            "message": "Eye localization failed: no ocular features identified",
            "cropped_eyes": None,
            "annotated_face": annotated,
        }


# Functional singleton
_default_localizer: Optional[EyeLocalizer] = None

def locate_eyes(
    face_image: np.ndarray,
    mode: Optional[str] = None,
    manual_box: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Functional convenience entrypoint for eye localization."""
    global _default_localizer
    if _default_localizer is None:
        _default_localizer = EyeLocalizer()
    return _default_localizer.locate_eyes(face_image, mode=mode, manual_box=manual_box)
