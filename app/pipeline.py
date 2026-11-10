"""DriveGuard Integrated Computer Vision Pipeline.
Connects all processing stages from raw frame to final binary feature vector:
Input Image -> Preprocessing -> Face Detection -> Eye Localization -> Eye ROI -> Improved HOG.
"""

import time
from typing import Any, Dict, Optional
import cv2
import numpy as np

from app.config import (
    EYE_LOCALIZATION_MODE,
    STANDARDIZE_ROI_SIZE,
    TARGET_ROI_HEIGHT,
    TARGET_ROI_WIDTH,
)
from app.eye_localization import locate_eyes
from app.face_detection import detect_face
from app.improved_hog import extract_improved_hog
from app.preprocessing import preprocess_image
from app.roi_extraction import extract_eye_roi
from app.visualization import (
    colorize_gradient_magnitude,
    colorize_gradient_orientation,
    image_to_base64_data_url,
    render_binary_matrix_raster,
    render_feature_vector_barcode,
    render_shifted_histograms_grid,
)


class DriveGuardPipeline:
    """End-to-end computer vision pipeline for driver drowsiness feature extraction."""

    def __init__(self):
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        eye_mode: str = EYE_LOCALIZATION_MODE,
        manual_eye_box: Optional[Dict[str, int]] = None,
        standardize_roi: bool = STANDARDIZE_ROI_SIZE,
        target_roi_h: int = TARGET_ROI_HEIGHT,
        target_roi_w: int = TARGET_ROI_WIDTH,
        include_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Execute full sequential pipeline on an input frame.

        Returns structured report suitable for both backend logging and web UI presentation.
        """
        start_time = time.perf_counter()

        if frame is None or frame.size == 0:
            return {
                "status": "failed",
                "stage_failed": "input_validation",
                "message": "Input frame is empty or None",
                "processing_time_ms": 0.0,
                "preprocessing": {},
                "face_detection": {},
                "eye_localization": {},
                "roi": {},
                "hog_features": {},
                "visualizations": {},
            }

        visualizations: Dict[str, str] = {}

        # -------------------------------------------------------------------
        # STAGE 1: Image Preprocessing (Gaussian Blur & Channel-independent CLAHE)
        # -------------------------------------------------------------------
        try:
            prep_dict = preprocess_image(frame)
            preprocessed_frame = prep_dict["preprocessed"]

            if include_visualizations:
                visualizations["original"] = image_to_base64_data_url(prep_dict["original"])
                visualizations["gaussian_blur"] = image_to_base64_data_url(prep_dict["gaussian_blur"])
                visualizations["clahe"] = image_to_base64_data_url(prep_dict["clahe"])
        except Exception as e:
            return self._build_failure_response(
                stage="preprocessing",
                message=f"Preprocessing failed: {str(e)}",
                start_time=start_time,
                visualizations=visualizations,
            )

        # -------------------------------------------------------------------
        # STAGE 2: Face Detection (AdaBoost Haar cascade approximation)
        # -------------------------------------------------------------------
        face_result = detect_face(preprocessed_frame)
        if not face_result["face_detected"]:
            if include_visualizations and face_result["annotated_image"] is not None:
                visualizations["face_annotated"] = image_to_base64_data_url(face_result["annotated_image"])
            return self._build_failure_response(
                stage="face_detection",
                message=face_result.get("message", "Driver face not detected"),
                start_time=start_time,
                preprocessing={"shape": preprocessed_frame.shape},
                face_detection=face_result,
                visualizations=visualizations,
            )

        cropped_face = face_result["cropped_face"]
        face_box = face_result["face_box"]

        if include_visualizations:
            visualizations["face_annotated"] = image_to_base64_data_url(face_result["annotated_image"])
            visualizations["cropped_face"] = image_to_base64_data_url(cropped_face)

        # -------------------------------------------------------------------
        # STAGE 3: Eye Localization (Approximation / Manual ROI / ASM placeholder)
        # -------------------------------------------------------------------
        eye_result = locate_eyes(
            face_image=cropped_face,
            mode=eye_mode,
            manual_box=manual_eye_box,
        )

        if not eye_result["eye_detected"]:
            if include_visualizations and eye_result["annotated_face"] is not None:
                visualizations["eye_annotated"] = image_to_base64_data_url(eye_result["annotated_face"])
            return self._build_failure_response(
                stage="eye_localization",
                message=eye_result.get("message", "Eye localization failed"),
                start_time=start_time,
                preprocessing={"shape": preprocessed_frame.shape},
                face_detection=face_result,
                eye_localization=eye_result,
                visualizations=visualizations,
            )

        if include_visualizations:
            visualizations["eye_annotated"] = image_to_base64_data_url(eye_result["annotated_face"])

        # -------------------------------------------------------------------
        # STAGE 4: Eye ROI Extraction & Standardization
        # -------------------------------------------------------------------
        eye_box = eye_result["eye_box"]
        roi_result = extract_eye_roi(
            image=cropped_face,
            eye_coordinates=eye_box,
            standardize=standardize_roi,
            target_height=target_roi_h,
            target_width=target_roi_w,
        )

        if not roi_result["success"]:
            return self._build_failure_response(
                stage="roi_extraction",
                message=roi_result.get("message", "ROI extraction failed"),
                start_time=start_time,
                preprocessing={"shape": preprocessed_frame.shape},
                face_detection=face_result,
                eye_localization=eye_result,
                roi=roi_result,
                visualizations=visualizations,
            )

        processed_roi = roi_result["processed_roi"]

        if include_visualizations:
            visualizations["eye_roi_raw"] = image_to_base64_data_url(roi_result["raw_roi"])
            visualizations["eye_roi_processed"] = image_to_base64_data_url(processed_roi)

        # -------------------------------------------------------------------
        # STAGE 5: Improved HOG Feature Extraction (Shifted Orientations)
        # -------------------------------------------------------------------
        try:
            hog_result = extract_improved_hog(processed_roi)
        except Exception as e:
            return self._build_failure_response(
                stage="improved_hog",
                message=f"Improved HOG extraction failed: {str(e)}",
                start_time=start_time,
                preprocessing={"shape": preprocessed_frame.shape},
                face_detection=face_result,
                eye_localization=eye_result,
                roi=roi_result,
                visualizations=visualizations,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if include_visualizations:
            # Colorized gradient magnitude
            mag_col = colorize_gradient_magnitude(hog_result["gradient_magnitude"])
            visualizations["hog_magnitude"] = image_to_base64_data_url(mag_col)

            # Colorized gradient orientation
            ori_col = colorize_gradient_orientation(
                hog_result["gradient_orientation"], hog_result["gradient_magnitude"]
            )
            visualizations["hog_orientation"] = image_to_base64_data_url(ori_col)

            # Sample shifted histogram matrix
            if hog_result.get("sample_shifted_histograms") is not None:
                shifted_img = render_shifted_histograms_grid(hog_result["sample_shifted_histograms"])
                visualizations["hog_shifted"] = image_to_base64_data_url(shifted_img)

            # Sample binary comparison matrix
            if hog_result.get("sample_binary_matrix") is not None:
                bin_img = render_binary_matrix_raster(hog_result["sample_binary_matrix"])
                visualizations["hog_binary_matrix"] = image_to_base64_data_url(bin_img)

            # 1D feature vector barcode plot
            barcode_img = render_feature_vector_barcode(hog_result["feature_vector"][:256])
            visualizations["hog_vector_plot"] = image_to_base64_data_url(barcode_img)

        # Build clean serializable response
        serializable_hog = {
            "feature_dimension": hog_result["feature_dimension"],
            "cell_size": hog_result["cell_size"],
            "block_size": hog_result["block_size"],
            "orientation_bins": hog_result["orientation_bins"],
            "orientation_shifts": hog_result["orientation_shifts"],
            "num_blocks": hog_result["num_blocks"],
            "pairing_strategy": hog_result["pairing_strategy"],
            "pairs_per_block": hog_result["pairs_per_block"],
            "feature_vector_sample": hog_result["feature_vector"][:64],
            "feature_vector": hog_result["feature_vector"],
        }

        return {
            "status": "success",
            "stage_failed": None,
            "message": "Complete pipeline executed successfully",
            "processing_time_ms": round(elapsed_ms, 2),
            "face_detection": {
                "detected": face_result["face_detected"],
                "box": face_result["face_box"],
                "method": face_result["method"],
                "status": face_result["method_status"],
            },
            "eye_localization": {
                "detected": eye_result["eye_detected"],
                "box": eye_result["eye_box"],
                "method": eye_result["method"],
                "status": eye_result["method_status"],
            },
            "roi": {
                "dimensions": roi_result["dimensions"],
                "coordinates": roi_result["coordinates"],
            },
            "hog_features": serializable_hog,
            "visualizations": visualizations,
        }

    def _build_failure_response(
        self,
        stage: str,
        message: str,
        start_time: float,
        preprocessing: Optional[Dict[str, Any]] = None,
        face_detection: Optional[Dict[str, Any]] = None,
        eye_localization: Optional[Dict[str, Any]] = None,
        roi: Optional[Dict[str, Any]] = None,
        visualizations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Construct a consistent failure response."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "status": "failed",
            "stage_failed": stage,
            "message": message,
            "processing_time_ms": round(elapsed_ms, 2),
            "preprocessing": preprocessing or {},
            "face_detection": {
                "detected": face_detection.get("face_detected", False) if face_detection else False,
                "box": face_detection.get("face_box") if face_detection else None,
                "method": face_detection.get("method", "") if face_detection else "",
                "status": face_detection.get("method_status", "") if face_detection else "",
            },
            "eye_localization": {
                "detected": eye_localization.get("eye_detected", False) if eye_localization else False,
                "box": eye_localization.get("eye_box") if eye_localization else None,
                "method": eye_localization.get("method", "") if eye_localization else "",
                "status": eye_localization.get("method_status", "") if eye_localization else "",
            },
            "roi": roi or {},
            "hog_features": {},
            "visualizations": visualizations or {},
        }


# Global pipeline instance
_default_pipeline: Optional[DriveGuardPipeline] = None

def get_pipeline() -> DriveGuardPipeline:
    """Access global singleton pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DriveGuardPipeline()
    return _default_pipeline
