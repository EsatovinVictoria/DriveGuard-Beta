"""DriveGuard Eye Region of Interest (ROI) Extraction Module.
Extracts, validates, clamps, and standardizes the eye-pair region for Improved HOG.

Requirements:
- Input image boundary clipping.
- Empty / non-positive dimension handling.
- Dimensionality alignment with HOG requirements (cells of 8x8, blocks of 2x2 cells = min 16x16).
- Configurable resizing or padding with full provenance tracking.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import (
    STANDARDIZE_ROI_SIZE,
    TARGET_ROI_HEIGHT,
    TARGET_ROI_WIDTH,
)


def validate_and_clamp_bbox(
    bbox: Dict[str, int],
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[int, int, int, int]]:
    """Validate bounding box and clamp coordinates to image dimensions.

    Returns (x, y, w, h) or None if box is invalid/degenerate.
    """
    if not bbox:
        return None

    img_h, img_w = image_shape[:2]
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("width", 0))
    h = int(bbox.get("height", 0))

    if w <= 0 or h <= 0:
        return None

    x_clamped = max(0, min(x, img_w - 1))
    y_clamped = max(0, min(y, img_h - 1))
    w_clamped = max(1, min(w, img_w - x_clamped))
    h_clamped = max(1, min(h, img_h - y_clamped))

    if w_clamped < 4 or h_clamped < 4:
        return None

    return x_clamped, y_clamped, w_clamped, h_clamped


def extract_eye_roi(
    image: np.ndarray,
    eye_coordinates: Dict[str, int],
    standardize: bool = STANDARDIZE_ROI_SIZE,
    target_height: int = TARGET_ROI_HEIGHT,
    target_width: int = TARGET_ROI_WIDTH,
) -> Dict[str, Any]:
    """Extract and condition the ocular ROI for feature extraction.

    Args:
        image: Source image containing face/eyes.
        eye_coordinates: Dict with 'x', 'y', 'width', 'height'.
        standardize: Whether to standardize dimensions to (target_height, target_width).
        target_height: Standardized height (must be multiple of 8, >= 16).
        target_width: Standardized width (must be multiple of 8, >= 16).

    Returns:
        Dict containing:
        - success: bool
        - raw_roi: np.ndarray (unscaled crop)
        - processed_roi: np.ndarray (ready for HOG)
        - coordinates: dict
        - dimensions: Tuple[int, int]
        - message: str
    """
    if image is None or image.size == 0:
        return {
            "success": False,
            "raw_roi": None,
            "processed_roi": None,
            "coordinates": None,
            "dimensions": (0, 0),
            "message": "Input image is None or empty",
        }

    clamped = validate_and_clamp_bbox(eye_coordinates, image.shape)
    if clamped is None:
        return {
            "success": False,
            "raw_roi": None,
            "processed_roi": None,
            "coordinates": eye_coordinates,
            "dimensions": (0, 0),
            "message": "Invalid or degenerate eye bounding coordinates",
        }

    x, y, w, h = clamped
    raw_crop = image[y : y + h, x : x + w].copy()

    if raw_crop.size == 0:
        return {
            "success": False,
            "raw_roi": None,
            "processed_roi": None,
            "coordinates": {"x": x, "y": y, "width": w, "height": h},
            "dimensions": (0, 0),
            "message": "Cropped ROI array is empty",
        }

    # Standardize or adjust dimensions for HOG (must be divisible by 8 and >= 16)
    if standardize:
        # Standardize using area interpolation if downsampling, linear if upsampling
        interp = cv2.INTER_AREA if (w >= target_width and h >= target_height) else cv2.INTER_LINEAR
        final_roi = cv2.resize(raw_crop, (target_width, target_height), interpolation=interp)
    else:
        # Pad / crop raw crop to multiples of 8 without scaling
        cur_h, cur_w = raw_crop.shape[:2]
        rem_h = cur_h % 8
        rem_w = cur_w % 8
        adj_h = cur_h - rem_h
        adj_w = cur_w - rem_w

        # Ensure minimum 2x2 cells (16x16 pixels)
        if adj_h < 16 or adj_w < 16:
            final_roi = cv2.resize(raw_crop, (max(16, adj_w), max(16, adj_h)), interpolation=cv2.INTER_LINEAR)
        else:
            final_roi = raw_crop[:adj_h, :adj_w].copy()

    return {
        "success": True,
        "raw_roi": raw_crop,
        "processed_roi": final_roi,
        "coordinates": {"x": x, "y": y, "width": w, "height": h},
        "dimensions": (final_roi.shape[0], final_roi.shape[1]),
        "message": f"ROI extracted successfully: {final_roi.shape[1]}x{final_roi.shape[0]} px",
    }
