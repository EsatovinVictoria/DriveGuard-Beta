"""DriveGuard Visualization Module.
Generates informative visualizations and base64 payloads for computer vision pipeline stages:
- Preprocessing comparison
- Face detection annotation
- Eye localization & ROI extraction
- Improved HOG gradient magnitude, orientation color maps, shifted histograms,
  and binary descriptor raster plots.
"""

import base64
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


def image_to_base64_data_url(image: np.ndarray, format_ext: str = ".jpg") -> str:
    """Encode OpenCV BGR or grayscale image to base64 data URL string."""
    if image is None or image.size == 0:
        return ""
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format_ext.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{encoded}"


def colorize_gradient_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Convert floating-point gradient magnitude to a colorized heatmap (INFERNO/JET)."""
    if magnitude is None or magnitude.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Normalize to 0-255
    mag_max = np.max(magnitude)
    if mag_max > 0:
        norm = np.clip((magnitude / mag_max) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(magnitude, dtype=np.uint8)

    colorized = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return colorized


def colorize_gradient_orientation(orientation_rad: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Encode orientation [0, pi) as Hue in HSV color space, weighted by magnitude Value."""
    if orientation_rad is None or orientation_rad.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Hue in OpenCV is [0, 180). Orientation in radians [0, pi) maps to [0, 180)
    hue = np.clip((orientation_rad / np.pi) * 180.0, 0, 179).astype(np.uint8)
    # Saturation maximum
    sat = np.full_like(hue, 255, dtype=np.uint8)
    # Value proportional to magnitude
    mag_max = np.max(magnitude)
    if mag_max > 0:
        val = np.clip((magnitude / mag_max) * 255.0, 0, 255).astype(np.uint8)
    else:
        val = np.zeros_like(hue, dtype=np.uint8)

    hsv = cv2.merge([hue, sat, val])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def render_shifted_histograms_grid(shifted_histograms: np.ndarray) -> np.ndarray:
    """Render 8 rows of shifted 8-bin histograms (Paper Figure 6) as an image.

    shifted_histograms shape: (8, 8)
    """
    if shifted_histograms is None or len(shifted_histograms) == 0:
        return np.zeros((100, 200, 3), dtype=np.uint8)

    # Dimensions
    row_height = 30
    bar_width = 16
    n_shifts, n_bins = shifted_histograms.shape
    img_h = n_shifts * row_height + 20
    img_w = n_bins * bar_width + 80

    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    max_val = np.max(shifted_histograms) if np.max(shifted_histograms) > 0 else 1.0

    for eps in range(n_shifts):
        y_base = (eps + 1) * row_height
        # Draw label
        cv2.putText(
            canvas,
            f"e={eps}:",
            (8, y_base - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        for b in range(n_bins):
            val = shifted_histograms[eps, b]
            h_bar = int((val / max_val) * (row_height - 6))
            x1 = 50 + b * bar_width
            y1 = y_base - h_bar
            x2 = x1 + bar_width - 3
            y2 = y_base
            # Blue-red gradient depending on bin index
            color = (int(220 - b * 20), int(100 + b * 15), int(40 + b * 25))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

    return canvas


def render_binary_matrix_raster(binary_matrix: np.ndarray, cell_px: int = 16) -> np.ndarray:
    """Render binary comparison matrix as a high-contrast pixel block raster."""
    if binary_matrix is None or binary_matrix.size == 0:
        return np.zeros((50, 50, 3), dtype=np.uint8)

    rows, cols = binary_matrix.shape
    img_h = rows * cell_px
    img_w = cols * cell_px

    # Create enlarged block image
    raster = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            val = binary_matrix[r, c]
            color = (255, 255, 255) if val == 1 else (20, 20, 20)
            y1, y2 = r * cell_px, (r + 1) * cell_px
            x1, x2 = c * cell_px, (c + 1) * cell_px
            raster[y1:y2, x1:x2] = color
            # Grid line
            cv2.rectangle(raster, (x1, y1), (x2, y2), (80, 80, 80), 1)

    return raster


def render_feature_vector_barcode(feature_vector: List[int], target_width: int = 600, height: int = 60) -> np.ndarray:
    """Render 1D binary feature descriptor as a 2D barcode plot matching Paper Figure 7(b)."""
    if not feature_vector:
        return np.zeros((height, target_width, 3), dtype=np.uint8)

    dim = len(feature_vector)
    canvas = np.full((height, max(target_width, dim), 3), 255, dtype=np.uint8)

    w_canvas = canvas.shape[1]
    step = w_canvas / dim

    for i, bit in enumerate(feature_vector):
        x1 = int(i * step)
        x2 = int((i + 1) * step)
        if bit == 1:
            # Blue impulse line (Paper Figure 7b uses blue vertical stems for value 1)
            cv2.line(canvas, (x1, height - 5), (x1, 5), (200, 50, 20), 1)

    # Draw bottom axis line
    cv2.line(canvas, (0, height - 5), (w_canvas, height - 5), (100, 100, 100), 1)
    return canvas
