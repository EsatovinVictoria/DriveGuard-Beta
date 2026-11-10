"""DriveGuard Image Preprocessing Module.
Implements the preprocessing pipeline described in Bakheet & Al-Hamadi (Brain Sciences 2021, Section 2.1):
1. 2D Gaussian blur filter (3x3 neighborhood, sigma=0.5) to suppress noise.
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) applied independently
   to each color channel for lighting compensation.
3. Optional resolution reduction for computational efficiency.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from app.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_PER_CHANNEL,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    RESIZE_ENABLED,
    RESIZE_TARGET,
)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Apply 2D Gaussian blur filter.

    Parameters from paper (Section 2.1):
    - Kernel size: 3x3 neighborhood (EXACT_FROM_PAPER)
    - Sigma: 0.5 uniform standard deviation (EXACT_FROM_PAPER)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image: array is empty or None")

    blurred = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_clahe_independent(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply CLAHE independently to each color channel.

    Paper specification (Section 2.1):
    - "each color channel is independently equalized to produce a better
      lighting-compensated image" (EXACT_FROM_PAPER)
    - clipLimit and tileGridSize are NOT defined in paper (TEMPORARY_ASSUMPTION).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image: array is empty or None")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Handle grayscale (single channel)
    if len(image.shape) == 2 or image.shape[2] == 1:
        return clahe.apply(image)

    # Multi-channel color image (BGR)
    channels = cv2.split(image)
    equalized_channels = [clahe.apply(ch) for ch in channels]
    equalized_image = cv2.merge(equalized_channels)
    return equalized_image


def resize_image_if_enabled(
    image: np.ndarray,
    enabled: bool = RESIZE_ENABLED,
    target_size: Tuple[int, int] = RESIZE_TARGET,
) -> np.ndarray:
    """Optionally reduce image resolution for computational efficiency.

    Paper note (Section 2.1):
    - Paper states resolution can be reduced, but does not provide explicit values.
    - Default setting is DISABLED to preserve original 640x480 resolution.
    """
    if not enabled:
        return image

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: np.ndarray,
    gaussian_kernel: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE,
    gaussian_sigma: float = GAUSSIAN_SIGMA,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
    resize_enabled: bool = RESIZE_ENABLED,
    resize_target: Tuple[int, int] = RESIZE_TARGET,
) -> Dict[str, np.ndarray]:
    """Execute complete preprocessing pipeline.

    Returns dictionary containing original, intermediate, and final preprocessed images.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty")

    original = image.copy()
    blurred = apply_gaussian_blur(original, kernel_size=gaussian_kernel, sigma=gaussian_sigma)
    clahe_img = apply_clahe_independent(blurred, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid)
    final_img = resize_image_if_enabled(clahe_img, enabled=resize_enabled, target_size=resize_target)

    return {
        "original": original,
        "gaussian_blur": blurred,
        "clahe": clahe_img,
        "preprocessed": final_img,
    }
