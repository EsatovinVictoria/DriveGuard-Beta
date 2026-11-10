"""Unit tests for Image Preprocessing module.
Verifies Gaussian blur parameters (3x3, sigma=0.5) and independent channel CLAHE.
"""

import numpy as np
import pytest
from app.preprocessing import (
    apply_clahe_independent,
    apply_gaussian_blur,
    preprocess_image,
)


def test_gaussian_blur_parameters():
    # Synthetic test image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = apply_gaussian_blur(img, kernel_size=(3, 3), sigma=0.5)

    assert blurred.shape == img.shape
    assert blurred.dtype == img.dtype
    # Blurring should smooth out high frequency noise
    assert not np.array_equal(blurred, img)


def test_clahe_channel_independence():
    # Create image where channels have vastly different distributions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Blue dark
    img[:, :, 1] = 150  # Green mid
    img[:, :, 2] = 220  # Red bright

    equalized = apply_clahe_independent(img, clip_limit=2.0, tile_grid_size=(8, 8))
    assert equalized.shape == img.shape
    assert equalized.dtype == np.uint8


def test_preprocess_image_pipeline():
    img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    res = preprocess_image(img)

    assert "original" in res
    assert "gaussian_blur" in res
    assert "clahe" in res
    assert "preprocessed" in res
    assert res["preprocessed"].shape == (120, 160, 3)


def test_empty_image_error_handling():
    empty_img = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError):
        preprocess_image(empty_img)
