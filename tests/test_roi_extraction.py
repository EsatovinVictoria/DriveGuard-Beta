"""Unit tests for Eye ROI Extraction module.
Verifies boundary clipping, dimension standardization, and empty ROI handling.
"""

import numpy as np
from app.roi_extraction import extract_eye_roi, validate_and_clamp_bbox


def test_validate_and_clamp_bbox():
    # Box exceeding image boundary
    bbox = {"x": 80, "y": 80, "width": 50, "height": 50}
    img_shape = (100, 100, 3)
    clamped = validate_and_clamp_bbox(bbox, img_shape)

    assert clamped is not None
    x, y, w, h = clamped
    assert x + w <= 100
    assert y + h <= 100
    assert x == 80 and y == 80
    assert w == 20 and h == 20


def test_invalid_negative_box():
    bbox = {"x": -10, "y": -10, "width": 0, "height": 5}
    clamped = validate_and_clamp_bbox(bbox, (100, 100, 3))
    assert clamped is None


def test_roi_standardization():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    bbox = {"x": 20, "y": 20, "width": 40, "height": 30}
    res = extract_eye_roi(img, bbox, standardize=True, target_height=24, target_width=64)

    assert res["success"] is True
    assert res["processed_roi"].shape == (24, 64, 3)
    assert res["dimensions"] == (24, 64)
