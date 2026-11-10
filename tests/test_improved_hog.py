"""Unit tests for Improved HOG Feature Extraction.
Verifies centered derivative mask [-1, 0, 1], orientation shifting,
exact Equation 5 binarization logic with manual truth tables, and feature dimensionality.
"""

import numpy as np
import pytest
from app.improved_hog import (
    binarize_cell_pair,
    compute_gradients_centered_mask,
    extract_improved_hog,
    generate_shifted_histograms,
)


def test_binarization_equation_5_manual_truth_table():
    """Manual truth table verification of Paper Equation 5:

    b_{c1, c2}(k, epsilon) = 1 if v_c1(k) >= v_c2((k + epsilon) % 8) else 0.
    """
    # Deterministic synthetic cell histograms
    # v_c1: [10, 20, 30, 40, 50, 60, 70, 80]
    # v_c2: [15, 25, 35, 45, 55, 65, 75, 85]
    c1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    c2 = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0])

    # 1. Test epsilon = 0:
    # v_c1[k] >= v_c2[k] -> 10>=15 (F), 20>=25 (F), ..., 80>=85 (F) -> all 0
    bin_eps0 = binarize_cell_pair(c1, c2, shifts=[0])
    expected_eps0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(bin_eps0, expected_eps0), f"Mismatch for eps=0: {bin_eps0}"

    # 2. Test epsilon = 1:
    # k=0..6: v_c1[k] < v_c2[k+1] -> False (0)
    # k=7: v_c1[7] = 80, v_c2[(7+1)%8] = v_c2[0] = 15 -> 80 >= 15 -> True (1)
    bin_eps1 = binarize_cell_pair(c1, c2, shifts=[1])
    expected_eps1 = np.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    assert np.array_equal(bin_eps1, expected_eps1), f"Mismatch for eps=1: {bin_eps1}"

    # 3. Test epsilon = 7:
    # k=0: v_c1[0]=10, v_c2[(0+7)%8]=v_c2[7]=85 -> 10>=85 -> False (0)
    # k=1..7: v_c1[k] >= v_c2[k-1] (e.g. 20>=15, 30>=25, ..., 80>=75) -> True (1)
    bin_eps7 = binarize_cell_pair(c1, c2, shifts=[7])
    expected_eps7 = np.array([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    assert np.array_equal(bin_eps7, expected_eps7), f"Mismatch for eps=7: {bin_eps7}"


def test_shifted_histograms_cyclic():
    c = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    shifted = generate_shifted_histograms(c, shifts=list(range(8)))

    assert shifted.shape == (8, 8)
    # Shift 0: identical
    assert np.array_equal(shifted[0], c)
    # Shift 1: [2, 3, 4, 5, 6, 7, 8, 1]
    assert np.array_equal(shifted[1], np.array([2, 3, 4, 5, 6, 7, 8, 1]))
    # Shift 7: [8, 1, 2, 3, 4, 5, 6, 7]
    assert np.array_equal(shifted[7], np.array([8, 1, 2, 3, 4, 5, 6, 7]))


def test_centered_derivative_mask_gradients():
    # Constant image -> gradients must be zero
    const_img = np.full((32, 32), 100, dtype=np.uint8)
    lx, ly, mag, ori = compute_gradients_centered_mask(const_img)

    assert np.allclose(lx, 0.0)
    assert np.allclose(ly, 0.0)
    assert np.allclose(mag, 0.0)


def test_extract_improved_hog_parameters_and_types():
    # 24x64 standard ocular patch (3x8 cells)
    patch = np.random.randint(0, 255, (24, 64), dtype=np.uint8)
    res = extract_improved_hog(patch)

    assert res["cell_size"] == [8, 8]
    assert res["block_size"] == [2, 2]
    assert res["orientation_bins"] == 8
    assert res["orientation_shifts"] == 8
    assert res["num_blocks"] == 14  # (3 - 2 + 1) * (8 - 2 + 1) = 2 * 7 = 14
    assert res["feature_dimension"] > 0
    assert len(res["feature_vector"]) == res["feature_dimension"]

    # Verify binary nature: all values must be strictly 0 or 1
    vec = np.array(res["feature_vector"])
    assert np.isin(vec, [0, 1]).all()
    # Check for NaN and Inf
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()
