"""DriveGuard Improved HOG Feature Extraction Module.
Faithfully implements the 'Binarized Histograms of Shifted Orientations' descriptor
from Bakheet & Al-Hamadi (Brain Sciences 2021, Section 2.3, Equations 1-5).

ALGORITHMIC PIPELINE:
1. 1D centered discrete derivative mask [-1, 0, 1] for horizontal & vertical gradients.
2. Gradient magnitude rho(x, y) and orientation gamma(x, y) computation.
3. Unsigned orientation binning into m=8 angular bins over 8x8 pixel cells.
4. Anti-aliasing soft voting to adjacent bins (Equations 3 & 4).
5. Overlapping 2x2 cell blocks (stride = 1 cell, each cell covered by 4 blocks).
6. Orientation shifting by factor epsilon in {0, 1, ..., 7}.
7. Binarization via cell pair comparison (Equation 5):
   b_{c1, c2}(k, epsilon) = 1 if v_c1(k) >= v_c2((k + epsilon) % 8) else 0.
8. Feature vector assembly and dimensional verification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import (
    BLOCK_OVERLAP_CELLS,
    BLOCK_SIZE_CELLS,
    CELL_PAIRING_STRATEGY,
    CELL_SIZE,
    ORIENTATION_BINS,
    ORIENTATION_SHIFTS,
)


def compute_gradients_centered_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute horizontal and vertical gradients using 1D centered mask [-1, 0, 1].

    Paper Section 2.3:
    - Mask: [-1, 0, 1] (EXACT_FROM_PAPER)
    - Equation 1:
        rho = sqrt(Lx^2 + Ly^2)
        gamma = arctan(Ly / Lx)
    """
    if image is None or image.size == 0:
        raise ValueError("Image cannot be None or empty for gradient computation")

    # Convert to grayscale luminance I
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    # 1D centered derivative kernels [-1, 0, 1]
    kernel_x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)
    kernel_y = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)

    # Convolve with reflective boundary conditions
    lx = cv2.filter2D(gray, cv2.CV_64F, kernel_x, borderType=cv2.BORDER_REFLECT)
    ly = cv2.filter2D(gray, cv2.CV_64F, kernel_y, borderType=cv2.BORDER_REFLECT)

    # Equation 1: Magnitude and direction
    magnitude = np.sqrt(lx**2 + ly**2)

    # Orientation in radians, mapped to unsigned range [0, pi) (0 to 180 degrees)
    # Equation 3 specifies gamma / pi * m, confirming unsigned range [0, pi)
    angle_rad = np.arctan2(ly, lx)  # [-pi, pi]
    angle_rad = np.where(angle_rad < 0, angle_rad + np.pi, angle_rad)
    # Ensure strictly in [0, pi)
    angle_rad = np.where(angle_rad >= np.pi, angle_rad - np.pi, angle_rad)

    return lx, ly, magnitude, angle_rad


def compute_cell_histograms(
    magnitude: np.ndarray,
    orientation: np.ndarray,
    cell_size: Tuple[int, int] = CELL_SIZE,
    num_bins: int = ORIENTATION_BINS,
) -> np.ndarray:
    """Compute 8-bin orientation histograms for each 8x8 cell using soft voting.

    Equations 3 & 4:
    alpha = b + 0.5 - (gamma / pi) * m
    gamma_tilde = (1 - alpha) * rho, gamma_hat = alpha * rho
    """
    h_img, w_img = magnitude.shape
    cell_h, cell_w = cell_size

    n_cells_y = h_img // cell_h
    n_cells_x = w_img // cell_w

    if n_cells_y == 0 or n_cells_x == 0:
        raise ValueError(f"Image dimensions ({w_img}x{h_img}) smaller than cell size {cell_size}")

    # Output grid: (n_cells_y, n_cells_x, num_bins)
    histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float64)

    bin_width = np.pi / num_bins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start = cy * cell_h
            y_end = y_start + cell_h
            x_start = cx * cell_w
            x_end = x_start + cell_w

            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ori = orientation[y_start:y_end, x_start:x_end]

            # Vectorized soft voting for the cell
            # Continuous bin position
            pos = cell_ori / bin_width
            b1 = np.floor(pos).astype(int) % num_bins
            b2 = (b1 + 1) % num_bins

            # Fractional distance to next bin
            # Paper Eq 3: alpha = b + 0.5 - (gamma / pi) * m
            # When gamma is at bin center (b + 0.5)*bin_width, alpha = 0.5
            frac = pos - np.floor(pos)
            alpha = np.clip(1.0 - frac, 0.0, 1.0)

            # Eq 4: weighted votes
            vote_b1 = alpha * cell_mag
            vote_b2 = (1.0 - alpha) * cell_mag

            for b in range(num_bins):
                histograms[cy, cx, b] = np.sum(vote_b1[b1 == b]) + np.sum(vote_b2[b2 == b])

    return histograms


def generate_shifted_histograms(
    cell_hist: np.ndarray,
    shifts: Optional[List[int]] = None,
) -> np.ndarray:
    """Incrementally shift an 8-bin histogram by epsilon in {0, ..., 7}.

    Paper Section 2.3:
    "the orientation bins of the 8-bin HOG histogram created from a single cell
    are incrementally shifted by a factor epsilon (epsilon = 0, 1, ..., 7),
    resulting in a total of eight 8-bin histograms."

    Returns:
        Array of shape (8, 8) where row epsilon is the shifted histogram:
        v_c^{(epsilon)}(k) = v_c((k + epsilon) % 8)
    """
    if shifts is None:
        shifts = ORIENTATION_SHIFTS

    num_bins = len(cell_hist)
    shifted = np.zeros((len(shifts), num_bins), dtype=np.float64)

    for i, eps in enumerate(shifts):
        # Cyclically roll/shift bins by eps
        shifted[i] = np.roll(cell_hist, -eps)

    return shifted


def binarize_cell_pair(
    hist_c1: np.ndarray,
    hist_c2: np.ndarray,
    shifts: Optional[List[int]] = None,
) -> np.ndarray:
    """Implement exact binarization formula from Paper (Equation 5).

    b_{c1, c2}(k, epsilon) = 1 if v_c1(k) >= v_c2((k + epsilon) % 8) else 0.

    Args:
        hist_c1: 8-bin histogram of first cell (shape: [8])
        hist_c2: 8-bin histogram of second cell (shape: [8])
        shifts: list of shifts epsilon (default: 0..7)

    Returns:
        Binary numpy array of shape (len(shifts), 8) with values in {0, 1}.
    """
    if shifts is None:
        shifts = ORIENTATION_SHIFTS

    num_bins = len(hist_c1)
    binary_matrix = np.zeros((len(shifts), num_bins), dtype=np.uint8)

    for eps_idx, eps in enumerate(shifts):
        for k in range(num_bins):
            k_shifted = (k + eps) % num_bins
            if hist_c1[k] >= hist_c2[k_shifted]:
                binary_matrix[eps_idx, k] = 1
            else:
                binary_matrix[eps_idx, k] = 0

    return binary_matrix


def get_block_cell_pairs(pairing_strategy: str = CELL_PAIRING_STRATEGY) -> List[Tuple[int, int]]:
    """Define which pairs of cells (c1, c2) are compared within a 2x2 cell block.

    Block cells indexed as:
    [ c0 (top-left) , c1 (top-right)   ]
    [ c2 (bottom-left), c3 (bottom-right)]

    Pairing Strategies:
    - "ADJACENT_PAIRS": 4 adjacent neighbors (c0, c1), (c2, c3), (c0, c2), (c1, c3)
    - "CANONICAL_6_PAIRS": All 6 unique combinations C(4, 2)
    - "HORIZONTAL_VERTICAL": (c0, c1) and (c0, c2)
    """
    if pairing_strategy == "CANONICAL_6_PAIRS":
        return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elif pairing_strategy == "HORIZONTAL_VERTICAL":
        return [(0, 1), (0, 2)]
    else:
        # Default "ADJACENT_PAIRS": captures horizontal, vertical, and primary diagonal structure
        return [(0, 1), (2, 3), (0, 2), (1, 3)]


def extract_improved_hog(
    eye_roi: np.ndarray,
    cell_size: Tuple[int, int] = CELL_SIZE,
    block_size_cells: Tuple[int, int] = BLOCK_SIZE_CELLS,
    num_bins: int = ORIENTATION_BINS,
    shifts: Optional[List[int]] = None,
    pairing_strategy: str = CELL_PAIRING_STRATEGY,
) -> Dict[str, Any]:
    """Extract complete Improved HOG feature vector from an ocular ROI.

    Follows Bakheet & Al-Hamadi (2021):
    - Cell size: 8x8 pixels (EXACT_FROM_PAPER)
    - Block size: 2x2 cells (EXACT_FROM_PAPER)
    - Blocks partially overlap: each cell covered by 4 blocks (stride=1 cell) (EXACT_FROM_PAPER)
    - 8 orientation bins (EXACT_FROM_PAPER)
    - 8 shifts epsilon in 0..7 (EXACT_FROM_PAPER)
    - Binarization formula Eq. 5 (EXACT_FROM_PAPER)

    Returns dictionary containing:
    - feature_vector: List[int] (pure binary descriptor)
    - feature_dimension: int
    - cell_size: List[int]
    - block_size: List[int]
    - orientation_bins: int
    - orientation_shifts: int
    - num_blocks: int
    - gradient_magnitude: np.ndarray
    - gradient_orientation: np.ndarray
    - cell_histograms: np.ndarray
    - sample_shifted_histograms: np.ndarray
    - sample_binary_matrix: np.ndarray
    """
    if eye_roi is None or eye_roi.size == 0:
        raise ValueError("Input eye_roi cannot be None or empty")

    if shifts is None:
        shifts = ORIENTATION_SHIFTS

    # Step 1 & 2: Centered gradient computation
    lx, ly, magnitude, orientation = compute_gradients_centered_mask(eye_roi)

    # Step 3 & 4: Cell histograms
    histograms = compute_cell_histograms(
        magnitude=magnitude,
        orientation=orientation,
        cell_size=cell_size,
        num_bins=num_bins,
    )

    n_cells_y, n_cells_x, _ = histograms.shape
    block_h, block_w = block_size_cells

    if n_cells_y < block_h or n_cells_x < block_w:
        raise ValueError(
            f"ROI cell grid ({n_cells_y}x{n_cells_x}) smaller than required block size ({block_h}x{block_w})"
        )

    # Overlapping blocks: stride = 1 cell (each internal cell covered by 4 blocks)
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1
    total_blocks = n_blocks_y * n_blocks_x

    pairs = get_block_cell_pairs(pairing_strategy)

    feature_bits: List[int] = []
    sample_binary_matrix: Optional[np.ndarray] = None
    sample_shifted: Optional[np.ndarray] = None

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # Extract the 4 cells in the 2x2 block
            # c0: (by, bx), c1: (by, bx+1), c2: (by+1, bx), c3: (by+1, bx+1)
            c0 = histograms[by, bx]
            c1 = histograms[by, bx + 1]
            c2 = histograms[by + 1, bx]
            c3 = histograms[by + 1, bx + 1]
            block_cells = [c0, c1, c2, c3]

            # Save shifted sample from first cell of first block for visualization
            if sample_shifted is None:
                sample_shifted = generate_shifted_histograms(c0, shifts)

            # Compare cell pairs inside the block
            for idx_1, idx_2 in pairs:
                bin_mat = binarize_cell_pair(block_cells[idx_1], block_cells[idx_2], shifts)
                if sample_binary_matrix is None:
                    sample_binary_matrix = bin_mat

                # Flatten and append binary descriptor bits
                feature_bits.extend(bin_mat.flatten().tolist())

    return {
        "feature_vector": feature_bits,
        "feature_dimension": len(feature_bits),
        "cell_size": list(cell_size),
        "block_size": list(block_size_cells),
        "orientation_bins": num_bins,
        "orientation_shifts": len(shifts),
        "num_blocks": total_blocks,
        "pairing_strategy": pairing_strategy,
        "pairs_per_block": len(pairs),
        "gradient_magnitude": magnitude,
        "gradient_orientation": orientation,
        "cell_histograms": histograms,
        "sample_shifted_histograms": sample_shifted,
        "sample_binary_matrix": sample_binary_matrix,
    }
