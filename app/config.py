"""DriveGuard Configuration Module.
Centralizes all hyperparameters, file paths, and algorithm settings.

Every parameter is tagged according to its provenance:
- EXACT_FROM_PAPER: Directly defined in Bakheet & Al-Hamadi (Brain Sciences 2021).
- NOT_SPECIFIED_IN_PAPER: Required for implementation but absent from paper text.
- TEMPORARY_ASSUMPTION: Default value adopted for prototype; subject to tuning.
- APPROXIMATION: Practical substitute for components lacking published model weights.
- APPROVED_IMPROVISATION: User-requested extension or fallback mode.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PREPROCESSED_DIR = OUTPUTS_DIR / "preprocessed"
FACE_DETECTION_DIR = OUTPUTS_DIR / "face_detection"
EYE_ROI_DIR = OUTPUTS_DIR / "eye_roi"
HOG_VIS_DIR = OUTPUTS_DIR / "hog_visualization"
FEATURES_DIR = OUTPUTS_DIR / "features"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure output directories exist
for _dir in [PREPROCESSED_DIR, FACE_DETECTION_DIR, EYE_ROI_DIR, HOG_VIS_DIR, FEATURES_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------------------------------
# Dataset paths: check root NTHU-DDD or dataset/NTHU-DDD
DATASET_CANDIDATE_PATHS: List[Path] = [
    BASE_DIR / "NTHU-DDD",
    BASE_DIR / "dataset" / "NTHU-DDD",
]

def get_dataset_root() -> Path:
    """Resolve available dataset directory."""
    for p in DATASET_CANDIDATE_PATHS:
        if p.exists() and p.is_dir():
            return p
    # Fallback to default expected path
    return BASE_DIR / "dataset" / "NTHU-DDD"

# Class Label Mapping (EXACT_FROM_PAPER / Dataset specification)
CLASS_LABELS: Dict[str, int] = {
    "drowsy": 1,
    "notdrowsy": 0,
}
REVERSE_CLASS_LABELS: Dict[int, str] = {v: k for k, v in CLASS_LABELS.items()}

# Supported file extensions
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv")


# ---------------------------------------------------------------------------
# PREPROCESSING PARAMETERS
# ---------------------------------------------------------------------------
# Gaussian Blur filter (Section 2.1, page 4)
# Paper: "convolved with a 2D Gaussian blur filter over a 3 x 3 pixel neighborhood
# and uniform standard deviation of 0.5"
GAUSSIAN_KERNEL_SIZE: Tuple[int, int] = (3, 3)  # EXACT_FROM_PAPER
GAUSSIAN_SIGMA: float = 0.5                     # EXACT_FROM_PAPER

# CLAHE (Section 2.1, page 5)
# Paper: "each color channel is independently equalized"
# Note: clipLimit and tileGridSize are NOT mentioned in the paper.
CLAHE_PER_CHANNEL: bool = True                  # EXACT_FROM_PAPER
CLAHE_CLIP_LIMIT: float = 2.0                   # TEMPORARY_ASSUMPTION (OpenCV standard default)
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)  # TEMPORARY_ASSUMPTION (Standard 8x8 grid)

# Resolution Reduction (Section 2.1, page 5)
# Paper: "the resolution of the light-compensated image can be reduced to increase the computational efficiency"
# Note: Paper does not specify resolution numbers.
RESIZE_ENABLED: bool = False                    # TEMPORARY_ASSUMPTION (Default: preserve native 640x480)
RESIZE_TARGET: Tuple[int, int] = (320, 240)     # TEMPORARY_ASSUMPTION (Used only if RESIZE_ENABLED is True)


# ---------------------------------------------------------------------------
# FACE DETECTION PARAMETERS
# ---------------------------------------------------------------------------
# Paper (Section 2.2, page 5): "improved cascaded adaBoost classifier based on an extended set of Haar-like features"
# Note: Authors trained a custom Haar cascade in C++ / OpenCV 4.1.2. Model weights were not published.
# In NTHU-DDD dashboard camera (640x480), the driver's face spans ~250x250 pixels.
# Setting minSize to (120, 120) strictly eliminates false positive detections in the mouth/collar region.
FACE_DETECTION_MODE: str = "APPROXIMATION_HAAR"  # APPROXIMATION
FACE_CASCADE_FILENAME: str = "haarcascade_frontalface_default.xml"
FACE_DETECTION_SCALE_FACTOR: float = 1.05       # TEMPORARY_ASSUMPTION (Optimized for close-up dashboard camera)
FACE_DETECTION_MIN_NEIGHBORS: int = 3          # TEMPORARY_ASSUMPTION
FACE_DETECTION_MIN_SIZE: Tuple[int, int] = (100, 100)  # Eliminates mouth/collar false positives


# ---------------------------------------------------------------------------
# EYE LOCALIZATION PARAMETERS
# ---------------------------------------------------------------------------
# Paper (Section 2.2, page 6): "improved active shape model (ASM) algorithm based on statistical learning models"
# Note: ASM landmark annotations, PCA components, cost function, and trained weights were NOT published.
# Modes:
# 1. "APPROXIMATION_HAAR": Uses OpenCV Haar eye cascade / anthropometric upper-face bounding box.
# 2. "MANUAL_ROI": Accepts coordinates drawn on the web UI canvas.
# 3. "ASM_UNAVAILABLE": Explicitly signals that ASM model weights are missing.
EYE_LOCALIZATION_MODE: str = "APPROXIMATION_HAAR"  # APPROXIMATION
EYE_CASCADE_FILENAME: str = "haarcascade_eye.xml"
EYE_FALLBACK_TO_UPPER_FACE: bool = True  # Use upper 1/3 face anthropometric box if eye cascade misses


# ---------------------------------------------------------------------------
# EYE ROI EXTRACTION PARAMETERS
# ---------------------------------------------------------------------------
# The ROI must accommodate 8x8 cells and 2x2 cell blocks.
# Minimum spatial dimension is 16x16 (1 block).
# Standardized default eye pair crop: (height=24, width=64) gives 3x8 = 24 cells.
STANDARDIZE_ROI_SIZE: bool = True               # TEMPORARY_ASSUMPTION
TARGET_ROI_HEIGHT: int = 24                     # TEMPORARY_ASSUMPTION (3 cells of 8px)
TARGET_ROI_WIDTH: int = 64                      # TEMPORARY_ASSUMPTION (8 cells of 8px)


# ---------------------------------------------------------------------------
# IMPROVED HOG PARAMETERS
# ---------------------------------------------------------------------------
# Paper (Section 2.3, pages 6-8)
CELL_SIZE: Tuple[int, int] = (8, 8)             # EXACT_FROM_PAPER
BLOCK_SIZE_CELLS: Tuple[int, int] = (2, 2)      # EXACT_FROM_PAPER
BLOCK_OVERLAP_CELLS: int = 1                    # EXACT_FROM_PAPER (each cell covered by 4 blocks -> stride=1)
ORIENTATION_BINS: int = 8                       # EXACT_FROM_PAPER
ORIENTATION_RANGE: Tuple[float, float] = (0.0, 180.0)  # EXACT_FROM_PAPER (Unsigned [0, pi))
ORIENTATION_SHIFTS: List[int] = list(range(8)) # EXACT_FROM_PAPER: epsilon in {0, 1, ..., 7}
GRADIENT_KERNEL_X: List[int] = [-1, 0, 1]       # EXACT_FROM_PAPER

# Cell Pairing Mode inside each 2x2 block
# In a block of 4 cells [c0, c1; c2, c3]:
# "ADJACENT_PAIRS": (c0, c1), (c0, c2), (c1, c3), (c2, c3), (c0, c3)
# "CANONICAL_PAIRS": 4 directed pairings (c0->c1, c1->c3, c3->c2, c2->c0)
# "ALL_COMBINATIONS": 6 combinations (c0,c1), (c0,c2), (c0,c3), (c1,c2), (c1,c3), (c2,c3)
CELL_PAIRING_STRATEGY: str = "ADJACENT_PAIRS"   # TEMPORARY_ASSUMPTION / CONFIGURABLE
