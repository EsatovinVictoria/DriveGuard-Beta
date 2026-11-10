# DriveGuard Implementation Notes (M12–M13)

This technical note documents the implementation mechanics, theoretical principles, interface contracts, and experimental verification of the DriveGuard computer vision pipeline, following Bakheet & Al-Hamadi (MDPI *Brain Sciences*, 2021).

---

## 1. Image Preprocessing

### Theory
Raw dashboard camera frames exhibit illumination fluctuation, thermal sensor noise, and vehicle vibration artifacts. Bakheet & Al-Hamadi specify a two-step conditioning sequence:
1. **2D Gaussian Filtering:** Suppresses high-frequency camera noise with a compact $3 \times 3$ kernel and small variance $\sigma = 0.5$ to avoid blurring subtle eyelid contour edges.
2. **Channel-Independent CLAHE:** Equalizes localized contrast dynamically across tiles without over-amplifying uniform background regions, applied across all color channels independently.

### Code Implementation
- **Module:** `app/preprocessing.py`
- **Primary Functions:** `apply_gaussian_blur()`, `apply_clahe_independent()`, `preprocess_image()`
- **Input:** Raw BGR image `np.ndarray` (uint8, typical size: $480 \times 640 \times 3$).
- **Output:** Dictionary containing `original`, `gaussian_blur`, `clahe`, and `preprocessed` arrays.

### Parameters
- `GAUSSIAN_KERNEL_SIZE = (3, 3)` (*EXACT_FROM_PAPER*)
- `GAUSSIAN_SIGMA = 0.5` (*EXACT_FROM_PAPER*)
- `CLAHE_CLIP_LIMIT = 2.0` (*TEMPORARY_ASSUMPTION* — configurable in `app/config.py`)
- `CLAHE_TILE_GRID_SIZE = (8, 8)` (*TEMPORARY_ASSUMPTION* — configurable in `app/config.py`)
- `RESIZE_ENABLED = False` (*TEMPORARY_ASSUMPTION*)

### Differences from Paper
The authors state that each color channel is independently equalized. In OpenCV, this is achieved by splitting the image into B, G, and R channels, invoking `clahe.apply()` per channel, and re-merging via `cv2.merge()`. Numerical limits (`clipLimit` and `tileGridSize`) omitted in the paper are exposed as modular constants.

---

## 2. Face Detection

### Theory
Driver faces appear across variable scales and positions. The paper adopts an **Improved Cascaded AdaBoost Classifier** utilizing extended Haar-like features:
- Edge features (two-rectangle)
- Line features (three-rectangle)
- Center-surrounded features (four-rectangle surround)
- Diagonal features (tilted line detectors)

Weak classifiers are trained with AdaBoost and cascaded in a waterfall rejection architecture to rapidly reject background patches.

### Code Implementation
- **Module:** `app/face_detection.py`
- **Primary Class / Function:** `FaceDetector`, `detect_face()`
- **Input:** Preprocessed BGR frame `np.ndarray`.
- **Output:** Structured dictionary:
  ```python
  {
      "face_detected": bool,
      "face_box": {"x": int, "y": int, "width": int, "height": int},
      "method": "APPROXIMATION_HAAR_CASCADE",
      "method_status": "APPROXIMATION",
      "cropped_face": np.ndarray,
      "annotated_image": np.ndarray
  }
  ```

### Differences from Paper
The paper's authors trained custom cascade stages within Microsoft Visual Studio 2015 and OpenCV 4.1.2 without publishing the cascade XML file or feature weights. This prototype implements OpenCV's pretrained frontalface Haar cascade as an **`APPROXIMATION`**. If face detection fails, the pipeline cleanly returns `face_detected: False` without passing empty or invalid matrices to downstream stages.

---

## 3. Eye Localization

### Theory
The paper utilizes an **Improved Active Shape Model (ASM)**:
1. Statistical point distribution models (PDMs) trained on manually labeled landmark points.
2. Principal Component Analysis (PCA) models primary modes of non-rigid facial deformation.
3. An iterative profile matching scheme optimizes a cost function to snap contour points to eye boundaries.

### Code Implementation
- **Module:** `app/eye_localization.py`
- **Primary Function:** `locate_eyes(face_image, mode, manual_box)`
- **Supported Modes:**
  1. `APPROXIMATION_HAAR`: Dual-eye cascade detection with upper-face anthropometric fallback.
  2. `MANUAL_ROI`: Allows the user on the web interface canvas to interactively draw a precise bounding box around the driver's eyes.
  3. `ASM_MODEL`: Structural placeholder returning `NOT_IMPLEMENTED` with an explanatory message.

### Differences from Paper
The paper omitted all ASM parameters (point count, PCA eigenvalues, cost function weights, convergence criteria). Rather than misrepresenting a third-party keypoint detector as the paper's ASM, our architecture clearly separates automated approximation from manual testing mode.

---

## 4. Eye Region of Interest (ROI) Extraction

### Theory
The localized eye region must be isolated and dimensionally conditioned so that it can be partitioned into $8 \times 8$ cells and overlapping $2 \times 2$ cell blocks without fractional pixel boundaries.

### Code Implementation
- **Module:** `app/roi_extraction.py`
- **Primary Functions:** `validate_and_clamp_bbox()`, `extract_eye_roi()`
- **Input:** Cropped face image `np.ndarray`, bounding box dictionary.
- **Output:** Dictionary containing `raw_roi`, `processed_roi`, `dimensions`, and status.
- **Standardized Size:** Height = 24 px (3 cells), Width = 64 px (8 cells).
- **Cell Grid:** $3 \times 8 = 24$ total cells, yielding $(3 - 2 + 1) \times (8 - 2 + 1) = 2 \times 7 = 14$ overlapping blocks.

---

## 5. Improved HOG Feature Extraction

### Theory & Mathematical Formulation
Standard HOG (Dalal & Triggs, 2005) extracts gradient orientation histograms normalized across blocks. Bakheet & Al-Hamadi introduce **Binarized Histograms of Shifted Orientations**:

1. **Discrete Derivative Filtering:**
   Convolve luminance $I$ with 1D centered mask $[-1, 0, 1]$:
   $$\rho(x, y) = \sqrt{L_x(x, y)^2 + L_y(x, y)^2}$$
   $$\gamma(x, y) = \arctan\left(\frac{L_y(x, y)}{L_x(x, y)}\right) \in [0, \pi)$$

2. **Angular Quantization & Anti-Aliasing Soft Voting:**
   Quantized into $m=8$ bins ($\Delta \theta = 22.5^\circ$). To prevent boundary aliasing:
   $$\alpha = b + 0.5 - \frac{\gamma}{\pi} m$$
   $$\tilde{\gamma} = (1 - \alpha) \rho, \quad \hat{\gamma} = \alpha \rho$$
   Votes are accumulated into 8-bin vector $v_c$ per $8 \times 8$ cell.

3. **Orientation Shifting:**
   For each cell histogram $v_c$, cyclically shift bins by factor $\varepsilon \in \{0, 1, \dots, 7\}$:
   $$v_c^{(\varepsilon)}(k) = v_c((k + \varepsilon) \bmod 8)$$

4. **Binarization Formula (Equation 5):**
   $$b_{c_1 c_2}(k, \varepsilon) = \begin{cases} 1, & \text{if } v_{c_1}(k) \ge v_{c_2}((k + \varepsilon) \bmod 8) \\ 0, & \text{otherwise} \end{cases}$$

5. **Compact Binary Descriptor:**
   Features are pure bits $\{0, 1\}$. No float normalization is needed, rendering the descriptor compact and immune to monotonic illumination scale shifts.

### Code Implementation
- **Module:** `app/improved_hog.py`
- **Primary Functions:**
  - `compute_gradients_centered_mask()`
  - `compute_cell_histograms()`
  - `generate_shifted_histograms()`
  - `binarize_cell_pair()`
  - `extract_improved_hog()`
- **Unit Verification:** Verified via deterministic synthetic truth table in `tests/test_improved_hog.py`.

---

## 6. Feature Export for Naïve Bayes Handoff

### Purpose
Prepares tabulated data files in `outputs/features/` for subsequent Bayesian classification:
- `hog_features.csv`: Header row with `sample_id, source_file, subject_id, label, label_id, hog_0000, ...`
- `feature_metadata.csv`: Extraction timing, scenario, subject ID, and dimensions.
- `extraction_report.json`: Full audit log of successful and failed extractions.
