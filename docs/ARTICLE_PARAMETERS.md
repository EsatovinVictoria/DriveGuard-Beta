# DriveGuard: Scientific Article Parameter Mapping

Reference Paper:
**"A Framework for Instantaneous Driver Drowsiness Detection Based on Improved HOG Features and Naïve Bayesian Classification"**
Authors: Samy Bakheet and Ayoub Al-Hamadi
Journal: *Brain Sciences*, MDPI, 2021, Vol. 11, Iss. 2, Art. 240.
DOI: [10.3390/brainsci11020240](https://doi.org/10.3390/brainsci11020240)

---

## Parameter Mapping Table

| Parameter / Algorithmic Element | Value / Setting | Paper Section & Page | Status | Description & Rationale |
|---|---|---|---|---|
| **Gaussian Blur Neighborhood** | `3 x 3` pixels | Section 2.1, Page 4 | `EXACT_FROM_PAPER` | Explicitly defined 2D convolution kernel to suppress disturbing noise while preserving structure. |
| **Gaussian Blur Sigma (&sigma;)** | `0.5` | Section 2.1, Page 4 | `EXACT_FROM_PAPER` | Uniform standard deviation for the 2D Gaussian filter. |
| **CLAHE Independent Channels** | `True` (Per color channel) | Section 2.1, Page 5 | `EXACT_FROM_PAPER` | Equalization applied independently to each color channel for lighting compensation. |
| **CLAHE clipLimit** | `2.0` | Section 2.1, Page 5 | `TEMPORARY_ASSUMPTION` | Paper omits numerical clipLimit. Standard OpenCV default adopted. |
| **CLAHE tileGridSize** | `(8, 8)` | Section 2.1, Page 5 | `TEMPORARY_ASSUMPTION` | Paper omits contextual grid partitioning. Standard 8x8 adopted. |
| **Image Resolution Reduction** | `Disabled (False)` | Section 2.1, Page 5 | `TEMPORARY_ASSUMPTION` | Paper states resolution can be reduced for efficiency, but omits target numbers. Kept at native 640x480. |
| **Face Detector Algorithm** | Cascaded AdaBoost + Haar | Section 2.2, Page 5 | `APPROXIMATION` | Paper used custom trained Haar cascade in C++ / OpenCV 4.1.2. Pretrained OpenCV frontalface cascade used as approximation. |
| **Eye Localization Algorithm** | Active Shape Model (ASM) | Section 2.2, Page 5-6 | `APPROXIMATION` / `NOT_SPECIFIED` | Paper trained custom statistical shape model (PCA + iterative cost). No landmarks or model weights published. OpenCV Haar/anthropometric upper-face box used as approximation; manual canvas selection provided. |
| **HOG Cell Size** | `8 x 8` pixels | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | Spatial pooling unit over which orientation histograms are constructed. |
| **HOG Block Size** | `2 x 2` cells (`16 x 16` px) | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | Fundamental computation block grouping 4 adjacent cells. |
| **HOG Block Overlap** | Stride = 1 cell (4 blocks/cell) | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | Blocks partially overlap so that each interior cell is covered by 4 blocks. |
| **Gradient Derivative Mask** | `[-1, 0, 1]` | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | 1D centered point discrete derivative mask for horizontal and vertical directions. |
| **Gradient Orientation Bins** | `8` bins | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | Angular quantization into 8 bins ($m=8$). |
| **Gradient Orientation Range** | `[0, pi)` unsigned | Section 2.3, Page 6, Eq. 3 | `EXACT_FROM_PAPER` | Unsigned orientation $[0, \pi)$ indicated by denominator $\pi$ in Eq. 3 ($\alpha = b + 0.5 - \frac{\gamma}{\pi} m$). |
| **Anti-aliasing Soft Voting** | Equations 3 & 4 | Section 2.3, Page 6 | `EXACT_FROM_PAPER` | Bilinear interpolation of votes across adjacent bins based on distance $\alpha$. |
| **Orientation Shifts (&epsilon;)** | `0, 1, 2, 3, 4, 5, 6, 7` | Section 2.3, Page 6-7 | `EXACT_FROM_PAPER` | 8 cyclic circular shifts applied to 8-bin histograms. |
| **Binarization Formula** | $b_{c1,c2}(k,\varepsilon) = \mathbb{I}[v_{c1}(k) \ge v_{c2}((k+\varepsilon)\%8)]$ | Section 2.3, Page 7, Eq. 5 | `EXACT_FROM_PAPER` | Direct step function comparing cell histogram bins with shifted orientation bins. |
| **Block Cell Pairing Strategy** | Adjacent pairs in 2x2 block | Section 2.3, Page 7 | `TEMPORARY_ASSUMPTION` | Paper specifies comparing two cell regions $c_1, c_2$ within blocks, but does not enumerate which specific pairs. Configurable. |
| **Standardized Eye ROI Size** | `24 x 64` pixels (3x8 cells) | Section 2.3 | `TEMPORARY_ASSUMPTION` | Ensures integer multiples of 8x8 cells and valid 2x2 overlapping blocks. |
| **Class Label Encoding** | `drowsy: 1`, `notdrowsy: 0` | Section 3, Page 11-13 | `EXACT_FROM_PAPER` | Strict ground truth assignment from dataset directory labels. |
| **Feature Vector Normalization** | None ($L_0$ binary bits) | Section 2.3, Page 7 | `EXACT_FROM_PAPER` | Eq. 5 outputs pure binary bits $\{0, 1\}$, obviating $L_1/L_2$ norm. |

---

## Status Classification Legend

- **EXACT_FROM_PAPER**: Value, formula, or behavior is explicitly written in the MDPI article text or mathematical equations.
- **NOT_SPECIFIED_IN_PAPER**: Necessary parameter or procedure that was completely omitted by the authors.
- **TEMPORARY_ASSUMPTION**: Reasonable default value chosen for the experimental prototype; fully configurable in `app/config.py`.
- **APPROXIMATION**: Standard library equivalent or heuristic implemented because author's proprietary model weights are unavailable.
- **APPROVED_IMPROVISATION**: Feature or interface extension designed to facilitate academic experimentation (e.g. manual canvas ROI selector).
