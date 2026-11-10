# DriveGuard: Scientific Decisions Required (M12–M13)

This document catalogs all algorithmic details and parameter choices that are **omitted or under-specified** in the reference paper by Bakheet & Al-Hamadi (*Brain Sciences*, MDPI, 2021). 

In accordance with academic rigor, each item below outlines:
1. What the paper omitted.
2. Why the detail is required in software implementation.
3. Plausible technical alternatives.
4. Methodological impact on research validity.
5. Temporary recommendation adopted in the prototype.
6. Explicit question requiring user confirmation.

---

### Decision 1: Custom Cascaded AdaBoost Classifier vs OpenCV Pretrained Cascade
1. **Omitted Detail:** The exact trained cascade XML weights, number of waterfall stages, weak learner count, and negative background dataset used by the authors were not published.
2. **Why Needed:** Face detection is the mandatory prerequisite stage for eye localization.
3. **Alternatives:**
   - *Alt A:* Train a bespoke AdaBoost cascade from scratch on facial datasets (computationally heavy, requires thousands of positive/negative crops).
   - *Alt B:* Use OpenCV's standardized `haarcascade_frontalface_default.xml` or `haarcascade_frontalface_alt2.xml` as an explicit **APPROXIMATION**.
4. **Methodological Impact:** Alt B uses identical Viola-Jones Haar features and AdaBoost waterfall logic, but decision boundaries differ slightly from the author's custom C++ model.
5. **Temporary Recommendation:** Adopt Alt B labeled transparently as `APPROXIMATION_HAAR`.
6. **Question for User:** *Do you approve using OpenCV's pretrained Haar cascade as the interim face detector approximation, pending any custom cascade training in future milestones?*

---

### Decision 2: Eye Localization (Active Shape Model vs Haar / Anthropometric Approximation vs Manual ROI)
1. **Omitted Detail:** The paper references an "Improved Active Shape Model (ASM)", but omits the landmark point schema (e.g. 68-point vs 17-point), manual annotation training set, PCA component count, cost function formula, iteration bounds, and model weight file.
2. **Why Needed:** An automated eye detector is required to crop the eye-pair ROI from the face image.
3. **Alternatives:**
   - *Alt A:* Seek a third-party ASM C++ library and train it on external face datasets (e.g. Helen/300-W). High engineering overhead, risk of divergence from author's proprietary scheme.
   - *Alt B:* Provide an automated approximation using OpenCV Haar eye cascade + anatomical upper-face band heuristic, explicitly labeled `APPROXIMATION`.
   - *Alt C:* Provide an interactive Manual ROI Canvas selection in the web interface to isolate the eye region without detector bias.
4. **Methodological Impact:** Neither Alt B nor Alt C is ASM. However, Alt C allows pristine evaluation of the primary contribution (Improved HOG) independent of eye-detection errors.
5. **Temporary Recommendation:** Combine Alt B (default automated fallback) with Alt C (manual selection toggle in the web interface), while logging ASM as `NOT_IMPLEMENTED_PENDING_WEIGHTS`.
6. **Question for User:** *Do you approve combining the Haar/anthropometric approximation with the interactive manual canvas ROI selection for testing Improved HOG?*

---

### Decision 3 & 4: CLAHE Numerical Hyperparameters (`clipLimit` & `tileGridSize`)
1. **Omitted Detail:** The paper explicitly states that CLAHE is applied to each color channel independently (Section 2.1), but completely omits numerical values for `clipLimit` and `tileGridSize`.
2. **Why Needed:** `cv2.createCLAHE()` requires both values to partition the image and clip histogram redistribution.
3. **Alternatives:**
   - *Alt A:* Standard defaults: `clipLimit=2.0`, `tileGridSize=(8, 8)`.
   - *Alt B:* Conservative values: `clipLimit=1.2`, `tileGridSize=(16, 16)`.
   - *Alt C:* Aggressive contrast enhancement: `clipLimit=4.0`, `tileGridSize=(4, 4)`.
4. **Methodological Impact:** Overly aggressive clipping amplifies camera sensor noise in dark night-time IR frames of NTHU-DDD; conservative clipping preserves natural eyelid edges.
5. **Temporary Recommendation:** Adopt Alt A (`clipLimit=2.0`, `tileGridSize=(8, 8)`) as modular constants in `app/config.py`.
6. **Question for User:** *Are you satisfied with the default clipLimit=2.0 and tileGridSize=(8, 8), or would you prefer a parameter tuning sweep across day/night driving subsets?*

---

### Decision 5: Resolution Reduction / Image Resize Dimensions
1. **Omitted Detail:** Section 2.1 notes: *"the resolution of the light-compensated image can be reduced to increase the computational efficiency"*, but gives no target spatial dimensions.
2. **Why Needed:** If resizing is performed, explicit target width and height must be specified.
3. **Alternatives:**
   - *Alt A:* Disable resizing (`RESIZE_ENABLED = False`) and retain native NTHU-DDD resolution ($640 \times 480$).
   - *Alt B:* Downscale by half to $320 \times 240$.
4. **Methodological Impact:** Downscaling reduces facial feature resolution and makes eye blink detection harder on distance frames.
5. **Temporary Recommendation:** Keep `RESIZE_ENABLED = False` (Alt A) by default.
6. **Question for User:** *Should we preserve the native 640x480 resolution throughout the pipeline, or enable downscaling to 320x240 for constrained embedded performance?*

---

### Decision 6: Color Space for Luminance Gradient Computation
1. **Omitted Detail:** Equation 2 mentions first-order Gaussian derivatives of "image patch luminance $I$". It does not specify whether $I$ is extracted via standard ITU-R 601 grayscale conversion ($0.299R + 0.587G + 0.114B$) or YCrCb luma channel $Y$.
2. **Why Needed:** Improved HOG calculates 1D derivatives on a single 2D intensity plane.
3. **Alternatives:**
   - *Alt A:* Standard OpenCV `cv2.COLOR_BGR2GRAY`.
   - *Alt B:* $Y$ channel of `cv2.COLOR_BGR2YCrCb`.
4. **Methodological Impact:** Negligible numerical difference (<1% intensity variance across standard integer coefficients).
5. **Temporary Recommendation:** Use standard grayscale `cv2.COLOR_BGR2GRAY` (Alt A).
6. **Question for User:** *Do you approve using standard OpenCV BGR-to-Grayscale conversion for luminance $I$?*

---

### Decision 7: HOG Orientation Range (Unsigned $[0, \pi)$ vs Signed $[0, 2\pi)$)
1. **Omitted Detail:** Section 2.3 states 8 angular bins are used, but does not explicitly state whether the range is signed ($0^\circ$ to $360^\circ$) or unsigned ($0^\circ$ to $180^\circ$).
2. **Why Needed:** Defines bin width $\Delta \theta$ and angle normalization.
3. **Paper Evidence:** Equation 3 gives the pixel bin weighting formula as:
   $$\alpha = b + 0.5 - \frac{\gamma}{\pi} m$$
   The explicit presence of $\pi$ in the denominator with $m=8$ mathematically proves that $\gamma \in [0, \pi)$!
4. **Methodological Impact:** Unsigned gradient orientation $[0, \pi)$ is robust to light-source polarity inversions (dark-to-light vs light-to-dark edges across diverse skin tones).
5. **Recommendation:** Standardize strictly on unsigned $[0, \pi)$ (`EXACT_FROM_PAPER` mathematical derivation).

---

### Decision 8: Cell Pairing Strategy Inside Overlapping 2x2 Blocks
1. **Omitted Detail:** Equation 5 defines binarization between two cell regions $c_1$ and $c_2$:
   $$b_{c_1 c_2}(k, \varepsilon) = 1 \text{ if } v_{c_1}(k) \ge v_{c_2}((k + \varepsilon) \bmod 8) \text{ else } 0$$
   Figure 6 illustrates "1st Cell" and "2nd Cell", but the paper does not explicitly state which cell pairs within each $2 \times 2$ block are compared.
2. **Why Needed:** A $2 \times 2$ block contains 4 cells $\{c_0, c_1, c_2, c_3\}$. We must define which pairs $(c_i, c_j)$ are evaluated.
3. **Alternatives:**
   - *Alt A (Adjacent 4 Pairs):* $(c_0, c_1)$ horizontal top, $(c_2, c_3)$ horizontal bottom, $(c_0, c_2)$ vertical left, $(c_1, c_3)$ vertical right.
   - *Alt B (All 6 Combinations):* All $\binom{4}{2} = 6$ pairs in the block.
   - *Alt C (Directed Cycle):* $c_0 \to c_1 \to c_3 \to c_2 \to c_0$.
4. **Methodological Impact:** Directly determines the feature vector dimensionality. For an ocular patch of $24 \times 64$ pixels (14 blocks):
   - Alt A yields $14 \times 4 \times 64 = 3,584$ bits.
   - Alt B yields $14 \times 6 \times 64 = 5,376$ bits.
5. **Temporary Recommendation:** Adopt Alt A (`ADJACENT_PAIRS`) as the configurable default in `app/config.py`.
6. **Question for User:** *Do you agree with the adjacent pairing strategy (4 pairs per block), or would you prefer evaluating all 6 combinations or a single representative diagonal pair?*

---

### Decision 9: Standardized Eye ROI Dimensions
1. **Omitted Detail:** The paper displays ocular crops in Figures 4, 5, and 7, but omits the exact pixel dimensions of the cropped patch before HOG cell pooling.
2. **Why Needed:** HOG requires cell partitioning where width and height are divisible by 8 and at least $16 \times 16$ pixels (1 block).
3. **Alternatives:**
   - *Alt A:* Fixed standardized size of $24 \times 64$ pixels ($3 \times 8$ cells).
   - *Alt B:* Fixed standardized size of $32 \times 80$ pixels ($4 \times 10$ cells).
   - *Alt C:* Dynamic cropping with nearest-multiple-of-8 padding without resizing.
4. **Methodological Impact:** Standardized rectangular ROI (Alt A) guarantees identical feature dimension across all frames, which is required for training Naïve Bayes.
5. **Temporary Recommendation:** Adopt Alt A ($24 \times 64$ pixels).
6. **Question for User:** *Do you approve standardizing the eye-pair ROI crop to 24x64 pixels (3x8 cells) to ensure uniform feature vector dimensionality?*

---

### Decision 10: Video Frame Labeling & Temporal Leakage Guard
1. **Omitted Detail:** The NTHU-DDD dataset was recorded as continuous video streams under various driving scenarios. In video sequences labeled `drowsy` (e.g. `sleepyCombination`), not every individual frame represents an eye-closure state (e.g. driver blinks intermittently).
2. **Why Needed:** Supervised learning requires accurate ground truth labels.
3. **Alternatives:**
   - *Alt A:* Strict folder label assignment (Folder `drowsy` = 1, `notdrowsy` = 0) as required by project assignment.
   - *Alt B:* Fine-grained PERCLOS frame annotation (requires manual eye-closure ground truth).
4. **Leakage Guard:** Frames from the same subject/video must NEVER be split randomly between train and test sets. We have parsed `subject_id` and `scenario` in `app/dataset_loader.py` to enable subject-independent splitting.
5. **Recommendation:** Maintain strict folder labels (Alt A) while preserving subject metadata.
6. **Question for User:** *When Raditya proceeds with Naïve Bayes training, do you plan to use Subject-Independent K-Fold Cross Validation (Leave-Subjects-Out) to avoid video frame leakage?*
