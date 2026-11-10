# DriveGuard — Computer Vision & Improved HOG Experimental Prototype (M12–M13)

Academic Prototype for **DriveGuard: Real-Time Driver Drowsiness Detection System**.

Implements the computer vision feature engineering stages of:
> **"A Framework for Instantaneous Driver Drowsiness Detection Based on Improved HOG Features and Naïve Bayesian Classification"**  
> *Samy Bakheet and Ayoub Al-Hamadi (Brain Sciences, MDPI, 2021)*  
> DOI: [10.3390/brainsci11020240](https://doi.org/10.3390/brainsci11020240)

---

## 1. Prototype Scope & Objectives

This repository contains the experimental milestone code for **M12–M13 (Computer Vision Engineer)**:
1. **Dataset Ingestion:** Ingestion, parsing, and metadata indexing for the **NTHU-DDD** (National Tsing Hua University Driver Drowsiness Detection) dataset.
2. **Image Preprocessing:** 2D Gaussian blur filter ($3 \times 3$, $\sigma=0.5$) and channel-independent CLAHE lighting compensation.
3. **Face Detection:** Cascaded AdaBoost classifier approximation with failure handling and bounding box output.
4. **Eye Localization:** Multi-mode ocular localizer supporting Haar cascade approximation and manual canvas ROI selection.
5. **Eye ROI Extraction:** Spatial boundary validation and standardized grid cropping ($24 \times 64$ pixels, $3 \times 8$ cells).
6. **Improved HOG Feature Extraction:** Exact implementation of **Binarized Histograms of Shifted Orientations** (Equations 1–5):
   - Centered discrete derivative mask $[-1, 0, 1]$
   - 8-bin unsigned orientation $[0, \pi)$ with anti-aliasing soft voting
   - 8 cyclic orientation shifts ($\varepsilon = 0..7$)
   - Binary step-function cell comparison $b_{c_1 c_2}(k, \varepsilon)$
7. **Feature Export:** Tabular serialization of ground-truth labels and feature vectors to CSV for subsequent Naïve Bayes modeling by Raditya.
8. **Interactive Local Web Interface:** Single-page dashboard built with HTML, CSS, and Vanilla JavaScript powered by FastAPI.

---

## 2. Directory Structure

```text
DriveGuard/
│
├── NTHU-DDD/                       # Academic Dataset (drowsy: 9000 jpg, notdrowsy: 9000 jpg)
│   ├── drowsy/
│   └── notdrowsy/
│
├── app/                            # Modular Python Computer Vision Package
│   ├── __init__.py
│   ├── config.py                   # Central hyperparameters & provenance tags
│   ├── dataset_loader.py           # Dataset scanning & subject metadata parser
│   ├── preprocessing.py            # Gaussian blur & channel-independent CLAHE
│   ├── face_detection.py           # Haar cascade face detection approximation
│   ├── eye_localization.py         # Eye pair localization (Haar / Manual ROI / ASM placeholder)
│   ├── roi_extraction.py           # Eye ROI extraction, clamping & standardization
│   ├── improved_hog.py             # Improved HOG (Equations 1-5, shifted orientations)
│   ├── pipeline.py                 # End-to-end integration pipeline
│   ├── feature_export.py           # Tabular CSV export & extraction audit logging
│   └── visualization.py            # Image rendering & base64 visualization generation
│
├── web/                            # Interactive Web Experiment Interface
│   ├── index.html                  # Single-page UI
│   ├── style.css                   # Responsive layout & theme
│   └── app.js                      # Vanilla JavaScript client logic & canvas ROI
│
├── tests/                          # Automated Pytest Suite (21/21 passing)
│   ├── test_dataset_loader.py
│   ├── test_preprocessing.py
│   ├── test_face_detection.py
│   ├── test_roi_extraction.py
│   ├── test_improved_hog.py        # Includes manual truth-table test for Eq. 5
│   └── test_pipeline.py
│
├── outputs/                        # Output Artifacts & Audit Logs
│   ├── preprocessed/
│   ├── face_detection/
│   ├── eye_roi/
│   ├── hog_visualization/
│   ├── features/                   # hog_features.csv, feature_metadata.csv, extraction_report.json
│   └── reports/
│
├── docs/                           # Academic & Engineering Documentation
│   ├── ARTICLE_PARAMETERS.md       # Full parameter mapping against MDPI paper
│   ├── IMPLEMENTATION_NOTES.md     # Theory, mechanics, and function contracts
│   └── DECISIONS_REQUIRED.md       # Analysis of omissions requiring user confirmation
│
├── requirements.txt                # Dependency specification
├── main.py                         # FastAPI local backend server
└── README.md
```

---

## 3. Installation & Setup (Windows PowerShell)

### Step 1: Create Virtual Environment
Open PowerShell inside `C:\MyFiles2\PCVK\DriveGuard`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. Running the Web Interface & Backend

Launch the FastAPI application:

```powershell
.\.venv\Scripts\python.exe main.py
```

The server initializes the dataset index and starts at:
```text
http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` in any web browser.

### Key Capabilities on the Web UI:
1. **Dataset Overview:** View root path, total sample count ($18,000$), drowsy ($9,000$), notdrowsy ($9,000$), and subject count.
2. **Sample Selection:** Select any frame from the dropdown list. Subject ID, scenario, and ground-truth folder label are displayed.
3. **Custom Upload:** Upload custom image files (.jpg, .png) for testing.
4. **Manual ROI Canvas Mode:** Check the "Manual Eye ROI Mode" checkbox to draw an exact ocular rectangle on the face image, bypassing automated eye localization.
5. **Interactive Pipeline Execution:** Click **"Execute Pipeline"** to visualize all stages:
   - Original Frame
   - Gaussian Blur ($3 \times 3$, $\sigma=0.5$)
   - Channel CLAHE (Independent B, G, R)
   - Face Detection Bounding Box
   - Eye Localization Bounding Box
   - Standardized Eye ROI ($24 \times 64$)
   - Improved HOG Gradient Magnitude (Inferno Heatmap)
   - Improved HOG Gradient Orientation (HSV Hue Map)
   - Shifted Histograms ($\varepsilon = 0..7$ bar charts)
   - Binary Comparison Matrix
   - 1D Feature Vector Barcode (Paper Figure 7b reproduction)
6. **Feature Export:** Click **"Export Current & Batch to CSV"** to write features to `outputs/features/hog_features.csv`.

---

## 5. Running Automated Unit Tests

Execute the complete pytest suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All 21 unit tests verify:
- Folder indexing and strict ground truth binary label assignment.
- Gaussian convolution and independent CLAHE channels.
- Face detector failure handling and bounding box clamping.
- Eye ROI boundary checks and standardization.
- **Equation 5 manual truth-table verification** across circular shifts $\varepsilon \in \{0, 1, 7\}$.
- Pure binary descriptor integrity ($\{0, 1\}$, zero NaN, zero Inf).
- End-to-end pipeline execution and CSV export.

---

## 6. Handoff to Naïve Bayes Module (Raditya)

The feature vectors generated by this prototype are stored in:
- `outputs/features/hog_features.csv`
- `outputs/features/feature_metadata.csv`
- `outputs/features/extraction_report.json`

Format:
```text
sample_id,source_file,subject_id,label,label_id,hog_0000,hog_0001,hog_0002,...
```
Raditya can load this CSV directly into scikit-learn or custom Bayesian classifier code to train the driver drowsiness model without data leakage by splitting on `subject_id`.
