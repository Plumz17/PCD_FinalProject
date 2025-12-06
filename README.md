# Digital Image Processing Final Project – Tuberculosis Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Project Overview

This repository contains our final project for a **Digital Image Processing** course:
an automated **Tuberculosis (TB) detection system** from chest X-ray (CXR) images using **traditional computer vision and machine learning**, **without deep learning**.

We implement a complete, interpretable image-processing pipeline that:

* Is built on **classic DIP operations** (filtering, thresholding, morphology).
* Uses **hand-crafted features** (GLCM, HOG, LBP) instead of CNNs.
* Trains **traditional ML classifiers** on those features.
* Targets **resource-constrained environments** where GPU-based deep learning is impractical.

---

## Key Features

* **AESFERM pipeline**
  Acquisition → Enhancement → Segmentation & Morphology → Feature Extraction → Feature Representation → Machine Learning

* **Manual implementation** of core image-processing steps using OpenCV & NumPy
  (no “black box” deep learning dependencies).

* **Multiple feature families**:

  * GLCM (texture)
  * HOG (shape/edges)
  * LBP (local texture)
    and their combinations.

* **Quantitative evaluation** of feature combinations for TB vs Normal classification.

* **Interpretable and lightweight**, suitable for environments with limited compute and expertise.

---

## Problem Statement

Tuberculosis remains a major global health problem, especially in regions with:

* High TB incidence,
* Limited access to radiologists,
* Limited computing resources.

Chest X-ray is widely used for screening, but **manual reading** suffers from:

* Subjectivity between readers,
* Time and workload constraints,
* Risk of misdiagnosis.

This project explores a **traditional computer vision + ML** approach to support **Computer-Aided Diagnosis (CAD)** for TB on CXR, focusing on:

* **Interpretability** (clear processing stages),
* **Lower computational cost** than deep learning,
* **Reasonable accuracy** on a public TB dataset.

---

## Dataset

* **Source:** [Kaggle – Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* **Total images:** 4,200 CXRs

  * 3,500 **Normal**
  * 700 **TB**
* **Format:** PNG, resized to **512×512**, grayscale.

All images are standardized before entering the pipeline.

---

## Methodology (AESFERM Pipeline)

Our system follows a structured **AESFERM** pipeline:

### 1️⃣ Acquisition

* Load CXR images from the Kaggle dataset.
* Resize to **512×512** pixels.
* Convert to grayscale.
* Ensure consistent format for downstream processing.

---

### 2️⃣ Enhancement

Three-stage enhancement to clean and emphasize relevant patterns:

1. **Gaussian Smoothing**

   * Reduces random noise while preserving overall lung structure.

2. **Laplacian Sharpening**

   * Highlights edges and boundaries (e.g., lung borders, lesion edges).

3. **Histogram Equalization / CLAHE**

   * Expands contrast to make subtle opacities and lesions more visible.

---

### 3️⃣ Segmentation & Morphology

Goal: **isolate lung regions** and remove irrelevant structures.

* **Otsu Thresholding**

  * Automatically finds a global threshold to separate foreground (lungs) from background.

* **Morphological Operations**

  * **Opening** (erosion + dilation) to remove small noise blobs.
  * **Closing** (dilation + erosion) to fill small holes.

* **Connected Component Analysis**

  * Label connected white regions in the binary mask.
  * Retain the **two largest components** as left & right lungs.
  * Produce a **clean lung mask**.

* **Final ROI Extraction**

  * Apply the lung mask to the enhanced CXR → only the lungs remain.
  * Optional **CLAHE inside ROI** for local contrast enhancement.

---

### 4️⃣ Feature Extraction

From the final lung ROI, we compute several feature sets:

* **GLCM (Gray-Level Co-occurrence Matrix)**

  * Captures **global texture statistics** (how intensities co-occur).
  * Useful for distinguishing healthy texture from TB-like mottling / opacities.

* **HOG (Histogram of Oriented Gradients)**

  * Captures **edge and gradient orientation distributions**.
  * Sensitive to lung shape, cavity boundaries, fibrosis lines, etc.

* **LBP (Local Binary Patterns)**

  * Captures **fine local texture patterns** at pixel neighborhoods.

Features are flattened into numeric vectors suitable for classical ML models.

---

### 5️⃣ Feature Representation & Machine Learning

We construct several variants by combining different feature families:

* GLCM
* HOG
* LBP
* GLCM + HOG
* GLCM + LBP
* HOG + LBP
* **GLCM + HOG + LBP**

Each feature combination is used to train a **traditional classifier** (e.g., SVM-style model) to predict **TB vs Normal**.

---

## Results (Updated)

We evaluate multiple feature combinations and compare their performance.
Below are the **updated metrics** (from our final report):

### Feature Combination Performance

| Method               | Accuracy   | False Negative | True Positive |
| -------------------- | ---------- | -------------- | ------------- |
| HOG + LBP            | 93.33%     | 33             | 204           |
| GLCM + LBP           | 95.59%     | 23             | 114           |
| GLCM + HOG           | 95.23%     | 24             | 113           |
| HOG                  | 93.33%     | 34             | 103           |
| LBP                  | 83.69%     | 137            | 0             |
| GLCM                 | 95.23%     | 24             | 113           |
| **GLCM + HOG + LBP** | **96.90%** | **16**         | **121**       |

**Highlights:**

* **Best-performing combination:** **GLCM + HOG + LBP**

  * Accuracy ≈ **96.90%**
  * Only **16 false negatives** and **121 true positives** for TB.
  * Combines **global texture (GLCM)**, **shape/edges (HOG)**, and **local micro-patterns (LBP)**.

* ⚠️ **LBP-only model:**

  * Accuracy ≈ 83.69%, but **fails to detect any TB cases**
  * 137 false negatives, 0 true positives
  * Shows that **LBP alone is unsuitable** for this medical imaging task, but can still contribute when combined with stronger descriptors.

These results suggest that **TB alters both the texture and structure** of the lungs, and that **hybrid feature sets** best capture these changes.

---

## Installation & Usage

### Prerequisites

```bash
pip install opencv-python numpy matplotlib scikit-learn jupyter
```

(Install any additional dependencies as needed, e.g., `scipy`, `pillow`.)

---

### Running the Project

1. **Clone the repository:**

```bash
git clone https://github.com/Plumz17/PCD_FinalProject.git
cd PCD_FinalProject
```

2. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

3. **Open and run the main notebook:**

* `PCD_FinalProject.ipynb`
  (Run all cells to reproduce the pipeline, visualizations, and evaluation.)

---

### Dataset Setup

The notebook includes code to load the **TB Chest X-ray dataset**.
Alternatively, you can manually download it from Kaggle:

* [Kaggle TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

Place the dataset in the expected directory structure (as referenced in the notebook) before running.

---

## Project Structure

```bash
PCD_FinalProject/
├── PCD_FinalProject.ipynb     # Main Jupyter notebook (pipeline + experiments)
├── normal.png                 # Sample normal chest X-ray
├── tbc.png                    # Sample TB chest X-ray
├── FinalReport                # Final IEEE-style project report
├── TBC_detector               # UI/UX Frontend TBC Detector Webpage
├── test set                   # Test Set for further testing
├── README.md                  # This file
└── ...                        # (Any additional scripts/assets)
```

If you added a Flask frontend or extra utilities, you can document them here as well.

---

## Key Technical Snippets

### Image Enhancement (Example)

```python
# Gaussian Smoothing
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Laplacian Sharpening
laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
img_sharpened = cv2.convertScaleAbs(img_gaussian - laplacian)

# Histogram Equalization
img_enhanced = cv2.equalizeHist(img_sharpened)
```

### Segmentation & Morphology (Example)

```python
# Otsu Thresholding
_, img_seg = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological Opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
img_opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)

# Connected Components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_opened, connectivity=8)
```

*(See notebook for full pipeline and feature extraction code.)*

---

## Limitations & Future Work

* Evaluation performed on a **single public dataset** → needs external validation on other hospitals / populations.
* Hand-crafted features may be **sensitive** to imaging protocol, scanner type, or demographic differences.
* Future directions:

  * Cross-dataset and cross-hospital testing.
  * Feature selection / dimensionality reduction to improve robustness.
  * When allowed, comparison with compact **CNN-based** models as a baseline.
  * Integration into a simple web interface (e.g., Flask) for demo purposes.

---

## Contributors

** Group 5 – Digital Image Processing Course**

* **Anders Emmanuel Tan** – Image processing pipeline implementation & Report writing
* **Evan Razzan Adytaputra** – Machine learning, evaluation, & Report writing
* **Indratanaya Budiman** – Frontend / Flask integration
* **Daffa M. Siddiq** – Report writing & documentation

---

## References

1. World Health Organization. *Global Tuberculosis Report* (2019).
2. Liu, Y. et al. “Deep learning assistance for tuberculosis diagnosis with chest radiography in low-resource settings.” *European Respiratory Journal*, 58(6), 2100633 (2021).
3. Qatar University; University of Dhaka; University of Malaya. *Tuberculosis (TB) Chest X-ray Database*, Kaggle (2018).

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file in this repository for details.
