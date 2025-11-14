# Digital Image Processing Final Project - Tuberculosis Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

This repository contains our final project for Digital Image Processing course - an automated Tuberculosis (TB) detection system using traditional computer vision techniques. The project implements a complete image processing pipeline to analyze chest X-ray images and distinguish between normal and TB-affected lungs.

**Key Features:**
- Complete AESFERM pipeline (Acquisition, Enhancement, Segmentation, Feature Extraction, Representation, Matching)
- Manual implementation of image processing algorithms without deep learning dependencies
- Interpretable features aligned with clinical TB manifestations
- Computationally efficient approach suitable for resource-constrained environments

## 🎯 Problem Statement

Tuberculosis remains a significant global health challenge, particularly in developing countries with limited access to medical experts. Manual interpretation of chest X-rays by radiologists faces challenges of subjectivity, time consumption, and potential misdiagnosis. This project aims to develop an automated computer-aided diagnosis (CAD) system to assist in TB detection.

## 🏗️ Methodology

Our system follows the comprehensive **AESFERM framework**:

### 1. Image Acquisition
- **Dataset**: TB Chest X-ray Database from Kaggle (4,200 images - 3,500 normal, 700 TB cases)
- **Format**: 512×512 PNG images
- **Preprocessing**: Grayscale conversion and standardization

### 2. Image Enhancement
Three-stage enhancement pipeline:
- **Gaussian Smoothing**: Noise reduction using Gaussian kernel
- **Histogram Equalization**: Contrast improvement through CDF mapping
- **Laplacian Sharpening**: Edge enhancement using second derivative

### 3. Image Segmentation
Lung region isolation using:
- **Otsu Thresholding**: Automatic optimal threshold calculation
- **Morphological Operations**: Opening and closing to refine masks
- **Connected Component Analysis**: Selection of largest lung regions

### 4. Feature Extraction
Multiple feature types for comprehensive analysis:
- **Canny Edge Detection**: Lung boundaries and cavity contours
- **Harris Corner Detection**: Inflammation points and structural changes
- **Hough Transform**: Line detection for fibrosis patterns

### 5. Feature Representation
- **Polygonal Approximation**: Boundary simplification and noise reduction
- **Signatures**: 1D functional representation of boundaries

### 6. Feature Matching
- **SIFT (Scale-Invariant Feature Transform)**: Scale and rotation invariant matching
- **Similarity Metrics**: Euclidean distance and cross-correlation

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install opencv-python numpy matplotlib jupyter

### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/Plumz17/PCD_FinalProject.git
cd PCD_FinalProject
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open and run `PCD_FinalProject.ipynb`

### Dataset Setup
The notebook includes code to download the TB Chest X-ray dataset from Kaggle. Alternatively, you can manually download from:
[Kaggle TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

## 📁 Project Structure

```
PCD_FinalProject/
├── PCD_FinalProject.ipynb          # Main Jupyter notebook with implementation
├── normal.png                      # Sample normal chest X-ray
├── tbc.png                         # Sample TB chest X-ray
├── Final_Report.pdf               # AI-generated IEEE format report
├── Proposal.pdf                   # Original project proposal (Indonesian)
└── README.md                      # This file
```

## 🔬 Key Technical Components

### Image Enhancement
```python
# Gaussian Smoothing
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Laplacian Sharpening
laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
img_sharpened = cv2.convertScaleAbs(img_gaussian - laplacian)

# Histogram Equalization
img_enhanced = cv2.equalizeHist(img_sharpened)
```

### Segmentation & Morphology
```python
# Otsu Thresholding
ret, img_seg = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_OTSU)

# Morphological Operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
img_opened = cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel)
```

## 📊 Preliminary Results

| Feature Type | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Edge Features Only | 76% | 74% | 77% | 75% |
| Corner Features Only | 72% | 70% | 73% | 71% |
| Line Features Only | 69% | 67% | 70% | 68% |
| **Combined Features** | **82%** | **80%** | **83%** | **81%** |

## 🎯 Clinical Relevance

The extracted features correspond to actual clinical manifestations of TB:
- **White spots** → Inflammation areas (Harris corners)
- **Cavity boundaries** → Lung tissue damage (Canny edges)
- **Fibrosis lines** → Scar tissue formation (Hough lines)
- **Asymmetrical shapes** → Volume loss in affected lungs

## 🚀 Future Work

- Expand evaluation with comprehensive experiments
- Optimize computational efficiency
- Explore additional traditional feature extraction techniques
- Clinical validation with medical experts
- Integration with modern deep learning approaches

## 👥 Contributors

**Group 5 - Digital Image Processing Course**
- Department of Computer Science and Electronics
- Universitas Gadjah Mada, Yogyakarta, Indonesia

## 📚 References

1. World Health Organization. (2019). Global Tuberculosis Report 2019
2. Liu et al. (2021). Deep learning assistance for tuberculosis diagnosis with chest radiography
3. Qatar University, University of Dhaka, University of Malaya. (2018). TB Chest X-ray Dataset

## 📄 License

This project is licensed under the MIT License - see the repository for details.
