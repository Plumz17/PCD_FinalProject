Yeah, let’s lock in the “final form” version. I’ll reuse the same overall flow as before, but now with your **updated numbers**.

I’ll keep it compact so you can scan + copy into slides easily.

---

## Slide 1 – Title

**Title:**
Traditional Machine Learning for TB Detection on Chest X-rays

**Subtitle:**
A Digital Image Processing (AESFERM) Pipeline

Bottom: Name, NIM, course, docent, date.

---

## Slide 2 – Why TB Detection Matters

* TB is still a major global health problem.
* Chest X-ray is cheap and widely available.
* Reading CXRs is **time-consuming and subjective**.

*(Optional tiny text: WHO, etc.)*

---

## Slide 3 – Deep Learning vs Our Approach

**Deep Learning (existing methods):**

* High accuracy in many papers.
* Needs large labeled datasets, GPUs, long training time.
* Hard to deploy in low-resource hospitals.

**Our angle:**

* Explore **classic DIP + traditional ML** as a lighter alternative.

---

## Slide 4 – Dataset & Example CXRs

* Kaggle TB Chest X-ray Dataset.
* **4,200 images**: 3,500 Normal, 700 TB.
* Resized to 512×512, converted to grayscale.

On the right: 2 Normal CXRs vs 2 TB CXRs.

---

## Slide 5 – AESFERM Overview

Show pipeline diagram:

**A**cquisition → **E**nhancement → **S**egmentation & Morphology → **F**eature Extraction → **R**epresentation → **M**achine Learning

One short line under it:
“Step-by-step pipeline from raw X-ray to TB / Normal prediction.”

---

## Slide 6 – Acquisition

* Load CXRs from dataset.
* Resize to 512×512.
* Convert to grayscale.

Image: original vs standardized grayscale image.

---

## Slide 7 – Enhancement: Concept

* Goal: reduce noise, sharpen structure, improve contrast.
* We use (in order):

  * **Gaussian smoothing**
  * **Laplacian sharpening**
  * **Histogram Equalization / CLAHE**

Image: Original → Smoothed → Sharpened / HE example.

---

## Slide 8 – Gaussian Smoothing

* 2D Gaussian filter to remove random noise.
* Preserves overall lung shape.

Image: Original vs Gaussian-smoothed.

---

## Slide 9 – Laplacian + Histogram Equalization

* Laplacian: highlights edges and boundaries.
* Histogram Equalization / CLAHE: stretches contrast so lesions are clearer.

Image: side-by-side enhancement for **TB** and **Normal**.

---

## Slide 10 – Segmentation with Otsu

* Apply **Otsu thresholding** on enhanced image.
* Invert so lungs are white, background black.

Image: enhanced CXR vs Otsu binary mask (TB + Normal).

---

## Slide 11 – Morphology

* **Opening** (erosion + dilation): remove small specks and noise.
* Structuring element: disk / ellipse.

Image: mask before vs after opening.

---

## Slide 12 – Connected Components & Final Lung Mask

* Label connected white regions.
* Keep only **two largest regions** → left & right lung.
* Optional closing to fill holes.

Image: CC labels and final clean lung mask.

---

## Slide 13 – ROI + CLAHE in Lungs

* Multiply mask with original to isolate lungs.
* Apply **CLAHE inside the lung ROI** for local contrast.

Image: background black, only lungs visible (TB & Normal).

Short line:
“This is the input for feature extraction, not the raw CXR.”

---

## Slide 14 – Feature Extraction (GLCM, HOG, LBP)

* **GLCM** (Gray-Level Co-occurrence Matrix):
  captures **global texture** patterns.
* **HOG** (Histogram of Oriented Gradients):
  captures **edges and shape** of lung structures and lesions.
* **LBP** (Local Binary Patterns):
  captures **fine local texture** (micro-patterns).

One diagram or three icons is enough.

---

## Slide 15 – Feature Representation & Classifier

* For each image, we compute feature vectors:

  * GLCM, HOG, LBP, or their combinations.
* Feed these vectors into a **traditional classifier** (e.g., SVM / similar)
  to predict **TB vs Normal**.

Simple diagram: image → features → classifier → TB / Normal.

---

## Slide 16 – Results: Updated Comparison

Use your updated table:

| Method               | Accuracy   | False Negative | True Positive |
| -------------------- | ---------- | -------------- | ------------- |
| HOG + LBP            | 93.33%     | 33             | 204           |
| GLCM + LBP           | 95.59%     | 23             | 114           |
| GLCM + HOG           | 95.23%     | 24             | 113           |
| HOG                  | 93.33%     | 34             | 103           |
| LBP                  | 83.69%     | 137            | 0             |
| GLCM                 | 95.23%     | 24             | 113           |
| **GLCM + HOG + LBP** | **96.90%** | **16**         | **121**       |

Highlight last row in bold on the slide.

Talking points:

* **Best combo:** GLCM + HOG + LBP

  * Highest accuracy: **96.90%**
  * Fewest missed TB cases: **16 false negatives**, **121 TB correctly detected**.
* LBP alone has **accuracy 83.69% but 137 FNs, 0 TPs** → fails to detect TB at all.

---

## Slide 17 – Interpreting the Results

Left:

* GLCM captures **overall lung texture**.
* HOG captures **edges, cavities, fibrosis lines**.
* LBP alone is unstable on smooth medical images but adds a bit of local detail when combined.

Right:

* GLCM + HOG + LBP:

  * Best accuracy (96.90%).
  * Best TB detection (fewest false negatives).

One line:
“TB changes both the **structure** and the **texture** of the lungs, so hybrid features perform best.”

---

## Slide 18 – (Optional but Strong) Error Examples

If you have time/space:

* Show 2–3 misclassified CXRs.
* One short line each: why the model might have been confused
  (overlapping patterns with pneumonia, faint lesions, etc.).

This shows you looked beyond raw metrics.

---

## Slide 19 – Limitations & Future Work

* Trained and evaluated on one dataset → need external validation.
* Handcrafted features may be sensitive to scanner / population differences.
* Future work:

  * Cross-dataset testing.
  * Automatic feature selection.
  * Comparison with compact deep models when allowed.

---

## Slide 20 – Conclusion

* Built full **AESFERM pipeline**: from raw CXR to TB vs Normal classification.
* Using only **classic DIP + traditional ML**, achieved **96.90% accuracy** with few missed TB cases.
* Shows that even without deep learning, a carefully designed pipeline can meaningfully support TB screening.

And then your closer:

> “Thank you.”

(Or the “we did it, we’re proud of it” line you like.)

---

If you want, next step I can write a **tight 5–6 minute script** that walks through exactly this slide order with updated numbers woven in.
