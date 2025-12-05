## ACT I – Problem, Deep Learning, and Our Angle

---

### **Slide 1 – Title**

**Title:**
**Traditional Machine Learning for TB Detection on Chest X-rays**

**Content:**

* Subtitle: *A Digital Image Processing Approach (AESFERM Pipeline)*
* Your name, NIM, course, docent, date

**Say:**
“Today I’ll show how we detect Tuberculosis from chest X-rays **without** deep learning, using only classic digital image processing and traditional machine learning.”

---

### **Slide 2 – Why TB Detection Matters**

**Visual:**

* Simple TB infographic or WHO numbers (or just text)

**Bullets (max 3):**

* TB is still a leading infectious killer worldwide.
* Chest X-ray is cheap and widely used for screening.
* Reading X-rays is **slow, subjective, and depends on expert radiologists**.

**Say:**
“The burden is big, the tool is common, but the interpretation still depends heavily on human experts.”

---

### **Slide 3 – Existing Deep Learning Methods**

**Title:**
**Deep Learning: Powerful but Heavy**

**Bullets:**

* CNN-based models for TB detection reach high accuracy in papers.
* But they need **large labeled datasets**, **GPUs**, and **long training time**.
* Hard to deploy in **low-resource hospitals**.

**Say:**
“Deep learning works well in research, but in many hospitals there’s no GPU, no engineer, and no time to babysit a training pipeline.”

*(Avoid claiming DL accuracy is “low”—frame the issue as **cost & practicality**, not performance.)*

---

### **Slide 4 – Our Idea**

**Title:**
**Our Alternative: Lightweight Traditional ML**

**Bullets:**

* Use **classical Digital Image Processing** + **handcrafted features**.
* Train **traditional ML** (no deep learning).
* Goal: A lighter system that still achieves **high accuracy** on TB vs Normal.

**Say:**
“We ask: *How far can we go using just textbook image processing and machine learning?*”

---

### **Slide 5 – Dataset & Sample CXRs**

**Visual:**

* 2–4 images: a couple **Normal** CXRs vs **TB** CXRs (from your dataset)

**Bullets:**

* Kaggle TB Chest X-ray dataset.
* 4,200 images: 3,500 Normal, 700 TB.
* All resized to 512×512 and converted to grayscale.

**Say:**
“These are the actual X-rays we use—some with TB, some completely normal.”

---

## ACT II – Our AESFERM Pipeline (Segmented into “Moving” Slides)

We’ll now “zoom in” on AESFERM and let it feel like an animation by splitting it up.

---

### **Slide 6 – AESFERM Overview**

**Title:**
**Pipeline Overview – AESFERM**

**Visual:**
A big horizontal flow:

> Acquisition → Enhancement → Segmentation & Morphology → Feature Extraction → Feature Representation → Machine Learning

**Bullets:**

* We designed a step-by-step pipeline.
* Each step transforms the image into something more “ML-friendly”.

**Say:**
“This is the spine of the project. I’ll walk you through each block quickly.”

---

### **Slide 7 – Acquisition**

**Title:**
**A – Acquisition**

**Bullets:**

* Load PNG CXRs from Kaggle.
* Resize to 512×512.
* Convert to grayscale, standardize format.

**Visual:**

* Original CXR + resized grayscale (can reuse from notebook)

**Say:**
“First step is just housekeeping: make sure all images have the same size and format.”

---

### **Slide 8 – Enhancement (Concept)**

**Title:**
**E – Enhancement (Concept)**

**Bullets:**

* Remove noise, emphasize edges, improve contrast.
* We use: **Gaussian smoothing → Laplacian → Histogram Equalization/CLAHE**.

**Visual:**

* One 3-panel image: Original → Smoothed → Sharpened/HE.

**Say:**
“Before we segment or extract features, we want the lungs to be as clean and informative as possible.”

---

### **Slide 9 – Gaussian Smoothing**

**Title:**
**Gaussian Smoothing – Reduce Noise**

**Bullets:**

* 2D Gaussian filter (e.g., 5×5).
* Removes random noise while preserving main lung structure.

**Visual:**

* Side-by-side: Original vs Gaussian-smoothed.

**Say:**
“This step removes salt-and-pepper noise and small artifacts from the radiograph.”

---

### **Slide 10 – Laplacian & Histogram Equalization / CLAHE**

**Title:**
**Sharpening & Contrast Enhancement**

**Bullets:**

* Laplacian: highlights edges and boundaries.
* Histogram Equalization / CLAHE: stretches contrast so lesions stand out.

**Visual:**

* Use your enhancement screenshot: Original vs after Laplacian + HE/CLAHE for **both TB and Normal**.

**Say:**
“Now the boundaries and bright TB lesions pop more clearly, which will help segmentation and features later.”

---

### **Slide 11 – Segmentation with Otsu**

**Title:**
**S – Segmentation with Otsu**

**Bullets:**

* Apply **Otsu thresholding** to separate lungs from background.
* Invert so lung becomes foreground (white), background black.

**Visual:**

* Enhanced image vs Otsu binary mask (TB and Normal).

**Say:**
“Here we roughly carve out where lung tissue is, based purely on grayscale intensity.”

---

### **Slide 12 – Morphology: Opening & Cleaning**

**Title:**
**Morphological Cleaning**

**Bullets:**

* Use **opening** (erosion + dilation) to remove small noise blobs.
* Use structuring element shaped like a disk/ellipse.

**Visual:**

* Binary mask before vs after opening (your “3b. Morphological Processes”).

**Say:**
“This step smooths the regions and removes small specks that aren’t part of the lungs.”

---

### **Slide 13 – Connected Components & Final Lung Mask**

**Title:**
**Connected Components: Isolate the Lungs**

**Bullets:**

* Label connected white regions.
* Keep only the **two largest components** → left & right lung.
* Optional closing to fill holes.

**Visual:**

* CC labels and final lung mask (your “4. Connected Component Analysis”).

**Say:**
“Now everything except the two big lung blobs is discarded. We end up with a clean lung mask.”

---

### **Slide 14 – ROI + CLAHE inside Lung Mask**

**Title:**
**Final ROI: Lungs Only**

**Bullets:**

* Multiply mask with original image → isolate lung region of interest.
* Apply **CLAHE** inside the lungs for local contrast enhancement.

**Visual:**

* Image showing black background + only lungs visible, for TB and Normal.

**Say:**
“This is the image we actually feed into our feature extraction stage: only lungs, with boosted local contrast.”

---

### **Slide 15 – Feature Extraction**

**Title:**
**F – Feature Extraction (GLCM, HOG, LBP)**

**Bullets:**

* **GLCM**: texture co-occurrence (smooth vs mottled).
* **HOG**: edge and shape distribution.
* **LBP**: small-scale local texture patterns.

**Visual:**

* Simple icon/diagram for each type (or just text if you’re lazy).

**Say:**
“We don’t feed raw pixels into ML. We convert each ROI into numerical descriptors capturing texture and shape.”

---

### **Slide 16 – Feature Representation & ML**

**Title:**
**R & M – Features → Classifier**

**Bullets:**

* Combine feature vectors:

  * GLCM only, HOG only, LBP only, and all combinations.
* Train a traditional classifier (e.g., SVM or similar) to classify **TB vs Normal**.

**Visual:**

* Simple diagram: Image → Features → ML classifier → TB/Normal.

**Say:**
“We then try several feature combinations to see which gives the best performance.”

---

## ACT III – Results, Best Combo, and Future Work

---

### **Slide 17 – Results: Feature Comparison**

**Title:**
**Performance of Different Feature Sets**

**Table (simplified):**

| Features             | Accuracy | TBC Sensitivity | F1-score  |
| -------------------- | -------- | --------------- | --------- |
| GLCM only            | ~95%     | ~0.82           | ~0.85     |
| HOG only             | ~93%     | ~0.75–0.76      | ~0.79     |
| GLCM + HOG           | ~95–96%  | ~0.83           | ~0.86     |
| **GLCM + LBP + HOG** | **~97%** | **~0.88**       | **~0.90** |
| LBP only             | ~84%     | **0.00**        | 0.00      |

*(Use your exact numbers, but highlight GLCM+LBP+HOG row.)*

**Visual:**

* Optionally show one confusion matrix from the best model.

**Say:**
“The best performance comes from combining **all three** feature types. LBP alone is terrible—it classifies everything as Normal—but in combination, it still adds useful local patterns.”

---

### **Slide 18 – Best Combination & Interpretation**

**Title:**
**Why the Best Combo Works**

**Bullets:**

* GLCM: captures **global texture patterns** of lung parenchyma.
* HOG: captures **edges, cavities, fibrosis lines** typical in TB.
* LBP: adds some **fine local variations**, but only useful when combined.
* Combined GLCM + LBP + HOG reaches **~97% accuracy** with **high TB sensitivity**.

**Say:**
“So TB isn’t just a texture problem or just a shape problem. The lungs change in both structure and texture, and the hybrid feature set captures that better.”

---

### **Slide 19 – Limitations & Future Work**

**Title:**
**Limitations & Future Work**

**Bullets:**

* Trained and tested on a **single public dataset** → need external validation.
* Traditional features may be sensitive to changes in scanner, exposure, or demographics.
* Future directions:

  * Cross-dataset testing (different hospitals).
  * Automatic feature selection / dimensionality reduction.
  * Compare with a small deep-learning model when allowed.

**Say:**
“This is a good traditional baseline, but not the end of the story. We’d like to see how it behaves on other populations and eventually compare to a CNN.”

---

### **Slide 20 – Closing Slide**

**Title:**
**Takeaways**

**Bullets:**

* We built a **full AESFERM pipeline**: from raw CXRs to TB vs Normal.
* With only traditional DIP + ML, we reached **≈97% accuracy**.
* This suggests that in low-resource settings, **well-designed classic methods** can still be very competitive.

**Say (last line):**
“Even without deep learning, careful image processing plus classic ML can already provide meaningful assistance for TB screening.”
  * Explains the **problem** and **why not just deep learning**.
  * Shows **sample TB vs Normal CXRs**.
  * Walks through AESFERM as a series of “moving” slides.
  * Compares feature combinations and highlights the **best one**.
  * Ends with **future work** and a crisp takeaway.
