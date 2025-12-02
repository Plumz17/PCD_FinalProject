from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
from skimage import feature, img_as_ubyte
import skimage
import uuid
import os

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = joblib.load("model/model.pkl")

app = Flask(__name__)

img = cv2.imread("normal.png", cv2.IMREAD_GRAYSCALE)
# -----------------------------------
# IMAGE PROCESSING PIPELINE
# -----------------------------------

def enhance_image(img):
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
    img_sharpened = cv2.convertScaleAbs(img_gaussian - img_laplacian)
    img_hist = cv2.equalizeHist(img_sharpened)
    return img_hist

def segment_image(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return cv2.bitwise_not(img)

def morphological_process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def select_lungs(segmented):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented, 8, cv2.CV_32S)
    h, w = segmented.shape

    candidates = []
    for i in range(1, num_labels):
        x, y = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        if w * 0.2 < x < w * 0.8:
            candidates.append((area, i))

    candidates.sort(key=lambda x: x[0], reverse=True)

    lung1 = np.zeros_like(segmented)
    lung2 = np.zeros_like(segmented)

    if len(candidates) >= 1:
        lung1[labels == candidates[0][1]] = 255
    if len(candidates) >= 2:
        lung2[labels == candidates[1][1]] = 255

    return lung1, lung2

def get_mask(img):
    lung1, lung2 = select_lungs(img)
    lungs = cv2.add(lung1, lung2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    return cv2.morphologyEx(lungs, cv2.MORPH_CLOSE, kernel)

def apply_mask(img, mask):
    # pastikan ukuran mask sama persis dengan img
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return cv2.bitwise_and(img, img, mask=mask)


# --------------- FEATURES -------------------
#Extract LBP Feature
def extract_lbp(image, P=8, R=1):
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    # Histogram (59 bins for uniform LBP)
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum()  # normalize
    return hist

#Extract GLCM Feature
def glcm_entropy(glcm):
    # Sum over distances & angles → shape: [levels, levels]
    p = glcm.sum(axis=(2,3))
    p = p / p.sum()
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))

def glcm_variance(glcm):
    p = glcm.sum(axis=(2,3))
    p = p / p.sum()
    i = np.arange(p.shape[0])
    j = np.arange(p.shape[1])
    ii, jj = np.meshgrid(i, j, indexing='ij')
    mean = np.sum(ii * p)
    return np.sum(((ii - mean) ** 2) * p)

def extract_glcm(image):
    image = img_as_ubyte(image)

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # compute GLCM
    glcm = feature.graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    props = ['ASM', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []

    # built-in props
    for prop in props:
        features.extend(feature.graycoprops(glcm, prop).flatten())

    # manual variance & entropy
    for d in range(len(distances)):
        for a in range(len(angles)):
            glcm_slice = glcm[:, :, d, a]
            features.append(np.var(glcm_slice))
            features.append(-np.sum(glcm_slice * np.log2(glcm_slice + 1e-10)))

    return np.array(features)


#Extract HOG Features
def extract_hog(image):
  #Calculated Histogram of Oriented Gradients
  hog = skimage.feature.hog(image, orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2,2),
                            block_norm='L2-Hys',
                            transform_sqrt = True,
                            feature_vector=True)
  return hog

#Use all prev functions to get the final vector
def extract_features(image):
  lbp_features = extract_lbp(image)
  glcm_features = extract_glcm(image)
  hog_features = extract_hog(image)

  final_vector = np.concatenate((lbp_features, glcm_features, hog_features)) #Combine the vectors
  return final_vector

def save_step_image(img, step_name):
    filename = f"{step_name}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join("static/steps", filename)
    cv2.imwrite(path, img)
    return filename

def process_image(img):
    steps = {}

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    steps["01_original_gray"] = save_step_image(gray, "01_original_gray")

    enhanced = enhance_image(gray)
    steps["02_enhanced"] = save_step_image(enhanced, "02_enhanced")

    segmented = segment_image(enhanced)
    steps["03_segmented"] = save_step_image(segmented, "03_segmented")

    morph = morphological_process(segmented)
    steps["04_morphological"] = save_step_image(morph, "04_morphological")

    mask = get_mask(morph)
    steps["05_mask"] = save_step_image(mask, "05_mask")

    masked = apply_mask(enhanced, mask)
    steps["06_masked_applied"] = save_step_image(masked, "06_masked_applied")

    return masked, steps


def process_and_get_vector(img):
    processed, steps = process_image(img)

    # --- LBP ---
    lbp = feature.local_binary_pattern(processed, 8, 1, method="uniform")
    lbp_img = lbp_to_image(lbp)
    steps["07_lbp"] = save_step_image(lbp_img, "07_lbp")

    # --- GLCM ---
    glcm = feature.graycomatrix(
        img_as_ubyte(processed),
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    glcm_matrix = glcm[:, :, 0, 0]
    glcm_img = glcm_to_image(glcm_matrix)
    steps["08_glcm"] = save_step_image(glcm_img, "08_glcm")

    # --- HOG ---
    hog_img, hog_features = hog_to_image(processed)
    steps["09_hog"] = save_step_image(hog_img, "09_hog")

    # ---- Extract feature vector ----
    vector = extract_features(processed)

    return vector, steps



# -----------------------------------
# FLASK ROUTES
# -----------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))

    vector, steps = process_and_get_vector(img)
    pred = model.predict([vector])[0]

    return render_template(
        "index.html",
        prediction=pred,
        steps=steps
    )

if __name__ == "__main__":
    app.run(debug=True)

