import cv2
import numpy as np
from skimage import feature, img_as_ubyte
import skimage

# ------------------------
# ENHANCEMENT
# ------------------------
def enhance_image(img):
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
    img_sharpened = cv2.convertScaleAbs(img_gaussian - img_laplacian)
    img_hist = cv2.equalizeHist(img_sharpened)
    return img_hist

# ------------------------
# SEGMENTATION
# ------------------------
def segment_image(img):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)
    return img

# ------------------------
# MORPHOLOGY
# ------------------------
def morphological_process(img):
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening)
    return img_opened

# ------------------------
# MASKING
# ------------------------
def apply_mask(img, mask):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img

# ------------------------
# SELECT LUNGS
# ------------------------
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

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    img = cv2.morphologyEx(lungs, cv2.MORPH_CLOSE, kernel_closing)
    return img

# ------------------------
# FEATURES
# ------------------------
def extract_lbp(image, P=8, R=1):
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def extract_glcm(image):
    image = img_as_ubyte(image)
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

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

    for prop in props:
        values = feature.graycoprops(glcm, prop).flatten()
        features.extend(values)

    return np.array(features)

def extract_hog(image):
    hog = skimage.feature.hog(
        image, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return hog

def extract_features(image):
    lbp = extract_lbp(image)
    glcm = extract_glcm(image)
    hog = extract_hog(image)
    return np.concatenate((lbp, glcm, hog))

# ------------------------
# MAIN PIPELINE
# ------------------------
def process_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    enhanced = enhance_image(img)
    segmented = segment_image(enhanced)
    morph = morphological_process(segmented)
    mask = get_mask(morph)
    masked = apply_mask(enhanced, mask)

    return masked

def process_and_get_vector(img):
    processed = process_image(img)
    return extract_features(processed)
