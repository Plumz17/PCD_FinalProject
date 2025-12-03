import os
import uuid
import atexit
import glob
import logging
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import cv2
import numpy as np
from skimage import feature, img_as_ubyte
import skimage



# ---------------------------
# CONFIG
# ---------------------------

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl") 
    UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
    STEPS_DIR = os.path.join(BASE_DIR, "static", "steps")
    IMAGE_SIZE = (512, 512)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  
    SESSION_FILE_TTL = 60 * 60 * 1  
    CLEANUP_OLDER_THAN = 24 * 3600  


# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("app")

# ---------------------------
# APP INIT
# ---------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(Config.UPLOADS_DIR, exist_ok=True)
os.makedirs(Config.STEPS_DIR, exist_ok=True)

# ---------------------------
# MODEL LOAD
# ---------------------------
model = None
model_loaded = False
try:
    if os.path.exists(Config.MODEL_PATH):
        model = joblib.load(Config.MODEL_PATH)
        model_loaded = True
        logger.info(f"Loaded model from {Config.MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {Config.MODEL_PATH}. model_loaded = False")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")
    model_loaded = False

# ---------------------------
# UTILITIES
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def ensure_uint8(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    # scale floats or other dtypes to uint8
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)
    else:
        # generic cast with normalization
        imin, imax = img.min(), img.max()
        if imax - imin == 0:
            return np.zeros(img.shape, dtype=np.uint8)
        norm = (img - imin) / (imax - imin)
        return (norm * 255).astype(np.uint8)

def save_step_image(img, step_name, session_id=None):  # Save an intermediate step image
    try:
        if img is None:
            return None
        filename = f"{step_name}_{session_id or 'global'}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(Config.STEPS_DIR, filename)
        img_to_save = ensure_uint8(img)
        cv2.imwrite(path, img_to_save)
        return os.path.join("static", "steps", filename).replace(os.path.sep, "/")
    except Exception as e:
        logger.exception(f"Failed saving step image {step_name}: {e}")
        return None

def cleanup_session_files(session_id): #Remove files belonging to a session (uploads and steps). Files are matched by session id substring in filename.
    try:
        patterns = [
            os.path.join(Config.UPLOADS_DIR, f"*{session_id}*.png"),
            os.path.join(Config.STEPS_DIR, f"*{session_id}*.png"),
            os.path.join(Config.UPLOADS_DIR, f"*{session_id}*.*"),
            os.path.join(Config.STEPS_DIR, f"*{session_id}*.*"),
        ]
        removed = []
        for pat in patterns:
            for f in glob.glob(pat):
                try:
                    os.remove(f)
                    removed.append(f)
                except Exception as e:
                    logger.warning(f"Could not remove file {f}: {e}")
        logger.info(f"cleanup_session_files({session_id}) removed {len(removed)} files")
        return removed
    except Exception as e:
        logger.exception(f"Error in cleanup_session_files: {e}")
        return []

def cleanup_all_temp_files(older_than_seconds=Config.CLEANUP_OLDER_THAN): #Cleanup all temporary files older than older_than_seconds
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=older_than_seconds)
        removed = []
        for folder in (Config.UPLOADS_DIR, Config.STEPS_DIR):
            for f in glob.glob(os.path.join(folder, "*")):
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(f))
                    if mtime < cutoff:
                        os.remove(f)
                        removed.append(f)
                except Exception as e:
                    logger.debug(f"Skipping file during cleanup {f}: {e}")
        logger.info(f"cleanup_all_temp_files removed {len(removed)} files older than {older_than_seconds} seconds")
        return removed
    except Exception as e:
        logger.exception(f"Error in cleanup_all_temp_files: {e}")
        return []

# Register global cleanup on shutdown
atexit.register(lambda: cleanup_all_temp_files())

# ---------------------------
# IMAGE PIPELINE
# ---------------------------
def enhance_image(img):
    ensure_uint8(img)
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    img_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)
    img_sharpened = cv2.convertScaleAbs(img_gaussian - img_laplacian)
    img_hist = cv2.equalizeHist(img_sharpened)
    return img_hist

def segment_image(img):
    _, img_thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return cv2.bitwise_not(img_thr)

def morphological_process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def select_lungs(segmented):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented, 8, cv2.CV_32S)
    h, w = segmented.shape

    candidates = []
    for i in range(1, num_labels):
        x = centroids[i][0] 
        y = centroids[i] [1]
        area = stats[i, cv2.CC_STAT_AREA]
        # select components roughly within central horizontal band
        if w * 0.2 < x < w * 0.8:
            candidates.append((area, i))

    candidates.sort(key=lambda x: x[0], reverse=True)

    lung1 = np.zeros_like(segmented)
    lung2 = np.zeros_like(segmented)

    if len(candidates) >= 1:
        largest_component_id = candidates[0][1]
        lung1[labels == largest_component_id] = 255
    if len(candidates) >= 2:
        second_largest_component_id = candidates[1][1]
        lung2[labels == second_largest_component_id] = 255

    return lung1, lung2

def get_mask(img):
    lung1, lung2 = select_lungs(img)
    lungs = cv2.add(lung1, lung2)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    return cv2.morphologyEx(lungs, cv2.MORPH_CLOSE, kernel_close)

def apply_mask(img, mask):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return cv2.bitwise_and(img_clahe, img_clahe, mask=mask)

# Feature extractors
def extract_lbp(image, P=8, R=1):
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    n_bins = P+2
    hist = np.bincount(lbp.astype(int).ravel(), minlength=n_bins)
    hist = hist.astype("float")
    if hist.sum() != 0:
        hist /= hist.sum()
    return hist

def extract_glcm(image):
  image = img_as_ubyte(image) 
  distances = [1, 2, 3] 
  angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] 
  glcm = feature.graycomatrix(image,
                              distances=distances,
                              angles=angles,
                              levels=256,
                              symmetric=True,
                              normed=True) 
  features = []
  props = ['ASM', 'variance', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'entropy']

  for prop in props: 
    values = feature.graycoprops(glcm, prop).flatten()
    features.extend(values)

  return np.array(features)

def extract_hog(image):
    hog = skimage.feature.hog(image, orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2,2),
                              block_norm='L2-Hys',
                              transform_sqrt=True,
                              feature_vector=True)
    return hog

def extract_features(image):
    lbp_features = extract_lbp(image)
    glcm_features = extract_glcm(image)
    hog_features = extract_hog(image)
    final_vector = np.concatenate((lbp_features, glcm_features, hog_features))
    return final_vector

# ---------------------------
# PROCESSING WRAPPERS
# ---------------------------
def process_image(img, session_id=None):
    if img is None:
        raise ValueError("Input image is None")
    steps = {}
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        steps["01_original_gray"] = save_step_image(gray, "01_original_gray", session_id)

        enhanced = enhance_image(gray)
        steps["02_enhanced"] = save_step_image(enhanced, "02_enhanced", session_id)

        segmented = segment_image(enhanced)
        steps["03_segmented"] = save_step_image(segmented, "03_segmented", session_id)

        morph = morphological_process(segmented)
        steps["04_morphological"] = save_step_image(morph, "04_morphological", session_id)

        mask = get_mask(morph)
        steps["05_mask"] = save_step_image(mask, "05_mask", session_id)

        masked = apply_mask(enhanced, mask)
        steps["06_masked_applied"] = save_step_image(masked, "06_masked_applied", session_id)

        return masked, steps
    except Exception as e:
        logger.exception(f"Error in process_image: {e}")
        raise

def process_and_get_vector(img, session_id=None):
    try:
        processed, steps = process_image(img, session_id)
        vector = extract_features(processed)
        return vector, steps
    except Exception as e:
        logger.exception(f"Error in process_and_get_vector: {e}")
        raise
if os.path.exists(Config.MODEL_PATH):
    model = joblib.load(Config.MODEL_PATH)
    model_loaded = True
else:
    logger.warning(f"Model not found at {Config.MODEL_PATH}. model_loaded = False")

# ---------------------------
# FLASK ROUTES
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded)

@app.route("/predict", methods=["POST"])
def predict():
    session_id = uuid.uuid4().hex[:12]
    if model is None:
        logger.error("Prediction attempted without loaded model")
        return render_template("index.html", error="Model not loaded. Please ensure model/model.pkl exists.", model_loaded=False), 500

    if "image" not in request.files:
        return render_template("index.html", error="No image file uploaded.", model_loaded=model_loaded), 400

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.", model_loaded=model_loaded), 400

    if not allowed_file(file.filename):
        return render_template("index.html", error=f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}", model_loaded=model_loaded), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return render_template("index.html", error="Invalid or corrupted image file.", model_loaded=model_loaded), 400

        img = cv2.resize(img, Config.IMAGE_SIZE)

        # Save upload
        upload_filename = f"upload_{session_id}.png"
        upload_path = os.path.join(Config.UPLOADS_DIR, upload_filename)
        cv2.imwrite(upload_path, img)
        upload_url = os.path.join("static", "uploads", upload_filename).replace(os.path.sep, "/")

        logger.info(f"Starting image processing for session {session_id}...")
        vector, steps = process_and_get_vector(img, session_id)

        if vector is None or len(vector) == 0:
            cleanup_session_files(session_id)
            return render_template("index.html", error="Feature extraction failed (empty vector).", model_loaded=model_loaded), 500

        logger.info("Making prediction...")
        pred = model.predict([vector])[0]


        response = render_template(
            "index.html",
            prediction=pred,
            steps=steps,
            upload_image=upload_url,
            model_loaded=model_loaded,
            session_id=session_id
        )

        return response
    except Exception as e:
        logger.exception("Error during prediction")
        cleanup_session_files(session_id)
        return render_template("index.html", error=f"Processing failed: {str(e)}", model_loaded=model_loaded), 500

@app.route("/cleanup/<session_id>", methods=["POST"])
def cleanup_session_endpoint(session_id):
    try:
        removed = cleanup_session_files(session_id)
        return jsonify({"status": "success", "removed": removed}), 200
    except Exception as e:
        logger.exception("cleanup endpoint error")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": Config.MODEL_PATH
    })

@app.route("/static/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(Config.UPLOADS_DIR, filename)

@app.route("/static/steps/<path:filename>")
def serve_steps(filename):
    return send_from_directory(Config.STEPS_DIR, filename)

# ---------------------------
# ERROR HANDLERS
# ---------------------------
@app.errorhandler(413)
def too_large(e):
    logger.warning("File upload exceeded size limit")
    return render_template("index.html", error=f"File size too large. Max is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB.", model_loaded=model_loaded), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error="Page not found.", model_loaded=model_loaded), 404

@app.errorhandler(500)
def internal_error(e):
    logger.exception(f"Internal server error: {e}")
    return render_template("index.html", error="Internal server error occurred. Please try again.", model_loaded=model_loaded), 500

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    logger.info("Starting Flask application...")
    logger.info(f"Model loaded: {model_loaded}")
    cleanup_all_temp_files()  # clean leftovers on start
    app.run(debug=True, host="0.0.0.0", port=5000)
