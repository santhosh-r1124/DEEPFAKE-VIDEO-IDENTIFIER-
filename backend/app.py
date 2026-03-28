"""
NOXIS — Flask REST API Backend v2
Serves the deepfake detection engine with frame-level predictions.

Endpoints:
    GET  /health   → system health check
    POST /predict  → upload video → frame-level predictions + Grad-CAM heatmaps
"""

import os
import sys
import uuid
import time
import logging
import traceback
from pathlib import Path

import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.model.network import DeepfakeDetector
from backend.utils.video_processing import extract_frames
from backend.utils.preprocessing import get_val_transforms
from backend.utils.face_detection import get_face_detector
from backend.utils.fft_features import batch_fft_magnitudes
from backend.gradcam.gradcam import generate_gradcam_overlays

# ── Configuration ────────────────────────────────────────────────────
WEIGHTS_PATH = PROJECT_ROOT / "backend" / "weights" / "noxis_model.pth"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
HEATMAP_DIR = FRONTEND_DIR / "assets" / "heatmaps"
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
NUM_FRAMES = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NOXIS-API")

# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# ── Model loading ───────────────────────────────────────────────────
model = None
model_loaded = False
face_detector = None


def load_model():
    """Load the trained model weights (if available)."""
    global model, model_loaded, face_detector

    model = DeepfakeDetector(
        num_frames=NUM_FRAMES,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.4,
        fft_dim=256,
        freeze_backbone=False,
        chunk_size=8,
    )

    if WEIGHTS_PATH.exists():
        try:
            checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model_loaded = True
            log.info(f"Model weights loaded from {WEIGHTS_PATH}")
        except Exception as e:
            log.warning(f"Could not load weights: {e}")
            model_loaded = False
    else:
        log.warning(f"No weights at {WEIGHTS_PATH}. Using random weights (demo mode).")
        model_loaded = False

    model.to(DEVICE)
    model.eval()

    face_detector = get_face_detector(IMG_SIZE)
    log.info(f"Model on {DEVICE} | Face detector ready")


# ── Helpers ──────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# =====================================================================
# Routes
# =====================================================================

@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
        "num_frames": NUM_FRAMES,
        "cuda_available": torch.cuda.is_available(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    total_start = time.time()

    # ── Validate upload ──────────────────────────────────────────
    if "video" not in request.files:
        return jsonify({"error": "No video file provided. Use form field 'video'."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # ── Save to temp ─────────────────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = str(UPLOAD_DIR / temp_filename)

    try:
        file.save(temp_path)
        log.info(f"Saved upload → {temp_path}")
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    try:
        # ── Extract frames ───────────────────────────────────────
        log.info("Extracting frames …")
        frames_pil = extract_frames(temp_path, num_frames=NUM_FRAMES,
                                     size=(IMG_SIZE, IMG_SIZE), return_pil=True)

        if len(frames_pil) == 0:
            return jsonify({"error": "No frames could be extracted from the video."}), 400

        while len(frames_pil) < NUM_FRAMES:
            frames_pil.append(frames_pil[-1])
        frames_pil = frames_pil[:NUM_FRAMES]

        # ── Face detection ───────────────────────────────────────
        log.info("Detecting faces …")
        original_frames = [f.copy() for f in frames_pil]
        if face_detector is not None:
            frames_pil = face_detector.crop_faces(frames_pil)

        # ── FFT features ─────────────────────────────────────────
        log.info("Computing FFT features …")
        fft_tensor = batch_fft_magnitudes(frames_pil, IMG_SIZE)  # (T, 1, H, W)

        # ── Inference ────────────────────────────────────────────
        transform = get_val_transforms()
        frames_tensor = torch.stack([transform(f) for f in frames_pil])
        batch_frames = frames_tensor.unsqueeze(0).to(DEVICE)
        batch_fft = fft_tensor.unsqueeze(0).to(DEVICE)

        log.info("Running inference …")
        with torch.no_grad():
            logit, frame_probs = model(batch_frames, batch_fft, return_frame_probs=True)
            
            # Apply a calibration shift to correct the dataset imbalance (5639 Fake vs 890 Real => ratio ~6.3)
            # log(6.33) ~ 1.84. We subtract this from the logits so it doesn't predict FAKE for everything.
            logit = logit - 2.2
            frame_probs = frame_probs - 2.2
            
            probability = torch.sigmoid(logit).squeeze().item()
            per_frame = torch.sigmoid(frame_probs).squeeze().cpu().numpy().tolist()

        # If it's just a single frame, make sure it's a list
        if not isinstance(per_frame, list):
            per_frame = [per_frame]

        prediction = "FAKE" if probability >= 0.50 else "REAL"
        confidence = probability if prediction == "FAKE" else (1.0 - probability)

        log.info(f"Prediction: {prediction} | Confidence: {confidence:.4f}")

        # ── Grad-CAM (per-frame) ─────────────────────────────────
        log.info("Generating Grad-CAM heatmaps …")
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

        for old in HEATMAP_DIR.glob("gradcam_*.png"):
            old.unlink()

        heatmap_files = generate_gradcam_overlays(
            model=model,
            frames_pil=frames_pil,
            transform=transform,
            device=DEVICE,
            output_dir=str(HEATMAP_DIR),
            max_frames=NUM_FRAMES,  # Generate for ALL frames
        )

        # Build frame-level response
        frame_data = []
        for i in range(len(per_frame)):
            entry = {
                "frame_index": i,
                "probability": round(per_frame[i], 4),
                "prediction": "FAKE" if per_frame[i] >= 0.5 else "REAL",
            }
            if i < len(heatmap_files):
                entry["heatmap"] = f"assets/heatmaps/{heatmap_files[i]}"
            frame_data.append(entry)

        elapsed = time.time() - total_start
        log.info(f"Total time: {elapsed:.2f}s | {NUM_FRAMES} frame predictions")

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probability": round(probability, 4),
            "num_frames_analyzed": NUM_FRAMES,
            "inference_time": round(elapsed, 2),
            "frame_predictions": frame_data,
            "heatmap_frames": [f"assets/heatmaps/{f}" for f in heatmap_files],
        })

    except Exception as e:
        log.error(f"Prediction failed: {traceback.format_exc()}")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ── Error handlers ───────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 200 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    log.info("Starting NOXIS API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
