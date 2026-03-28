# NOXIS — Deepfake Forensics Engine

> AI-powered deepfake video detection platform with EfficientNet-B3 + BiLSTM temporal modeling and Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-REST%20API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    NOXIS Platform                    │
├──────────────┬───────────────────┬──────────────────┤
│   Frontend   │   Flask Backend   │  Deep Learning   │
│  HTML/CSS/JS │   REST API        │  PyTorch Engine  │
├──────────────┼───────────────────┼──────────────────┤
│ Landing page │ GET  /health      │ EfficientNet-B3  │
│ Detect page  │ POST /predict     │ BiLSTM (temporal)│
│ Grad-CAM UI  │ File handling     │ Grad-CAM XAI     │
│ Glassmorphism│ CORS / Logging    │ Mixed precision  │
└──────────────┴───────────────────┴──────────────────┘
```

## Project Structure

```
NOXIS/
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── model/
│   │   └── network.py         # EfficientNet-B3 + BiLSTM
│   ├── utils/
│   │   ├── video_processing.py # Frame extraction
│   │   └── preprocessing.py    # Image transforms
│   ├── gradcam/
│   │   └── gradcam.py          # Grad-CAM implementation
│   └── weights/
│       └── noxis_model.pth     # Trained model (after training)
├── frontend/
│   ├── index.html              # Landing page
│   ├── detect.html             # Detection page
│   ├── css/
│   │   ├── style.css           # Design system
│   │   └── detect.css          # Detection page styles
│   ├── js/
│   │   ├── main.js             # Landing page interactions
│   │   └── detect.js           # Upload & results controller
│   └── logo.png
├── dataset/
│   ├── real/
│   │   ├── Celeb-real/         # 590 real videos
│   │   └── YouTube-real/       # 300 real videos
│   ├── fake/
│   │   └── Celeb-synthesis/    # 5639 deepfake videos
│   └── splits/
│       └── test_list.txt       # Official Celeb-DF v2 test split
├── train.py                    # Training script
├── requirements.txt
└── README.md
```

---

## Setup (Windows)

### 1. Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (tested on GTX 1650, 4 GB VRAM)
- **CUDA Toolkit** + cuDNN installed

### 2. Install Dependencies

```bash
cd "DEEPFAKE IDENTIFIER"
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

Training will:
- Load Celeb-DF v2 dataset from `dataset/`
- Train EfficientNet-B3 + BiLSTM with mixed precision
- Apply early stopping (patience = 5 epochs)
- Save the best model to `backend/weights/noxis_model.pth`
- Evaluate on the official test split and print metrics

**Expected output:**
```
═══════════════════════════════════════════
  Test Results
  ─────────────────────────────────────────
  Loss     : 0.XXXX
  Accuracy : 0.XXXX
  AUC      : 0.XXXX
  F1 Score : 0.XXXX
═══════════════════════════════════════════
```

### 4. Run the Server

```bash
python backend/app.py
```

Server starts at **http://localhost:5000**.

### 5. Use the Application

1. Open **http://localhost:5000** in your browser
2. Click **Launch Detection**
3. Drag & drop a video file (or click to browse)
4. Click **Begin Forensic Scan**
5. View results: REAL/FAKE verdict, confidence score, probability bar, Grad-CAM attention maps

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true
}
```

### `POST /predict`

**Request:** `multipart/form-data` with `video` field.

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.9234,
  "probability": 0.9234,
  "heatmap_frames": [
    "assets/heatmaps/gradcam_000_a1b2c3.png",
    "assets/heatmaps/gradcam_001_d4e5f6.png"
  ],
  "num_frames_analyzed": 16,
  "inference_time": 3.42
}
```

---

## Model Details

| Component | Specification |
|---|---|
| Backbone | EfficientNet-B3 (pretrained, partially frozen) |
| Temporal | BiLSTM, hidden=256, 2 layers, bidirectional |
| Dropout | 0.4 |
| Frames | 16 per video, 224×224 |
| Training | Mixed precision (AMP), gradient clipping |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau |

---

## Tech Stack

- **Frontend:** Pure HTML5 / CSS3 / Vanilla JavaScript
- **Backend:** Flask, Flask-CORS
- **Deep Learning:** PyTorch, EfficientNet-PyTorch
- **Explainability:** Grad-CAM (gradient-weighted class activation mapping)
- **Computer Vision:** OpenCV, Pillow
- **Dataset:** Celeb-DF v2

---

## License

MIT License — Built for research, education, and portfolio demonstration.
