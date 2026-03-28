"""
NOXIS — README Generator
Reads benchmark results from CSV and best_model.txt,
then generates a professional GitHub README.md.

Usage:
    python generate_readme.py
"""

import csv
import os
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH    = RESULTS_DIR / "benchmark_results.csv"
BEST_PATH   = RESULTS_DIR / "best_model.txt"
README_PATH = BASE_DIR / "README.md"


def load_csv():
    """Load benchmark results from CSV."""
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_best_model():
    """Read best model name."""
    return BEST_PATH.read_text().strip()


def build_results_table(rows):
    """Build a markdown table from result rows."""
    lines = []
    lines.append("| Model | Accuracy | AUC | F1 | Precision | Recall |")
    lines.append("|:------|:--------:|:---:|:--:|:---------:|:------:|")
    for r in rows:
        name = r["Model"]
        acc  = r["Accuracy"]
        auc  = r["AUC"]
        f1   = r["F1"]
        prec = r["Precision"]
        rec  = r["Recall"]
        lines.append(f"| {name} | {acc} | {auc} | {f1} | {prec} | {rec} |")
    return "\n".join(lines)


def find_best_row(rows, best_name):
    """Find the row matching the best model."""
    for r in rows:
        if r["Model"] == best_name:
            return r
    return rows[0] if rows else {}


def generate():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run 'python benchmark.py' first.")
        return
    if not BEST_PATH.exists():
        print(f"ERROR: {BEST_PATH} not found. Run 'python benchmark.py' first.")
        return

    rows = load_csv()
    best_name = load_best_model()
    best = find_best_row(rows, best_name)
    table = build_results_table(rows)

    # Check for chart images
    has_accuracy_chart = (RESULTS_DIR / "accuracy_comparison.png").exists()
    has_combined_chart = (RESULTS_DIR / "combined_comparison.png").exists()

    readme = f"""<div align="center">

# NOXIS

**Deepfake Forensics Engine**

AI-powered deepfake video detection with multi-model benchmarking,
temporal sequence analysis, and Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000?style=flat-square&logo=flask&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

</div>

---

## Overview

NOXIS is a research-grade deepfake detection platform built on the Celeb-DF v2 dataset. It implements a structured multi-model benchmarking framework that trains, evaluates, and compares five deep learning architectures under identical conditions for fair and reproducible results.

### Key Capabilities

- **Multi-model benchmarking** — 5 architectures evaluated on the same test split
- **Temporal modeling** — EfficientNet-B3 + BiLSTM captures inter-frame patterns
- **Explainability** — Grad-CAM attention maps highlight manipulated regions
- **Ensemble voting** — Optional majority-vote aggregation of top-performing models
- **Web interface** — Flask API + custom forensics-style frontend

---

## Benchmark Results

> Evaluated on the official Celeb-DF v2 test split (`test_list.txt`).
> All models trained with identical hyperparameters for fair comparison.

{table}

### Best Performing Model

| | |
|---|---|
| **Model** | **{best_name}** |
| **Accuracy** | {best.get("Accuracy", "—")} |
| **AUC** | {best.get("AUC", "—")} |
| **F1 Score** | {best.get("F1", "—")} |
| **Precision** | {best.get("Precision", "—")} |
| **Recall** | {best.get("Recall", "—")} |

> Best model auto-selected by AUC (primary) and F1 (secondary).

"""

    if has_accuracy_chart:
        readme += """### Accuracy Comparison

<div align="center">
<img src="results/accuracy_comparison.png" width="720" alt="Accuracy Comparison Chart">
</div>

"""

    if has_combined_chart:
        readme += """### Combined Metrics

<div align="center">
<img src="results/combined_comparison.png" width="720" alt="Combined Metrics Chart">
</div>

"""

    readme += f"""---

## Architecture

```
NOXIS/
├── models/                         # Model implementations
│   ├── resnet.py                   #   ResNet50
│   ├── efficientnet_b0.py          #   EfficientNet-B0
│   ├── efficientnet_b3.py          #   EfficientNet-B3
│   ├── mobilenet.py                #   MobileNetV3-Large
│   └── efficientnet_lstm.py        #   EfficientNet-B3 + BiLSTM
├── backend/
│   ├── app.py                      # Flask REST API
│   ├── model/network.py            # Inference model
│   ├── utils/                      # Frame extraction, preprocessing
│   └── gradcam/gradcam.py          # Grad-CAM implementation
├── frontend/                       # Custom web interface
│   ├── index.html                  # Landing page
│   ├── detect.html                 # Detection page
│   ├── css/                        # Stylesheets
│   └── js/                         # Client-side logic
├── dataset/
│   ├── real/                       # Celeb-real + YouTube-real
│   ├── fake/                       # Celeb-synthesis
│   └── splits/test_list.txt        # Official test split
├── results/                        # Generated outputs
│   ├── benchmark_results.csv       # Metric comparison
│   ├── accuracy_comparison.png     # Accuracy chart
│   ├── combined_comparison.png     # Combined metrics chart
│   ├── best_model.txt              # Auto-selected best model
│   └── weights/                    # Trained model weights
├── benchmark.py                    # Multi-model benchmark pipeline
├── metrics.py                      # Evaluation metrics
├── visualize.py                    # Chart generation
├── train.py                        # Single-model training script
└── requirements.txt
```

---

## Models

| Architecture | Parameters | Temporal | VRAM Usage |
|:-------------|:-----------|:---------|:-----------|
| ResNet50 | ~25M | Avg pool | Low |
| EfficientNet-B0 | ~5M | Avg pool | Very low |
| EfficientNet-B3 | ~12M | Avg pool | Moderate |
| MobileNetV3-Large | ~5M | Avg pool | Very low |
| EfficientNet-B3 + BiLSTM | ~14M | BiLSTM | Moderate |

All models use pretrained ImageNet weights with early layers frozen for transfer learning.

---

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Frames per video | 16 |
| Input resolution | 224 × 224 |
| Batch size | 2 |
| Optimizer | Adam (lr=1e-4, wd=1e-5) |
| Scheduler | ReduceLROnPlateau |
| Loss | BCEWithLogitsLoss |
| Mixed precision | ✓ (AMP) |
| Gradient clipping | 1.0 |
| Early stopping | Patience = 4 |
| Train/Val split | 80/20 (seed=42) |

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on GTX 1650, 4 GB VRAM)
- Windows 10/11

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Benchmark

```bash
python benchmark.py
```

This trains all 5 models, evaluates on the official test split, and generates:
- `results/benchmark_results.csv`
- `results/accuracy_comparison.png`
- `results/best_model.txt`

### 3. Launch Web Interface

```bash
python backend/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. Optional: Ensemble Mode

```bash
python benchmark.py --ensemble
```

---

## API

### `GET /health`

```json
{{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}}
```

### `POST /predict`

Upload video as `multipart/form-data` with field `video`.

```json
{{
  "prediction": "FAKE",
  "confidence": 0.94,
  "probability": 0.94,
  "heatmap_frames": ["assets/heatmaps/gradcam_000.png"],
  "num_frames_analyzed": 16,
  "inference_time": 2.31
}}
```

---

## Evaluation Methodology

1. Dataset: **Celeb-DF v2** (590 real + 5639 fake videos)
2. Test split: Official `test_list.txt` (518 videos)
3. Labels: Inverted from source (FAKE = positive class)
4. Metrics: Accuracy, AUC, F1, Precision, Recall
5. All models evaluated with identical preprocessing and frame sampling

---

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Deep Learning | PyTorch, EfficientNet-PyTorch |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Explainability | Grad-CAM |
| Vision | OpenCV, Pillow, torchvision |
| Metrics | scikit-learn |
| Visualization | matplotlib |
| Dataset | Celeb-DF v2 |

---

## License

MIT License — Built for research and education.
"""

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"README.md generated → {{README_PATH}}")
    print(f"Best model: {{best_name}}")
    print(f"Models benchmarked: {{len(rows)}}")


if __name__ == "__main__":
    generate()
