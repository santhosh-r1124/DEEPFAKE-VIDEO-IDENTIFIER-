"""
NOXIS — Multi-Model Benchmark Pipeline
═══════════════════════════════════════════════════════════════════

Trains and evaluates all registered models on the Celeb-DF v2
dataset using identical configuration for fair comparison.

Usage:
    python benchmark.py                # full benchmark
    python benchmark.py --models ResNet50 EfficientNet-B0
    python benchmark.py --ensemble     # enable ensemble voting
    python benchmark.py --skip-train   # eval only (weights must exist)
"""

import os
import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Local imports
from models import get_model, list_models
from metrics import compute_metrics, compute_confusion_matrix, format_metrics_table
from visualize import generate_all_charts
from backend.utils.video_processing import extract_frames
from backend.utils.preprocessing import get_train_transforms, get_val_transforms

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

BASE_DIR       = Path(__file__).resolve().parent
DATASET_DIR    = BASE_DIR / "dataset"
RESULTS_DIR    = BASE_DIR / "results"
WEIGHTS_DIR    = BASE_DIR / "results" / "weights"
TEST_LIST      = DATASET_DIR / "splits" / "test_list.txt"

NUM_FRAMES     = 16
IMG_SIZE       = 224
BATCH_SIZE     = 2          # Safe for 4 GB VRAM
NUM_WORKERS    = 0          # Windows compatibility
NUM_EPOCHS     = 10
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
PATIENCE       = 4          # Early stopping patience
GRAD_CLIP      = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class CelebDFDataset(Dataset):
    """
    Loads video samples from Celeb-DF v2.

    test_list.txt format:  <label> <relative_path>
    Labels:  1 → REAL,  0 → FAKE
    We invert so FAKE = 1 (positive class for BCEWithLogitsLoss).
    """

    def __init__(self, video_paths: List[str], labels: List[int],
                 num_frames: int, transform):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        frames = extract_frames(path, self.num_frames)

        if len(frames) == 0:
            # Fallback: black frames
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * self.num_frames

        # Pad or trim to exact num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]

        tensor_frames = []
        for f in frames:
            from PIL import Image
            img = Image.fromarray(f)
            tensor_frames.append(self.transform(img))

        return torch.stack(tensor_frames), torch.tensor([label], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def _resolve_video_path(rel_path: str) -> str:
    """Resolve a test_list.txt relative path to an absolute path."""
    rel_path = rel_path.strip().replace("/", os.sep)

    # Direct check under dataset/
    direct = DATASET_DIR / rel_path
    if direct.exists():
        return str(direct)

    # Try under real/ and fake/ subdirs
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        folder, filename = parts[0], os.sep.join(parts[1:])

        for sub in ("real", "fake"):
            candidate = DATASET_DIR / sub / folder / filename
            if candidate.exists():
                return str(candidate)

        # Try folder directly inside real/ or fake/
        for sub in ("real", "fake"):
            candidate = DATASET_DIR / sub / rel_path
            if candidate.exists():
                return str(candidate)

    return str(direct)


def load_split(split: str = "test") -> Tuple[List[str], List[int]]:
    """
    Parse test_list.txt and return (paths, labels).
    If split='all', returns all entries. We filter by split type.
    """
    paths, labels = [], []

    with open(TEST_LIST, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            orig_label = int(parts[0])   # 1=REAL, 0=FAKE in test_list.txt
            rel_path = parts[1].strip()
            abs_path = _resolve_video_path(rel_path)

            if not os.path.isfile(abs_path):
                continue

            # Invert: FAKE=1 (positive class)
            label = 0 if orig_label == 1 else 1
            paths.append(abs_path)
            labels.append(label)

    return paths, labels


def build_train_val_split() -> Tuple:
    """
    Build train/val split from all non-test videos.
    Scans real/ and fake/ directories, excludes test-set videos.
    """
    test_paths, _ = load_split("test")
    test_set = set(os.path.normpath(p).lower() for p in test_paths)

    all_paths, all_labels = [], []

    # Real videos
    for real_dir in ["Celeb-real", "YouTube-real"]:
        real_root = DATASET_DIR / "real" / real_dir
        if not real_root.exists():
            real_root = DATASET_DIR / real_dir
        if real_root.exists():
            for f in sorted(real_root.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                    norm = os.path.normpath(str(f)).lower()
                    if norm not in test_set:
                        all_paths.append(str(f))
                        all_labels.append(0)  # REAL

    # Fake videos
    for fake_dir in ["Celeb-synthesis"]:
        fake_root = DATASET_DIR / "fake" / fake_dir
        if not fake_root.exists():
            fake_root = DATASET_DIR / fake_dir
        if fake_root.exists():
            for f in sorted(fake_root.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                    norm = os.path.normpath(str(f)).lower()
                    if norm not in test_set:
                        all_paths.append(str(f))
                        all_labels.append(1)  # FAKE

    # 80/20 split
    n = len(all_paths)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * n)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    return train_paths, train_labels, val_paths, val_labels


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_model(model_name: str) -> str:
    """
    Train a single model. Returns path to saved weights.
    """
    log.info(f"{'═' * 60}")
    log.info(f"  TRAINING: {model_name}")
    log.info(f"{'═' * 60}")

    # Build datasets
    train_paths, train_labels, val_paths, val_labels = build_train_val_split()
    log.info(f"Train: {len(train_paths)} videos | Val: {len(val_paths)} videos")

    train_ds = CelebDFDataset(train_paths, train_labels, NUM_FRAMES, get_train_transforms())
    val_ds = CelebDFDataset(val_paths, val_labels, NUM_FRAMES, get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = get_model(model_name, num_frames=NUM_FRAMES).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total:,} total | {trainable:,} trainable")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    weight_path = str(WEIGHTS_DIR / f"{model_name.replace('+', '_').replace(' ', '_')}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{NUM_EPOCHS} [train]",
                     leave=False, ncols=90)
        for frames, labels in pbar:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / max(train_steps, 1)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast():
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        scheduler.step(avg_val)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"  Epoch {epoch:02d} │ Train: {avg_train:.4f} │ Val: {avg_val:.4f} │ LR: {lr_now:.1e}")

        # ── Checkpoint ────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), weight_path)
            log.info(f"  ✓ Saved best weights → {weight_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"  ⚠ Early stopping at epoch {epoch}")
                break

        # Free VRAM between epochs
        torch.cuda.empty_cache()

    log.info(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    return weight_path


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, weight_path: str) -> Dict:
    """
    Evaluate a trained model on the official test split.
    Returns dict with Model name + all metrics.
    """
    log.info(f"  Evaluating {model_name} on test split...")

    test_paths, test_labels = load_split("test")
    test_ds = CelebDFDataset(test_paths, test_labels, NUM_FRAMES, get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model(model_name, num_frames=NUM_FRAMES).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc=f"  [{model_name}] test", leave=False, ncols=90):
            frames = frames.to(DEVICE)
            with autocast():
                outputs = model(frames)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().flatten().astype(int).tolist())

    mets = compute_metrics(all_labels, all_probs)
    cm = compute_confusion_matrix(all_labels, all_probs)

    log.info(f"  {model_name} │ Acc: {mets['Accuracy']:.4f} │ AUC: {mets['AUC']:.4f} │ F1: {mets['F1']:.4f}")
    log.info(f"  Confusion Matrix:\n{cm}")

    return {"Model": model_name, **mets}


# ═══════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════

def ensemble_predict(
    top_models: List[Tuple[str, str]],
) -> Dict:
    """
    Majority voting ensemble over top N models.
    top_models: [(model_name, weight_path), ...]
    """
    log.info(f"  Ensemble mode: {[m[0] for m in top_models]}")

    test_paths, test_labels = load_split("test")
    test_ds = CelebDFDataset(test_paths, test_labels, NUM_FRAMES, get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Collect predictions from each model
    model_probs = []
    for mname, wpath in top_models:
        model = get_model(mname, num_frames=NUM_FRAMES).to(DEVICE)
        model.load_state_dict(torch.load(wpath, map_location=DEVICE))
        model.eval()

        probs = []
        with torch.no_grad():
            for frames, _ in test_loader:
                frames = frames.to(DEVICE)
                with autocast():
                    out = model(frames)
                p = torch.sigmoid(out).cpu().numpy().flatten()
                probs.extend(p.tolist())
        model_probs.append(probs)

        # Free VRAM
        del model
        torch.cuda.empty_cache()

    # Majority voting
    all_labels = test_labels[:len(model_probs[0])]
    avg_probs = np.mean(model_probs, axis=0).tolist()

    mets = compute_metrics(all_labels, avg_probs)
    log.info(f"  Ensemble │ Acc: {mets['Accuracy']:.4f} │ AUC: {mets['AUC']:.4f} │ F1: {mets['F1']:.4f}")

    return {"Model": "Ensemble (Top-3)", **mets}


# ═══════════════════════════════════════════════════════════════════
# Auto-select best model
# ═══════════════════════════════════════════════════════════════════

def select_best(results: List[Dict]) -> Dict:
    """
    Select best model by AUC (primary), F1 (secondary).
    Returns the winning result dict.
    """
    ranked = sorted(results, key=lambda r: (r["AUC"], r["F1"]), reverse=True)
    return ranked[0]


# ═══════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════

def export_csv(results: List[Dict], path: str):
    """Export results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["Model", "Accuracy", "AUC", "F1", "Precision", "Recall"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    log.info(f"  CSV saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NOXIS Multi-Model Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to benchmark (default: all)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble majority voting with top-3 models")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, evaluate existing weights only")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    args = parser.parse_args()

    global NUM_EPOCHS
    NUM_EPOCHS = args.epochs

    log.info("═" * 60)
    log.info("  NOXIS — Multi-Model Benchmark Pipeline")
    log.info(f"  Device : {DEVICE}")
    log.info(f"  Epochs : {NUM_EPOCHS}")
    log.info(f"  Frames : {NUM_FRAMES}")
    log.info(f"  Batch  : {BATCH_SIZE}")
    log.info("═" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    model_names = args.models if args.models else list_models()
    log.info(f"  Models to benchmark: {model_names}")

    # ── Train & Evaluate ──────────────────────────────────────
    all_results = []
    weight_paths = {}

    for name in model_names:
        weight_file = WEIGHTS_DIR / f"{name.replace('+', '_').replace(' ', '_')}.pth"

        if not args.skip_train:
            wpath = train_model(name)
        elif weight_file.exists():
            wpath = str(weight_file)
            log.info(f"  Using existing weights: {wpath}")
        else:
            log.warning(f"  ⚠ No weights found for {name}, skipping evaluation.")
            continue

        weight_paths[name] = wpath
        result = evaluate_model(name, wpath)
        all_results.append(result)

        # Free VRAM between models
        torch.cuda.empty_cache()

    if not all_results:
        log.error("  No results collected. Exiting.")
        return

    # ── Ensemble ──────────────────────────────────────────────
    if args.ensemble and len(all_results) >= 3:
        ranked = sorted(all_results, key=lambda r: (r["AUC"], r["F1"]), reverse=True)
        top3_names = [r["Model"] for r in ranked[:3]]
        top3 = [(n, weight_paths[n]) for n in top3_names if n in weight_paths]
        if len(top3) >= 2:
            ens_result = ensemble_predict(top3)
            all_results.append(ens_result)

    # ── Results ───────────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  BENCHMARK RESULTS")
    log.info("═" * 60)
    log.info("\n" + format_metrics_table(all_results))

    # CSV
    csv_path = str(RESULTS_DIR / "benchmark_results.csv")
    export_csv(all_results, csv_path)

    # Charts
    log.info("  Generating charts...")
    generate_all_charts(all_results, str(RESULTS_DIR))
    log.info(f"  Charts saved → {RESULTS_DIR}")

    # Best model
    best = select_best([r for r in all_results if "Ensemble" not in r["Model"]])
    best_path = RESULTS_DIR / "best_model.txt"
    with open(best_path, "w") as f:
        f.write(best["Model"])
    log.info(f"\n  🏆 Best Model: {best['Model']}")
    log.info(f"     AUC: {best['AUC']:.4f} | F1: {best['F1']:.4f} | Acc: {best['Accuracy']:.4f}")
    log.info(f"     Saved → {best_path}")

    log.info("\n" + "═" * 60)
    log.info("  Benchmark complete.")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
