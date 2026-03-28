"""
NOXIS — Training Pipeline v2
═══════════════════════════════════════════════════════════════

Upgrades:
  • MTCNN face detection before model input
  • 32 frames per video
  • FFT frequency-domain features fused with spatial
  • AdamW optimizer (lr=3e-5, wd=1e-4)
  • Label smoothing (0.05)
  • Hard example mining (oversample misclassified fakes after round 1)
  • 20 epochs with early stopping (patience=5)
  • Per-frame probability logging
  • VRAM-safe: batch_size=1, chunked processing

Usage:
    python train.py
    python train.py --epochs 10 --no-face-detect
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.model.network import DeepfakeDetector
from backend.utils.video_processing import extract_frames
from backend.utils.preprocessing import get_train_transforms, get_val_transforms
from backend.utils.face_detection import get_face_detector
from backend.utils.fft_features import batch_fft_magnitudes

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

DATASET_DIR    = PROJECT_ROOT / "dataset"
WEIGHTS_DIR    = PROJECT_ROOT / "backend" / "weights"
TEST_LIST      = DATASET_DIR / "splits" / "test_list.txt"

NUM_FRAMES     = 32
IMG_SIZE       = 224
BATCH_SIZE     = 1          # Single video per batch — 4 GB VRAM safe
NUM_WORKERS    = 0          # Windows compatibility
NUM_EPOCHS     = 20
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
PATIENCE       = 5
GRAD_CLIP      = 1.0
LABEL_SMOOTH   = 0.1
USE_FACE_DETECT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ═══════════════════════════════════════════════════════════════════
# Label Smoothing Loss
# ═══════════════════════════════════════════════════════════════════

class LabelSmoothBCELoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing."""

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth: 0 → smoothing, 1 → 1-smoothing
        smoothed = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        return self.bce(logits, smoothed)


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class CelebDFDataset(Dataset):
    """
    Celeb-DF v2 dataset with face detection and FFT feature extraction.

    Returns:
        frames_tensor  : (T, C, H, W) face-cropped RGB frames
        fft_tensor     : (T, 1, H, W) FFT magnitude images
        label          : (1,) float tensor
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int,
        transform,
        use_face_detect: bool = True,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.use_face_detect = use_face_detect

        if use_face_detect:
            self.face_detector = get_face_detector(IMG_SIZE)
        else:
            self.face_detector = None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames
        try:
            frames = extract_frames(path, self.num_frames, size=(IMG_SIZE, IMG_SIZE), return_pil=True)
        except Exception:
            frames = []

        if len(frames) == 0:
            frames = [Image.new("RGB", (IMG_SIZE, IMG_SIZE))] * self.num_frames

        # Pad/trim
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]

        # Face detection
        if self.face_detector is not None:
            frames = self.face_detector.crop_faces(frames)

        # FFT magnitudes (before augmentation)
        fft_tensor = batch_fft_magnitudes(frames, IMG_SIZE)  # (T, 1, H, W)

        # Apply transforms
        frames_tensor = torch.stack([self.transform(f) for f in frames])  # (T, C, H, W)

        return frames_tensor, fft_tensor, torch.tensor([label], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def _resolve_path(rel_path: str) -> str:
    """Resolve test_list.txt relative path to absolute."""
    rel_path = rel_path.strip().replace("/", os.sep)
    direct = DATASET_DIR / rel_path
    if direct.exists():
        return str(direct)

    parts = Path(rel_path).parts
    if len(parts) >= 2:
        folder, fn = parts[0], os.sep.join(parts[1:])
        for sub in ("real", "fake"):
            c = DATASET_DIR / sub / folder / fn
            if c.exists():
                return str(c)
        for sub in ("real", "fake"):
            c = DATASET_DIR / sub / rel_path
            if c.exists():
                return str(c)
    return str(direct)


def parse_test_list() -> Tuple[List[str], List[int]]:
    """Parse official test split. Returns (paths, labels) with inverted labels."""
    paths, labels = [], []
    with open(TEST_LIST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            orig_label = int(parts[0])
            rel = parts[1].strip()
            abs_path = _resolve_path(rel)
            if not os.path.isfile(abs_path):
                continue
            label = 0 if orig_label == 1 else 1  # invert: FAKE=1
            paths.append(abs_path)
            labels.append(label)
    return paths, labels


def build_splits(max_per_class: int = 0) -> Tuple:
    """Build train/val splits excluding test videos.
    
    Args:
        max_per_class: If > 0, limit each class to this many videos (balanced).
    """
    test_paths, _ = parse_test_list()
    test_set = set(os.path.normpath(p).lower() for p in test_paths)

    real_paths, fake_paths = [], []
    for real_dir in ["Celeb-real", "YouTube-real"]:
        root = DATASET_DIR / "real" / real_dir
        if not root.exists():
            root = DATASET_DIR / real_dir
        if root.exists():
            for f in sorted(root.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                    if os.path.normpath(str(f)).lower() not in test_set:
                        real_paths.append(str(f))

    for fake_dir in ["Celeb-synthesis"]:
        root = DATASET_DIR / "fake" / fake_dir
        if not root.exists():
            root = DATASET_DIR / fake_dir
        if root.exists():
            for f in sorted(root.iterdir()):
                if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                    if os.path.normpath(str(f)).lower() not in test_set:
                        fake_paths.append(str(f))

    # Shuffle before limiting
    np.random.seed(42)
    np.random.shuffle(real_paths)
    np.random.shuffle(fake_paths)

    # Limit per class if requested
    if max_per_class > 0:
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
        log.info(f"  Limited to {len(real_paths)} real + {len(fake_paths)} fake = {len(real_paths)+len(fake_paths)} videos")

    all_paths = real_paths + fake_paths
    # Correct labels (Real=0, Fake=1)
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    n = len(all_paths)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)

    train_p = [all_paths[i] for i in idx[:split]]
    train_l = [all_labels[i] for i in idx[:split]]
    val_p = [all_paths[i] for i in idx[split:]]
    val_l = [all_labels[i] for i in idx[split:]]

    return train_p, train_l, val_p, val_l


# ═══════════════════════════════════════════════════════════════════
# Hard Example Mining
# ═══════════════════════════════════════════════════════════════════

def identify_hard_examples(
    model: nn.Module,
    paths: List[str],
    labels: List[int],
    transform,
    use_face_detect: bool,
) -> List[int]:
    """
    Run inference on training set, return indices of misclassified fakes.
    These will be oversampled in the next training round.
    """
    log.info("  Hard example mining: identifying misclassified fakes...")
    model.eval()
    hard_indices = []

    ds = CelebDFDataset(paths, labels, NUM_FRAMES, transform, use_face_detect)

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="  Mining", leave=False, ncols=90):
            frames, fft, lbl = ds[i]
            frames = frames.unsqueeze(0).to(DEVICE)
            fft = fft.unsqueeze(0).to(DEVICE)

            with autocast():
                logit = model(frames, fft)

            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0
            true = int(lbl.item())

            # Misclassified fake
            if true == 1 and pred == 0:
                hard_indices.append(i)

            torch.cuda.empty_cache()

    log.info(f"  Found {len(hard_indices)} hard fake examples")
    return hard_indices


def oversample_hard_examples(
    paths: List[str],
    labels: List[int],
    hard_indices: List[int],
    factor: int = 3,
) -> Tuple[List[str], List[int]]:
    """Duplicate hard examples by factor, appending to training set."""
    extra_paths = [paths[i] for i in hard_indices] * factor
    extra_labels = [labels[i] for i in hard_indices] * factor
    return paths + extra_paths, labels + extra_labels


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train(args):
    num_frames = args.num_frames

    log.info("═" * 60)
    log.info("  NOXIS — Training Pipeline v2")
    log.info(f"  Device     : {DEVICE}")
    log.info(f"  Frames     : {num_frames}")
    log.info(f"  Batch      : {BATCH_SIZE}")
    log.info(f"  Epochs     : {args.epochs}")
    log.info(f"  LR         : {LEARNING_RATE}")
    log.info(f"  Label Smooth: {LABEL_SMOOTH}")
    log.info(f"  Face Detect: {args.face_detect}")
    log.info(f"  Max/Class  : {args.max_per_class if args.max_per_class > 0 else 'ALL'}")
    log.info("═" * 60)

    # ── Data ──────────────────────────────────────────────────────
    train_paths, train_labels, val_paths, val_labels = build_splits(args.max_per_class)
    log.info(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = CelebDFDataset(train_paths, train_labels, num_frames,
                               get_train_transforms(), args.face_detect)
    val_ds = CelebDFDataset(val_paths, val_labels, num_frames,
                             get_val_transforms(), args.face_detect)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = DeepfakeDetector(
        num_frames=num_frames,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.4,
        fft_dim=256,
        chunk_size=8,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Params: {total:,} total | {trainable:,} trainable")

    criterion = LabelSmoothBCELoss(smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = str(WEIGHTS_DIR / "noxis_model.pth")

    # ── Training Rounds ───────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # ── Hard example mining after epoch 3 ─────────────────────
        if epoch == 4 and args.hard_mine:
            hard_idx = identify_hard_examples(
                model, train_paths, train_labels,
                get_val_transforms(), args.face_detect,
            )
            if len(hard_idx) > 0:
                aug_paths, aug_labels = oversample_hard_examples(
                    train_paths, train_labels, hard_idx, factor=3,
                )
                train_ds = CelebDFDataset(aug_paths, aug_labels, num_frames,
                                           get_train_transforms(), args.face_detect)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
                log.info(f"  Augmented training set: {len(aug_paths)} videos")

        # ── Train epoch ───────────────────────────────────────────
        model.train()
        train_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch:02d}/{args.epochs} [train]",
                     leave=False, ncols=100)
        for frames, fft, labels in pbar:
            frames = frames.to(DEVICE)
            fft = fft.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                logits = model(frames, fft)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            torch.cuda.empty_cache()

        avg_train = train_loss / max(steps, 1)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = 0
        frame_probs_log = []

        with torch.no_grad():
            for frames, fft, labels in val_loader:
                frames = frames.to(DEVICE)
                fft = fft.to(DEVICE)
                labels = labels.to(DEVICE)

                with autocast():
                    logits, f_probs = model(frames, fft, return_frame_probs=True)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_steps += 1

                # Log per-frame probabilities
                frame_probs_log.append(f_probs.cpu().numpy())

                torch.cuda.empty_cache()

        avg_val = val_loss / max(val_steps, 1)
        scheduler.step(avg_val)
        lr_now = optimizer.param_groups[0]["lr"]

        log.info(f"  Epoch {epoch:02d} │ Train: {avg_train:.4f} │ Val: {avg_val:.4f} │ LR: {lr_now:.1e}")

        # Log frame-level stats
        if frame_probs_log:
            all_fp = np.concatenate(frame_probs_log, axis=0)
            mean_fp = all_fp.mean()
            std_fp = all_fp.std()
            log.info(f"           │ Frame probs: mean={mean_fp:.4f}, std={std_fp:.4f}")

        # ── Checkpoint ────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
                "num_frames": num_frames,
                "fft_dim": 256,
            }, weight_path)
            log.info(f"  ✓ Saved → {weight_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"  ⚠ Early stopping at epoch {epoch}")
                break

    # ── Final evaluation on test split ────────────────────────────
    log.info("\n  Evaluating on official test split...")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE)["model_state_dict"])
    model.eval()

    test_paths, test_labels = parse_test_list()
    test_ds = CelebDFDataset(test_paths, test_labels, num_frames,
                              get_val_transforms(), args.face_detect)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    all_labels, all_probs = [], []
    with torch.no_grad():
        for frames, fft, labels in tqdm(test_loader, desc="  Test eval", leave=False, ncols=90):
            frames = frames.to(DEVICE)
            fft = fft.to(DEVICE)
            with autocast():
                logits = model(frames, fft)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(prob.tolist())
            all_labels.extend(labels.numpy().flatten().astype(int).tolist())
            torch.cuda.empty_cache()

    y_pred = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, y_pred)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, y_pred, zero_division=0)

    log.info("═" * 60)
    log.info("  TEST RESULTS")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  AUC      : {auc:.4f}")
    log.info(f"  F1 Score : {f1:.4f}")
    log.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOXIS Training v2")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                        help="Frames per video (default: 32, use 16 for speed)")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Max videos per class (0=all). Use to limit dataset size.")
    parser.add_argument("--no-face-detect", dest="face_detect", action="store_false")
    parser.add_argument("--no-hard-mine", dest="hard_mine", action="store_false")
    parser.set_defaults(face_detect=USE_FACE_DETECT, hard_mine=True)
    args = parser.parse_args()
    train(args)
