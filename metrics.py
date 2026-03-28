"""
NOXIS — Metrics Computation Module
Computes all evaluation metrics on the official test split.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all benchmark metrics.

    Args:
        labels       : Ground-truth labels (0=REAL, 1=FAKE).
        probabilities: Model output probabilities (P(FAKE)).
        threshold    : Decision boundary.

    Returns:
        Dict with keys: Accuracy, AUC, F1, Precision, Recall.
    """
    y_true = np.array(labels, dtype=int)
    y_prob = np.array(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # AUC requires both classes present
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
    }


def compute_confusion_matrix(
    labels: List[int],
    probabilities: List[float],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix.
    Returns 2×2 array: [[TN, FP], [FN, TP]]
    """
    y_true = np.array(labels, dtype=int)
    y_pred = (np.array(probabilities) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def format_metrics_table(results: List[Dict]) -> str:
    """
    Pretty-print a list of model result dicts as a table.
    Each dict must have 'Model' key plus metric keys.
    """
    if not results:
        return ""

    headers = ["Model", "Accuracy", "AUC", "F1", "Precision", "Recall"]
    col_w = [max(len(h), max(len(str(r.get(h, ""))) for r in results)) + 2 for h in headers]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr = "|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|"

    lines = [sep, hdr, sep]
    for r in results:
        row = "|" + "|".join(str(r.get(h, "")).center(w) for h, w in zip(headers, col_w)) + "|"
        lines.append(row)
    lines.append(sep)

    return "\n".join(lines)
