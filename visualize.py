"""
NOXIS — Visualization Module
Generates professional benchmark comparison charts.
"""

import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ── Style constants ──────────────────────────────────────────────────
BG_COLOR = "#0E0E14"
PANEL_COLOR = "#1C1C28"
ACCENT = "#8B5CF6"
ACCENT_ALT = "#A78BFA"
TEXT_COLOR = "#F0F0F5"
GRID_COLOR = "#2A2A3A"
GREEN = "#34D399"
RED = "#F43F5E"
DPI = 300


def _apply_dark_style(ax, fig):
    """Apply consistent dark theme to axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.5)


def plot_accuracy_comparison(
    results: List[Dict],
    output_path: str,
):
    """
    Generate a bar chart comparing Accuracy across models.
    """
    names = [r["Model"] for r in results]
    accs = [r["Accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(ax, fig)

    bars = ax.bar(names, accs, color=ACCENT, width=0.55, edgecolor=ACCENT_ALT, linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom",
            color=TEXT_COLOR, fontsize=8, fontweight="bold",
        )

    ax.set_xlabel("Model Architecture", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
    ax.set_title("NOXIS Benchmark — Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(accs) + 0.12))
    plt.xticks(rotation=18, ha="right", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)


def plot_f1_comparison(
    results: List[Dict],
    output_path: str,
):
    """
    Generate a bar chart comparing F1-score across models.
    """
    names = [r["Model"] for r in results]
    f1s = [r["F1"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(ax, fig)

    bars = ax.bar(names, f1s, color=GREEN, width=0.55, edgecolor="#2DD4BF", linewidth=0.5)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom",
            color=TEXT_COLOR, fontsize=8, fontweight="bold",
        )

    ax.set_xlabel("Model Architecture", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
    ax.set_title("NOXIS Benchmark — F1 Score Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, min(1.0, max(f1s) + 0.12))
    plt.xticks(rotation=18, ha="right", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)


def plot_combined_comparison(
    results: List[Dict],
    output_path: str,
):
    """
    Combined grouped bar chart: Accuracy, AUC, F1 per model.
    """
    names = [r["Model"] for r in results]
    metrics_keys = ["Accuracy", "AUC", "F1"]
    colors = [ACCENT, "#6366F1", GREEN]
    n_models = len(names)
    n_metrics = len(metrics_keys)

    x = np.arange(n_models)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(12, 5.5))
    _apply_dark_style(ax, fig)

    for i, (key, color) in enumerate(zip(metrics_keys, colors)):
        vals = [r[key] for r in results]
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=key, color=color, edgecolor=color, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{val:.2f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=6.5, fontweight="bold",
            )

    ax.set_xlabel("Model Architecture", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_title("NOXIS Benchmark — Combined Metrics", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)


def generate_all_charts(results: List[Dict], output_dir: str):
    """Generate all benchmark charts."""
    plot_accuracy_comparison(results, os.path.join(output_dir, "accuracy_comparison.png"))
    plot_f1_comparison(results, os.path.join(output_dir, "f1_comparison.png"))
    plot_combined_comparison(results, os.path.join(output_dir, "combined_comparison.png"))
