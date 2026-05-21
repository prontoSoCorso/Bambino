"""Centralised plotting with strict palette enforcement.

Palette contract (NO EXCEPTIONS — every figure across the project must draw
from this set):

    PRIMARY     #882255  Wine     — positive class / stimulus / anomaly
    SECONDARY   #4477AA  Blue     — negative class / control / baseline
    TERTIARY_A  #44AA99  Teal     — group A (e.g. female, age-bucket-1)
    TERTIARY_B  #DDCC77  Sand     — group B (e.g. male,   age-bucket-2)
    ACCENT      #CC6677  Rose     — emphasis / highlights
    REFERENCE   #98A4B0  Grey     — chance line, reference, background

`get_class_colors()` is the canonical lookup; pass label index → colour. Any
plotting code that hard-codes a hex outside the palette will fail review.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ─── Palette ────────────────────────────────────────────────────────────────
PRIMARY = "#882255"
SECONDARY = "#4477AA"
TERTIARY_A = "#44AA99"
TERTIARY_B = "#DDCC77"
ACCENT = "#CC6677"
REFERENCE = "#98A4B0"

PALETTE: Dict[str, str] = {
    "primary": PRIMARY,
    "secondary": SECONDARY,
    "tertiary_a": TERTIARY_A,
    "tertiary_b": TERTIARY_B,
    "accent": ACCENT,
    "reference": REFERENCE,
}

# Class semantics: 0 = control (negative), 1 = stimulus (positive).
CLASS_COLORS: Dict[int, str] = {0: SECONDARY, 1: PRIMARY}


def get_class_colors(labels: Iterable[int]) -> list:
    return [CLASS_COLORS[int(l)] for l in labels]


def apply_default_style() -> None:
    """Default rcParams: no gridlines on top of bars, sensible figure size,
    palette-aware default cycler."""
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": REFERENCE,
        "axes.prop_cycle": plt.cycler(
            "color",
            [PRIMARY, SECONDARY, TERTIARY_A, TERTIARY_B, ACCENT, REFERENCE],
        ),
        "font.family": "DejaVu Sans",
    })


# ─── Plot primitives ────────────────────────────────────────────────────────
def plot_roc_curve(fpr, tpr, auc: float, out_path: str, title: str = "ROC Curve") -> None:
    apply_default_style()
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color=PRIMARY, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, out_path: str, labels=("Control", "Stimulus")) -> None:
    apply_default_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="RdGy_r")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_path)


def plot_training_history(history: Dict[str, list], out_path: str) -> None:
    """Two-panel: loss (left), balanced_accuracy / AUROC (right)."""
    apply_default_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], color=PRIMARY, label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], color=SECONDARY, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)

    if "val_balanced_accuracy" in history:
        axes[1].plot(history["val_balanced_accuracy"], color=TERTIARY_A, label="Val Balanced Acc")
    if "val_roc_auc" in history:
        axes[1].plot(history["val_roc_auc"], color=ACCENT, label="Val AUROC")
    axes[1].axhline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)

    _save(fig, out_path)


def plot_umap_2d(
    embeddings: np.ndarray,
    labels: Sequence[int],
    out_path: str,
    title: str = "UMAP",
    label_names: Sequence[str] = ("Control", "Stimulus"),
) -> None:
    apply_default_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    labels = np.asarray(labels)
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            color=CLASS_COLORS.get(int(cls), TERTIARY_A),
            label=label_names[int(cls)] if int(cls) < len(label_names) else f"class {cls}",
            s=22,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_anomaly_score_distribution(
    scores: Sequence[float],
    labels: Sequence[int],
    out_path: str,
    threshold: Optional[float] = None,
    title: str = "Anomaly Scores",
) -> None:
    """Score histogram split by class; optional threshold line in `ACCENT`."""
    apply_default_style()
    fig, ax = plt.subplots()
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    bins = np.linspace(scores.min(), scores.max(), 30)
    for cls, name, color in (
        (0, "Control (baseline)", SECONDARY),
        (1, "Stimulus (target)", PRIMARY),
    ):
        ax.hist(scores[labels == cls], bins=bins, alpha=0.65, color=color, label=name)
    if threshold is not None:
        ax.axvline(threshold, color=ACCENT, linestyle="--", lw=1.5, label=f"threshold={threshold:.3f}")
    ax.axhline(0, color=REFERENCE, lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    _save(fig, out_path)


def plot_habituation_buckets(
    bucket_metrics: Dict[str, Dict[str, float]],
    metric_key: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """Bar chart of `metric_key` (e.g. 'roc_auc') over early/mid/late buckets."""
    apply_default_style()
    fig, ax = plt.subplots()
    names = ["early", "mid", "late"]
    colors = [TERTIARY_A, TERTIARY_B, ACCENT]
    values = [bucket_metrics.get(n, {}).get(metric_key, 0.0) for n in names]
    ax.bar(names, values, color=colors, edgecolor=REFERENCE)
    ax.axhline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"{metric_key} by session bucket")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _save(fig, out_path)


# ─── Internal ───────────────────────────────────────────────────────────────
def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
