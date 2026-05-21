"""Metrics with bootstrap CIs and habituation-bucketed reporting.

Per PROJECT_STATE §2.1: bootstrap CIs use 200 resamples × 70% of the test set.
Per PROJECT_STATE §3.2.2: trial-bucketed metrics report early ⅓ / mid ⅓ / late ⅓
of session, so a model that only detects fresh responses is correctly credited.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Single-sample metric set; rounded to 4 decimals to match legacy reports."""
    out: Dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except ValueError:
        out["brier"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out["specificity"] = float(tn / max(tn + fp, 1))
    return {k: round(v, 4) for k, v in out.items()}


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    metric_fn,
    n_resamples: int = 200,
    sample_frac: float = 0.7,
    seed: int = 2025,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) at 95% via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(int(round(n * sample_frac)), 2)

    values: List[float] = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=k, replace=True)
        try:
            values.append(metric_fn(y_true[idx], y_pred[idx], y_score[idx]))
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def habituation_bucketed_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    trial_positions: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Stratify metrics by early ⅓ / mid ⅓ / late ⅓ of session.

    Trial position is normalised to [0, 1] per subject in the dataset.
    """
    pos = np.asarray(trial_positions)
    buckets = {
        "early": pos < 1 / 3,
        "mid": (pos >= 1 / 3) & (pos < 2 / 3),
        "late": pos >= 2 / 3,
    }
    out: Dict[str, Dict[str, float]] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    for name, mask in buckets.items():
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            out[name] = {"n": int(mask.sum())}
            continue
        m = core_metrics(y_true[mask], y_pred[mask], y_score[mask])
        m["n"] = int(mask.sum())
        out[name] = m
    return out
