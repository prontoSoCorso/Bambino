"""Hand-crafted statistical / temporal / complexity descriptors.

Reproduces the ~17 per-channel descriptors from PROJECT_STATE §2.1 (LogReg
baseline). For 38 channels this yields ~650 features per trial.

Categories (from TODO.txt + paper Section 2.2):
    * statistical: mean, std, min, max, percentiles (25/50/75), skew, kurtosis
    * temporal:    autocorr lag1, autocorr lag5, zero-crossing rate,
                   slope (linear fit), absolute energy, mean abs diff
    * complexity:  CID_CE (complexity-invariant distance estimate)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as scistats

from .dataset import BambinoDataset


def _autocorr(x: np.ndarray, lag: int) -> float:
    if x.size <= lag:
        return 0.0
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float((x[:-lag] * x[lag:]).sum() / denom)


def _zero_cross_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(((x[:-1] * x[1:]) < 0).sum()) / (x.size - 1)


def _slope(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    t = np.arange(x.size, dtype=np.float32)
    a, _ = np.polyfit(t, x, 1)
    return float(a)


def _cid_ce(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.sqrt(np.sum(np.diff(x) ** 2)))


DESCRIPTOR_NAMES = (
    "mean", "std", "min", "max", "peak2peak",
    "skew", "kurtosis", "p25", "p50", "p75",
    "auto_lag1", "auto_lag5", "mean_cross", "zero_cross",
    "slope", "abs_energy", "cid_ce",
)
"""Names of the 17 descriptors per channel — matches `_03_train/logistic_regression/log_regr.ipynb::compute_descriptors`."""


def per_channel_descriptors(x: np.ndarray) -> np.ndarray:
    """17 descriptors for a single 1-D series. Returned as float32 vector.

    Mirrors the legacy `compute_descriptors` (in
    `_03_train/logistic_regression/log_regr.ipynb`):

        statistical: mean, std, min, max, peak2peak, skew, kurtosis,
                     p25, p50, p75
        temporal:    auto_lag1, auto_lag5, mean_cross_rate, zero_cross_rate,
                     slope, abs_energy
        complexity:  cid_ce

    Any NaN/inf produced by edge-case channels (constant series, all-zero
    arrays, polyfit on near-singular grids) is sanitised to 0.0 — sklearn
    classifiers reject NaN inputs and the 0-imputation matches the legacy
    pipeline's behaviour.
    """
    if x.size == 0:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    has_var = std_val > 1e-12
    centered = x - mean_val
    if x.size > 1:
        mean_cross = float(((centered[:-1] * centered[1:]) < 0).sum()) / (x.size - 1)
    else:
        mean_cross = 0.0
    raw = np.array([
        mean_val,
        std_val,
        min_val,
        max_val,
        max_val - min_val,
        scistats.skew(x, bias=False) if has_var else 0.0,
        scistats.kurtosis(x, bias=False) if has_var else 0.0,
        np.percentile(x, 25),
        np.percentile(x, 50),
        np.percentile(x, 75),
        _autocorr(x, lag=1),
        _autocorr(x, lag=5),
        mean_cross,
        _zero_cross_rate(x),
        _slope(x),
        float((x ** 2).sum()),
        _cid_ce(x),
    ], dtype=np.float32)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def trial_feature_vector(
    arr: np.ndarray,
    feature_names: List[str] | None = None,
    channel_prefix: str = "",
) -> Tuple[np.ndarray, List[str]]:
    """Stack `per_channel_descriptors` across all D channels of `arr [T, D]`."""
    feats = []
    names: List[str] = []
    for d in range(arr.shape[1]):
        v = per_channel_descriptors(arr[:, d])
        feats.append(v)
        if feature_names is not None or channel_prefix:
            for n in DESCRIPTOR_NAMES:
                names.append(f"{channel_prefix}c{d}_{n}")
    return np.concatenate(feats, axis=0), names


def build_feature_matrix(
    dataset: BambinoDataset,
    use_pre_stim_context: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, weights, feature_names) over all trials in `dataset`.

    By default, features are extracted from the post-stimulus window only. If
    `use_pre_stim_context=True`, descriptors are also extracted from the
    pre-stim window and concatenated.
    """
    pre_frames = dataset.cfg.window.pre_stim_frames
    rows: List[np.ndarray] = []
    labels: List[int] = []
    weights: List[float] = []
    names: List[str] = []

    for i, inst in enumerate(dataset.instances):
        per_modality_post = []
        per_modality_pre = []
        for key in dataset.modalities:
            arr = inst.get_modality(key)
            post = arr[pre_frames:]
            v_post, n_post = trial_feature_vector(post, feature_names=names if i == 0 else None,
                                                  channel_prefix=f"{key}_post_")
            per_modality_post.append(v_post)
            if i == 0:
                names.extend(n_post)
            if use_pre_stim_context:
                pre = arr[:pre_frames]
                v_pre, n_pre = trial_feature_vector(pre, feature_names=names if i == 0 else None,
                                                    channel_prefix=f"{key}_pre_")
                per_modality_pre.append(v_pre)
                if i == 0:
                    names.extend(n_pre)
        feats = np.concatenate(per_modality_post + per_modality_pre, axis=0)
        # Trial-order scalar (Workstream B.2.1)
        feats = np.concatenate([feats, [dataset.trial_position(inst)]], axis=0)
        if i == 0:
            names.append("trial_position")
        rows.append(feats)
        labels.append(int(inst.trial_type))
        weights.append(float(inst.sample_weight))

    X = np.stack(rows, axis=0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int64)
    w = np.array(weights, dtype=np.float32)
    return X, y, w, names
