"""MiniRocket transform + metadata fusion + HistGradientBoosting.

Mirrors `_03_train/minirocket_and_metadata/minirocket_xgb.ipynb`:

    1. Build (N, C=38, L) tensor from train/val/test BambinoDatasets.
    2. Apply `sktime.transformations.panel.rocket.MiniRocket` with
       `num_kernels=2000, max_dilations_per_kernel=32` — fit on train, then
       transform val/test.
    3. Concatenate per-trial (age scalar, sex one-hot, optional trial_position)
       to the rocket features (legacy "minirocket + metadata" recipe).
    4. Sanitise NaN/Inf (rocket can produce Infs on degenerate channels) →
       float32.
    5. Fit `HistGradientBoostingClassifier` with the legacy hyperparameters
       (`max_iter=200, learning_rate=0.05, max_depth=3, max_features=0.2,
       max_bins=128, l2_regularization=0.1, early_stopping`).
    6. Threshold-tune on val (max balanced accuracy) → bootstrap CI on test
       → habituation-bucketed AUROC.

This is sklearn-style — no Lightning involved. The orchestrator routes the
run through `fit_minirocket(...)` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..configs import MiniRocketConfig
from ..data.dataset import BambinoDataset
from ..utils.metrics import bootstrap_ci, core_metrics, habituation_bucketed_metrics


# ─── Time-series → numpy in (N, C, L) layout ────────────────────────────────
def dataset_to_tensor(dataset: BambinoDataset, use_pre_stim: bool = False) -> np.ndarray:
    """Stack the post-stimulus segments of every trial into (N, 38, L).

    If `use_pre_stim=True`, the pre-stim and post-stim segments are
    concatenated along time — useful when comparing baseline-aware MiniRocket
    against the supervised default (PROJECT_STATE §3.2.1 step 4).
    """
    pre_frames = dataset.cfg.window.pre_stim_frames
    arrs = []
    for inst in dataset.instances:
        full = np.concatenate(
            [inst.gaze_info, inst.head_info, inst.face_info], axis=1
        )  # (T, 38)
        if use_pre_stim:
            seg = full
        else:
            seg = full[pre_frames:]
        arrs.append(seg.T)  # (38, L)
    return np.stack(arrs, axis=0).astype(np.float32)


def metadata_matrix(dataset: BambinoDataset, include_trial_position: bool) -> np.ndarray:
    """(N, 3 [+1]) — age, sex_one_hot, [trial_position]."""
    rows = []
    for inst in dataset.instances:
        age = float(inst.age)
        sex = int(inst.sex)
        sex_oh = [1.0 if sex == 0 else 0.0, 1.0 if sex == 1 else 0.0]
        row = [age, *sex_oh]
        if include_trial_position:
            row.append(dataset.trial_position(inst))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _sanitize(X: np.ndarray) -> np.ndarray:
    """Clip Inf/NaN → 0.0 and force float32 (matches legacy `enforce_numeric`)."""
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Result dataclass ──────────────────────────────────────────────────────
@dataclass
class MiniRocketResult:
    minirocket: Any           # fit MiniRocket transform
    classifier: Any           # fit HistGradientBoostingClassifier
    test_metrics: Dict[str, float]
    bootstrap_auc: tuple
    bucket_metrics: Dict[str, Dict[str, float]]


# ─── Fit / evaluate ─────────────────────────────────────────────────────────
def fit_minirocket(
    train_ds: BambinoDataset,
    val_ds: BambinoDataset,
    test_ds: BambinoDataset,
    cfg: MiniRocketConfig,
    seed: int = 2025,
    include_trial_position: bool = True,
) -> MiniRocketResult:
    """End-to-end MiniRocket+HistGB pipeline."""
    try:
        from sktime.transformations.panel.rocket import MiniRocket
    except ImportError as e:
        raise ImportError(
            "sktime is required for MiniRocket. `pip install 'sktime>=0.27'`."
        ) from e
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    # 1. Time-series tensors
    X_train_ts = dataset_to_tensor(train_ds)
    X_val_ts = dataset_to_tensor(val_ds)
    X_test_ts = dataset_to_tensor(test_ds)

    # 2. MiniRocket fit/transform — sktime requires (N, C, L) numpy.
    mr = MiniRocket(
        num_kernels=cfg.num_kernels,
        max_dilations_per_kernel=32,
        random_state=seed,
    )
    F_tr = _sanitize(mr.fit_transform(X_train_ts))
    F_va = _sanitize(mr.transform(X_val_ts))
    F_te = _sanitize(mr.transform(X_test_ts))

    # 3. Metadata fusion
    M_tr = metadata_matrix(train_ds, include_trial_position)
    M_va = metadata_matrix(val_ds, include_trial_position)
    M_te = metadata_matrix(test_ds, include_trial_position)

    if cfg.use_metadata:
        X_tr = np.hstack([F_tr, M_tr])
        X_va = np.hstack([F_va, M_va])
        X_te = np.hstack([F_te, M_te])
    else:
        X_tr, X_va, X_te = F_tr, F_va, F_te

    X_tr, X_va, X_te = _sanitize(X_tr), _sanitize(X_va), _sanitize(X_te)
    y_tr = train_ds.labels
    y_va = val_ds.labels
    y_te = test_ds.labels

    # 4. Classifier — legacy hyperparameters verbatim
    if cfg.classifier == "histgb":
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=cfg.max_depth,
            max_features=0.2,
            max_bins=128,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=seed,
        )
    elif cfg.classifier == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost not installed. `pip install xgboost`.") from e
        clf = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=0.05,
            objective="binary:logistic",
            random_state=seed,
            tree_method="hist",
        )
    else:
        raise ValueError(f"Unknown classifier: {cfg.classifier}")

    clf.fit(X_tr, y_tr)

    # 5. Threshold tuning on val
    val_scores = clf.predict_proba(X_va)[:, 1]
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.0, 1.0, 101):
        ba = balanced_accuracy_score(y_va, (val_scores >= thr).astype(np.int64))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)

    # 6. Test
    test_scores = clf.predict_proba(X_te)[:, 1]
    test_preds = (test_scores >= best_thr).astype(np.int64)
    test_metrics = core_metrics(y_te, test_preds, test_scores)
    test_metrics["threshold"] = best_thr

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        y_te, test_preds, test_scores,
        metric_fn=lambda y, _p, s: float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s)),
        seed=seed,
    )
    test_positions = np.asarray(
        [test_ds.trial_position(i) for i in test_ds.instances], dtype=np.float32
    )
    bucket = habituation_bucketed_metrics(y_te, test_preds, test_scores, test_positions)

    return MiniRocketResult(
        minirocket=mr,
        classifier=clf,
        test_metrics=test_metrics,
        bootstrap_auc=(auc_mean, auc_lo, auc_hi),
        bucket_metrics=bucket,
    )
