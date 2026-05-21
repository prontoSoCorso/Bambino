"""L1-penalised Logistic Regression on hand-crafted descriptors.

Mirrors `_03_train/logistic_regression/log_regr.ipynb` exactly:

    * Feature matrix: 17 descriptors × 38 channels (≈650 numeric features) +
      age (numeric, scaled) + sex (one-hot) — built by
      `data.features.build_feature_matrix` with the descriptor set defined
      in `data.features.DESCRIPTOR_NAMES`.
    * Classifier: `LogisticRegressionCV(Cs=10, cv=5, penalty='l1',
      solver='liblinear', class_weight='balanced')` — selects regularisation
      strength via 5-fold CV on the training set.
    * Threshold: tuned on the validation set to maximise balanced accuracy.
    * Test: bootstrap-CI AUROC + habituation-bucketed AUROC over early/mid/late.

This module is NOT a LightningModule — it's an sklearn-style fit function
returning a `LogRegResult` dataclass. The orchestrator in `main.py` routes
the run through `fit_logreg(...)` rather than a Lightning trainer.

Sample weights compose class-balance × habituation-decay × augmentation-share
per PROJECT_STATE §3.2.2 and are forwarded via the sklearn `sample_weight=`
argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..configs import LogRegConfig
from ..utils.metrics import bootstrap_ci, core_metrics, habituation_bucketed_metrics


@dataclass
class LogRegResult:
    """Output of `fit_logreg`. `model` is the trained sklearn estimator."""
    model: Any
    scaler: StandardScaler
    test_metrics: Dict[str, float]
    bootstrap_auc: tuple  # (mean, ci_low, ci_high)
    bucket_metrics: Dict[str, Dict[str, float]]
    selected_C: Optional[float] = None
    n_nonzero_coefs: Optional[int] = None


def fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_trial_positions: np.ndarray,
    cfg: LogRegConfig,
    sample_weight_train: Optional[np.ndarray] = None,
    seed: int = 2025,
    use_cv: bool = True,
) -> LogRegResult:
    """Fit L1 LogReg on train, threshold-tune on val, evaluate on test.

    `use_cv=True` (default) selects the L1 strength `C` by 5-fold CV inside
    the training set, matching the legacy `LogisticRegressionCV` recipe.
    Setting `use_cv=False` falls back to a single-shot `LogisticRegression(C=cfg.C)`
    — useful for fast smoke tests.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    if use_cv and cfg.penalty == "l1":
        # Legacy: liblinear is reliable for L1; saga also works but slower.
        model = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="liblinear",
            class_weight=cfg.class_weight,
            max_iter=cfg.max_iter,
            random_state=seed,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(
            penalty=cfg.penalty,
            C=cfg.C,
            solver=cfg.solver if cfg.penalty != "l1" else "liblinear",
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            random_state=seed,
            l1_ratio=0.5 if cfg.penalty == "elasticnet" else None,
        )
    model.fit(X_tr, y_train, sample_weight=sample_weight_train)

    selected_C = float(model.C_[0]) if hasattr(model, "C_") else cfg.C
    coefs = model.coef_.flatten() if hasattr(model, "coef_") else np.array([])
    n_nonzero = int(np.sum(np.abs(coefs) > 1e-5))

    # Threshold-tune on val
    val_scores = model.predict_proba(X_va)[:, 1]
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        ba = balanced_accuracy_score(y_val, (val_scores >= thr).astype(np.int64))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)

    # Test
    test_scores = model.predict_proba(X_te)[:, 1]
    test_preds = (test_scores >= best_thr).astype(np.int64)
    test_metrics = core_metrics(y_test, test_preds, test_scores)
    test_metrics["threshold"] = best_thr
    test_metrics["selected_C"] = selected_C
    test_metrics["n_nonzero_coefs"] = n_nonzero

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        y_test, test_preds, test_scores,
        metric_fn=lambda y, _p, s: _safe_auc(y, s),
        seed=seed,
    )
    bucket = habituation_bucketed_metrics(
        y_test, test_preds, test_scores, test_trial_positions
    )

    return LogRegResult(
        model=model,
        scaler=scaler,
        test_metrics=test_metrics,
        bootstrap_auc=(auc_mean, auc_lo, auc_hi),
        bucket_metrics=bucket,
        selected_C=selected_C,
        n_nonzero_coefs=n_nonzero,
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
