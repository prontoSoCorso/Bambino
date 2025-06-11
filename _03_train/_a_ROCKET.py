import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# add project root to sys.path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "Bambino":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train as conf, utils
from DataUtils.BoaOpenFaceDataset import BoaOpenFaceDataset
from _utils_ import models_utils, dataset_utils, plot_utils


def classification_model(head_type="RF"):
    if head_type.upper() == "RF":
        return RandomForestClassifier(
            n_estimators=conf.rf_n_estimators,
            random_state=utils.seed,
            max_depth=conf.rf_max_depth,
            min_samples_split=conf.rf_min_split,
            max_features=conf.rf_max_features,
            n_jobs=conf.rf_n_jobs,
            class_weight=conf.rf_class_weight
        )
    elif head_type.upper() == "LR":
        return LogisticRegression(
            max_iter=conf.lr_max_iter,
            random_state=utils.seed,
            solver=conf.solver,
            penalty=conf.penalty,
            l1_ratio=conf.lr_l1_ratio,
            class_weight=conf.lr_class_weight
        )
    elif head_type.upper() == "XGB":
        return XGBClassifier(
            random_state=utils.seed,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        raise ValueError(f"Unknown head_type {head_type}")


def find_best_threshold(model, X_val_feat, y_val, thresholds=np.linspace(0, 1, 101)):
    probs = model.predict_proba(X_val_feat)[:, 1]
    best_thr, best_bal = 0.5, 0.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        bal = balanced_accuracy_score(y_val, preds)
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    logging.info(f"Best threshold: {best_thr:.2f}  bal_acc={best_bal:.3f}")
    return best_thr


def main():
    do_analysis = False

    # ─── Logging & directories ─────────────────────────────────────────
    models_utils.config_logging(log_dir="logs", log_filename="train_rocket.log")
    os.makedirs(conf.output_model_base_dir, exist_ok=True)
    if conf.save_plots:
        os.makedirs(conf.output_dir_plots, exist_ok=True)

    print("🔧 Initializing ROCKET Training Pipeline")
    print("=" * 60)

    # ─── Load datasets ──────────────────────────────────────────────────
    print("\n📂 Loading datasets...")
    datasets = {}
    for split, path in [("train", utils.train_path),
                        ("val",   utils.validation_path),
                        ("test",  utils.test_path)]:
        print(f"Loading {split} set from {path}")
        datasets[split] = BoaOpenFaceDataset.load_dataset(path)
    train_ds = datasets["train"]
    val_ds   = datasets["val"]
    test_ds  = datasets["test"]

    # ─── Compute statistics ─────────────────────────────────────────────
    print("\n📊 Computing dataset statistics...")
    train_ds.compute_statistics()
    val_ds.trial_id_stats  = train_ds.trial_id_stats
    test_ds.trial_id_stats = train_ds.trial_id_stats

    # ─── Data quality analysis ─────────────────────────────────────────
    if do_analysis: dataset_utils.analyze_data_quality(datasets)

    # ─── NORMALIZATION STEP ───────────────────────────────────────────
    print("\n🔄 Computing normalization parameters on TRAIN set...")
    norm_params = dataset_utils.compute_normalization_params(train_ds)

    print("🔄 Applying normalization to all splits (train/val/test)...")
    for ds in (train_ds, val_ds, test_ds):
        dataset_utils.apply_normalization(ds, norm_params)

    # ─── Verify normalization effects ──────────────────────────────────
    if do_analysis: print("\n📊 Post-normalization statistics:")
    if do_analysis: dataset_utils.analyze_data_quality(datasets)

    # ─── Convert Dataset objects to NumPy arrays ──────────────────────
    def ds_to_numpy(ds):
        X_list, y_list = [], []
        for inst in ds.instances:
            g = inst.gaze_info     # shape [L, G]
            h = inst.head_info     # shape [L, H]
            f = inst.face_info     # shape [L, F]
            X = np.concatenate([g, h, f], axis=1)  # shape [L, G+H+F]
            X_list.append(X.T)     # convert to [C, L]
            y_list.append(inst.trial_type)
        X_arr = np.stack(X_list, axis=0)  # [N, C, L]
        y_arr = np.array(y_list)
        return X_arr, y_arr

    print("\n🔄 Converting datasets to NumPy arrays...")
    X_train, y_train = ds_to_numpy(train_ds)
    X_val,   y_val   = ds_to_numpy(val_ds)
    X_test,  y_test  = ds_to_numpy(test_ds)

    # ─── ROCKET transform ─────────────────────────────────────────────
    print("\n🔄 Fitting and transforming features with ROCKET...")
    rocket = Rocket(num_kernels=conf.rocket_kernels,
                    random_state=utils.seed,
                    n_jobs=-1)
    X_train_feat = rocket.fit_transform(X_train)
    X_val_feat   = rocket.transform(X_val)
    X_test_feat  = rocket.transform(X_test)

    # ─── Train classifier ─────────────────────────────────────────────
    print(f"\n🧠 Training classifier: {conf.classifier}")
    clf = classification_model(conf.classifier)
    clf.fit(X_train_feat, y_train)

    # ─── Find best threshold on validation set ────────────────────────
    print("\n🔎 Finding best threshold on validation set...")
    best_thr = find_best_threshold(clf, X_val_feat, y_val)

    # ─── Final Test Evaluation ────────────────────────────────────────
    print("\n🎯 Evaluating on test set...")
    probs_test   = clf.predict_proba(X_test_feat)[:, 1]
    y_pred_test  = (probs_test >= best_thr).astype(int)
    test_metrics = models_utils.calculate_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        y_prob=probs_test
        )

    logging.info(f"=== FINAL TEST RESULTS (thr={best_thr:.2f}) ===")
    for metric_name, value in test_metrics.items():
        if metric_name not in ('fpr', 'tpr', 'conf_matrix'):
            logging.info(f"{metric_name:<20}: {value:.4f}")
    logging.info(f"'conf_matrix': {test_metrics['conf_matrix']}")

    # ─── Save model, ROCKET transformer, threshold, and metrics ──────
    state = {
        'rocket':        rocket,
        'classifier':    clf,
        'best_threshold': best_thr,
        'params': {
            'rocket_kernels': conf.rocket_kernels,
            'classifier':     conf.classifier
        },
        'test_metrics':  test_metrics
    }
    out_path = os.path.join(conf.output_model_base_dir, 'rocket_model.joblib')
    joblib.dump(state, out_path)
    logging.info(f"Saved state to {out_path}")

    # ─── Save plots (ROC + Confusion Matrix) ─────────────────────────
    if conf.save_plots:
        print("\n📈 Saving plots...")
        cm_path  = os.path.join(conf.output_dir_plots, "cm_rocket.png")
        roc_path = os.path.join(conf.output_dir_plots, "roc_rocket.png")

        plot_utils.save_confusion_matrix(
            test_metrics['conf_matrix'],
            cm_path,
            f"ROCKET (thr={best_thr:.2f})"
        )
        plot_utils.plot_roc_curve(
            test_metrics['fpr'],
            test_metrics['tpr'],
            test_metrics['roc_auc'],
            roc_path
        )
        print(f"✅ Plots saved to: {conf.output_dir_plots}")
        logging.info(f"Saved confusion matrix to {cm_path}")
        logging.info(f"Saved ROC curve to {roc_path}")


if __name__ == '__main__':
    main()
