import os
import sys
import logging
import numpy as np
import joblib
import itertools
from sklearn.metrics import matthews_corrcoef
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


def find_best_threshold_mcc(model, X_val_feat, y_val, thresholds=np.linspace(0, 1, 101)):
    probs = model.predict_proba(X_val_feat)[:, 1]
    best_thr, best_mcc = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        mcc = matthews_corrcoef(y_val, preds)
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    logging.info(f"    → best mcc={best_mcc:.4f} at thr={best_thr:.2f}")
    return best_thr, best_mcc


def ds_to_numpy(ds, modalities):
    """
    Converts a BoaOpenFaceDataset into X: [N, C, L] and y: [N], for the given modalities list.
    """
    modality_map = {
        'g': 'gaze_info',
        'h': 'head_info',
        'f': 'face_info'
        # add more if needed
    }
    X_list, y_list = [], []
    for inst in ds.instances:
        mats = []
        for m in modalities:
            attr = modality_map.get(m)
            if attr is None:
                raise ValueError(f"Unknown modality '{m}'")
            mats.append(getattr(inst, attr))  # [L, D_m]
        X = np.concatenate(mats, axis=1)  # [L, sum(D_m)]
        X_list.append(X.T)                # [C, L]
        y_list.append(inst.trial_type)
    return np.stack(X_list, axis=0), np.array(y_list)


def main(dataset_type):
    # ─── Logging & dirs ────────────────────────────────────────────────
    models_utils.config_logging(log_dir="logs", log_filename="train_rocket_grid.log")
    os.makedirs(conf.output_model_base_dir, exist_ok=True)
    if conf.save_plots:
        os.makedirs(conf.output_dir_plots, exist_ok=True)

    print("🔧 Starting ROCKET Grid Search (MCC)…")
    print("=" * 60)

    # ─── Load & Prep Data ──────────────────────────────────────────────
    print("\n📂 Loading datasets...")
    modalities_grid = conf.modality_combinations   # e.g. [['g'], ['f'], ['g','f'], …]
    kernels_grid    = conf.rocket_kernels_list     # e.g. [10_000, 20_000, 50_000]

    datasets = {}
    modalities = list(conf.modality_dims.keys())
    for split, path in [("train", utils.get_dataset_path(dataset_type, utils.train_filename)),
                        ("val",   utils.get_dataset_path(dataset_type, utils.validation_filename)),
                        ("test",  utils.get_dataset_path(dataset_type, utils.test_filename))]:
        datasets[split] = BoaOpenFaceDataset.load_dataset(path, modalities=modalities)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # compute normalization on train only, apply to all
    norm_params = dataset_utils.compute_normalization_params(train_ds)
    for ds in (train_ds, val_ds, test_ds):
        dataset_utils.apply_normalization(ds, norm_params)

    # convert all to numpy once, then pick modalities
    full_X = {
        'train': ds_to_numpy(train_ds, modalities=modalities),
        'val':   ds_to_numpy(val_ds,   modalities=modalities),
        'test':  ds_to_numpy(test_ds,  modalities=modalities)
        }

    best_overall = {
        'mcc':    0,
        'modalities': None,
        'n_kernels': None,
        'threshold': None,
        'rocket': None,
        'clf': None
        }

    # ─── Grid Search ───────────────────────────────────────────────────
    for mods, n_kern in itertools.product(modalities_grid, kernels_grid):
        print(f"\n🔍 Trying modalities={mods}, kernels={n_kern}")
        # slice numpy arrays per modality
        X_train, y_train = ds_to_numpy(train_ds, mods)
        X_val,   y_val   = ds_to_numpy(val_ds,   mods)

        # ROCKET
        rocket = Rocket(num_kernels=n_kern,
                        random_state=utils.seed,
                        n_jobs=-1)
        X_train_feat = rocket.fit_transform(X_train)
        X_val_feat   = rocket.transform(X_val)

        # classifier
        clf = classification_model(conf.classifier)
        clf.fit(X_train_feat, y_train)

        # threshold by MCC
        thr, mcc = find_best_threshold_mcc(clf, X_val_feat, y_val, thresholds=np.linspace(0,0.8,101))
        if abs(mcc) > abs(best_overall['mcc']):
            best_overall.update({
                'mcc': mcc,
                'modalities': mods,
                'n_kernels': n_kern,
                'threshold': thr,
                'rocket': rocket,
                'clf': clf
            })

    # ─── Final Evaluate on Test ────────────────────────────────────────
    print("\n🎯 Best config:")
    print(f"    modalities: {best_overall['modalities']}")
    print(f"    n_kernels : {best_overall['n_kernels']}")
    print(f"    val MCC   : {best_overall['mcc']:.4f} at thr={best_overall['threshold']:.2f}")

    # prepare test features using best rocket
    X_test, y_test = ds_to_numpy(test_ds, best_overall['modalities'])
    X_test_feat = best_overall['rocket'].transform(X_test)
    probs_test  = best_overall['clf'].predict_proba(X_test_feat)[:, 1]
    y_pred_test = (probs_test >= best_overall['threshold']).astype(int)

    test_metrics = models_utils.calculate_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        y_prob=probs_test
    )
    print("=" *60)
    logging.info(f"=== FINAL TEST RESULTS (thr={best_overall['threshold']:.2f}) ===")
    for metric_name, value in test_metrics.items():
        if metric_name not in ('fpr', 'tpr', 'conf_matrix'):
            logging.info(f"{metric_name:<20}: {value:.4f}")
    logging.info(f"'conf_matrix': {test_metrics['conf_matrix']}")

    # log & save
    state = {
        'best_overall': best_overall,
        'test_metrics': test_metrics,
        'params': {
            'modality': best_overall['modalities'],
            'rocket_kernels': best_overall['n_kernels'],
            'classifier': conf.classifier
        }
    }
    out_path = os.path.join(conf.output_model_base_dir, 'rocket_grid_search.joblib')
    joblib.dump(state, out_path)
    print(f"\n✅ Saved best state & metrics to {out_path}")

    # optional: save plots
    if conf.save_plots:
        cm_path  = os.path.join(conf.output_dir_plots, "cm_rocket_best.png")
        roc_path = os.path.join(conf.output_dir_plots, "roc_rocket_best.png")
        plot_utils.save_confusion_matrix(
            test_metrics['conf_matrix'], cm_path,
            f"Best ROCKET (thr={best_overall['threshold']:.2f})"
        )
        plot_utils.plot_roc_curve(
            test_metrics['fpr'], test_metrics['tpr'],
            test_metrics['roc_auc'], roc_path
        )
        print(f"📈 Saved best ROC & CM to {conf.output_dir_plots}")


if __name__ == '__main__':
    main(dataset_type=conf.dataset_type)
