"""BAMBINO training & evaluation entrypoint.

The `--model` flag accepts every active model family from PROJECT_STATE §2.1:

    Sklearn / TabPFN paths (no Lightning, fast):
        logreg, minirocket,
        moment_histgb, moment_logreg, moment_tabpfn, moment_pca_tabpfn

    Lightning paths:
        inception_time, film_cnn, resnet_gasf, moment_mlp, anomaly_detector

Loggers: TensorBoard + CSV under `results/<run_id>/{tb_logs,csv_logs}/`.
Sklearn artifacts: `results/<run_id>/metrics.json`.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
from dataclasses import asdict, replace
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.configs import (
    AnomalyDetectorConfig,
    AugmentationConfig,
    DataConfig,
    FiLMCNNConfig,
    InceptionTimeConfig,
    LogRegConfig,
    MiniRocketConfig,
    MomentConfig,
    ResNetGASFConfig,
    RunConfig,
    TrainerConfig,
)
from src.data import (
    BambinoDataModule,
    BambinoDataset,
    build_feature_matrix,
)
from src.utils import seed_everything


# ─── Argparse ──────────────────────────────────────────────────────────────
ALL_MODELS = (
    "logreg",
    "minirocket",
    "inception_time",
    "film_cnn",
    "resnet_gasf",
    "moment_mlp",
    "moment_histgb",
    "moment_logreg",
    "moment_tabpfn",
    "moment_pca_tabpfn",
    "anomaly_detector",
)

SKLEARN_PATH_MODELS = {
    "logreg",
    "minirocket",
    "moment_histgb",
    "moment_logreg",
    "moment_tabpfn",
    "moment_pca_tabpfn",
}

MOMENT_SKLEARN_HEAD_MAP = {
    "moment_histgb": "embeddings+histgb",
    "moment_logreg": "embeddings+logreg",
    "moment_tabpfn": "tabpfn",
    "moment_pca_tabpfn": "pca+tabpfn",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BAMBINO training entrypoint")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--run-id", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--baseline-norm-mode", choices=["global", "per_trial", "per_subject"], default="global")
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--use-pre-stim-context", action="store_true")
    p.add_argument("--use-habituation-decay", action="store_true")
    p.add_argument("--decay-lambda", type=float, default=0.05)
    return p.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Wire CLI flags into a frozen RunConfig dataclass tree."""
    if args.model == "logreg":
        model_cfg = LogRegConfig()
    elif args.model == "minirocket":
        model_cfg = MiniRocketConfig()
    elif args.model == "inception_time":
        model_cfg = InceptionTimeConfig(use_pre_stim_context=args.use_pre_stim_context)
    elif args.model == "film_cnn":
        model_cfg = FiLMCNNConfig()
    elif args.model == "resnet_gasf":
        model_cfg = ResNetGASFConfig()
    elif args.model == "moment_mlp":
        model_cfg = MomentConfig(name="moment_mlp", head="mlp",
                                 contrast_pre_post=args.use_pre_stim_context)
    elif args.model == "moment_histgb":
        model_cfg = MomentConfig(name="moment_histgb", head="histgb")
    elif args.model == "moment_logreg":
        model_cfg = MomentConfig(name="moment_logreg", head="logreg")
    elif args.model == "moment_tabpfn":
        model_cfg = MomentConfig(name="moment_tabpfn", head="tabpfn")
    elif args.model == "moment_pca_tabpfn":
        model_cfg = MomentConfig(name="moment_pca_tabpfn", head="pca_tabpfn")
    elif args.model == "anomaly_detector":
        model_cfg = AnomalyDetectorConfig()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    run = RunConfig(run_id=args.run_id, model=model_cfg)
    run = replace(run, base=replace(run.base, seed=args.seed))
    run = replace(run, data=replace(
        run.data,
        baseline_norm_mode=args.baseline_norm_mode,
        expose_pre_stim=True,
    ))
    run = replace(run, trainer=replace(
        run.trainer,
        use_habituation_decay_weights=args.use_habituation_decay,
        habituation_decay_lambda=args.decay_lambda,
        max_epochs=args.max_epochs or run.trainer.max_epochs,
    ))
    if args.no_augmentation:
        run = replace(run, augmentation=AugmentationConfig(
            n_aug_per_positive=0, n_aug_per_negative=0,
        ))
    return run


# ─── Dynamic monitor resolution ────────────────────────────────────────────
def _resolve_monitor(run_cfg: RunConfig, model: pl.LightningModule) -> tuple:
    """Determine (metric, mode) for EarlyStopping / ModelCheckpoint.

    Priority:
        1. CLI/RunConfig override (`trainer.monitor` set).
        2. Model's declared `monitor_metric` / `monitor_mode` attrs (if any).
        3. Hard fallback: `val/balanced_accuracy` / `max`.
    """
    if run_cfg.trainer.monitor is not None:
        return run_cfg.trainer.monitor, run_cfg.trainer.monitor_mode or "max"
    metric = getattr(model, "monitor_metric", None) or "val/balanced_accuracy"
    mode = getattr(model, "monitor_mode", None) or "max"
    return metric, mode


def _make_loggers(run_cfg: RunConfig):
    base = run_cfg.base.paths.results_root
    return [
        TensorBoardLogger(save_dir=os.path.join(base, "tb_logs"), name=run_cfg.run_id),
        CSVLogger(save_dir=os.path.join(base, "csv_logs"), name=run_cfg.run_id),
    ]


def _make_callbacks(run_cfg: RunConfig, monitor: str, mode: str):
    cb = []
    if run_cfg.trainer.early_stopping:
        cb.append(EarlyStopping(monitor=monitor, mode=mode, patience=run_cfg.trainer.patience))
    cb.append(ModelCheckpoint(
        dirpath=os.path.join(run_cfg.base.paths.results_root, run_cfg.run_id, "checkpoints"),
        monitor=monitor, mode=mode, save_top_k=run_cfg.trainer.save_top_k,
        filename="best-{epoch:02d}-{step}",
    ))
    return cb


def _make_trainer(run_cfg: RunConfig, monitor: str, mode: str) -> pl.Trainer:
    return pl.Trainer(
        accelerator=run_cfg.trainer.accelerator,
        devices=run_cfg.trainer.devices,
        precision=run_cfg.trainer.precision,
        max_epochs=run_cfg.trainer.max_epochs,
        check_val_every_n_epoch=run_cfg.trainer.check_val_every_n_epoch,
        gradient_clip_val=run_cfg.trainer.gradient_clip_val,
        deterministic=run_cfg.trainer.deterministic,
        log_every_n_steps=run_cfg.trainer.log_every_n_steps,
        logger=_make_loggers(run_cfg),
        callbacks=_make_callbacks(run_cfg, monitor, mode),
    )


# ─── Sklearn paths ──────────────────────────────────────────────────────────
def _save_sklearn_metrics(run_cfg: RunConfig, payload: Dict[str, Any]) -> None:
    out = os.path.join(run_cfg.base.paths.results_root, run_cfg.run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[{run_cfg.model.name}] metrics → {out}/metrics.json")


def run_logreg(run_cfg: RunConfig, dm: BambinoDataModule) -> None:
    from src.models import fit_logreg

    dm.prepare_data(); dm.setup("fit")
    X_tr, y_tr, w_tr, _ = build_feature_matrix(dm.train_set)
    X_va, y_va, _, _ = build_feature_matrix(dm.val_set)
    X_te, y_te, _, _ = build_feature_matrix(dm.test_set)
    pos_te = np.asarray(
        [dm.test_set.trial_position(i) for i in dm.test_set.instances], dtype=np.float32
    )
    res = fit_logreg(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        test_trial_positions=pos_te,
        cfg=run_cfg.model,
        sample_weight_train=w_tr,
        seed=run_cfg.base.seed,
    )
    _save_sklearn_metrics(run_cfg, {
        "test": res.test_metrics,
        "bootstrap_auc": res.bootstrap_auc,
        "bucket": res.bucket_metrics,
        "selected_C": res.selected_C,
        "n_nonzero_coefs": res.n_nonzero_coefs,
    })
    print(f"[logreg] test: {res.test_metrics}")


def run_minirocket(run_cfg: RunConfig, dm: BambinoDataModule) -> None:
    from src.models import fit_minirocket

    dm.prepare_data(); dm.setup("fit")
    res = fit_minirocket(dm.train_set, dm.val_set, dm.test_set, run_cfg.model, seed=run_cfg.base.seed)
    _save_sklearn_metrics(run_cfg, {
        "test": res.test_metrics,
        "bootstrap_auc": res.bootstrap_auc,
        "bucket": res.bucket_metrics,
    })
    print(f"[minirocket] test: {res.test_metrics}")


def run_moment_sklearn(run_cfg: RunConfig, dm: BambinoDataModule) -> None:
    """MOMENT embeddings → sklearn / TabPFN head (no Lightning)."""
    from src.models import (
        MomentEmbedder, extract_embeddings, fit_moment_sklearn_head, fuse_features,
    )

    head_name = MOMENT_SKLEARN_HEAD_MAP[run_cfg.model.name]
    dm.prepare_data(); dm.setup("fit")
    embedder = MomentEmbedder(run_cfg.model, run_cfg.base.modalities)

    # Extraction loop — frozen MOMENT, no grad.
    device = "cuda" if run_cfg.base.device.type == "cuda" else "cpu"

    def _emb(ds: BambinoDataset):
        loader = DataLoader(ds, batch_size=run_cfg.model.batch_size, shuffle=False,
                            collate_fn=_collate_for_moment)
        return extract_embeddings(embedder, loader,
                                  contrast_pre_post=run_cfg.model.contrast_pre_post,
                                  device=device)

    X_tr_e, y_tr, age_tr, sex_tr, _ = _emb(dm.train_set)
    X_va_e, y_va, age_va, sex_va, _ = _emb(dm.val_set)
    X_te_e, y_te, age_te, sex_te, pos_te = _emb(dm.test_set)

    X_tr = fuse_features(X_tr_e, age_tr, sex_tr)
    X_va = fuse_features(X_va_e, age_va, sex_va)
    X_te = fuse_features(X_te_e, age_te, sex_te)

    res = fit_moment_sklearn_head(
        head_name, X_tr, y_tr, X_va, y_va, X_te, y_te,
        test_trial_positions=pos_te, seed=run_cfg.base.seed,
    )
    _save_sklearn_metrics(run_cfg, {
        "test": res.test_metrics,
        "bootstrap_auc": res.bootstrap_auc,
        "bucket": res.bucket_metrics,
        "head": res.head,
    })
    print(f"[{run_cfg.model.name}] test: {res.test_metrics}")


def _collate_for_moment(batch):
    import torch as _t
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "meta":
            out[k] = {mk: [m[mk] for m in vals] for mk in vals[0].keys()}
        elif k in ("x_post", "x_pre"):
            out[k] = {mod: _t.stack([d[mod] for d in vals], dim=0) for mod in vals[0].keys()}
        else:
            out[k] = _t.stack(vals, dim=0)
    return out


# ─── Lightning paths ────────────────────────────────────────────────────────
def _build_lightning_model(run_cfg: RunConfig, dm: Optional[BambinoDataModule] = None):
    """Instantiate the matching LightningModule for `run_cfg.model.name`."""
    if run_cfg.model.name == "inception_time":
        from src.models import InceptionTimeModel
        return InceptionTimeModel(run_cfg.model, run_cfg.base.modalities)
    if run_cfg.model.name == "film_cnn":
        from src.models import FiLMCNNModel
        return FiLMCNNModel(run_cfg.model, run_cfg.base.modalities)
    if run_cfg.model.name == "moment_mlp":
        from src.models import MomentMLP
        return MomentMLP(run_cfg.model, run_cfg.base.modalities)
    if run_cfg.model.name == "anomaly_detector":
        assert dm is not None
        from src.models import SubjectConditionedAnomalyDetector
        subjects = sorted(set(
            dm.train_set.subject_ids + dm.val_set.subject_ids + dm.test_set.subject_ids
        ))
        m = SubjectConditionedAnomalyDetector(
            run_cfg.model, run_cfg.base.modalities, num_subjects=len(subjects),
        )
        m.register_subjects(subjects)
        return m
    raise ValueError(run_cfg.model.name)


def run_lightning_supervised(run_cfg: RunConfig, dm: BambinoDataModule) -> None:
    """InceptionTime / FiLM-CNN / MomentMLP / AnomalyDetector via Lightning."""
    if run_cfg.model.name == "anomaly_detector":
        dm.prepare_data(); dm.setup("fit")
    model = _build_lightning_model(run_cfg, dm)
    monitor, mode = _resolve_monitor(run_cfg, model)
    print(f"[{run_cfg.model.name}] monitoring {monitor!r} (mode={mode})")
    trainer = _make_trainer(run_cfg, monitor, mode)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


def run_resnet_gasf(run_cfg: RunConfig, dm: BambinoDataModule) -> None:
    """ResNet-GASF needs its own dataset (in-memory GASF images)."""
    from src.models import GASFDataset, GASFPCAs, ResNetGASFModel, gasf_collate

    dm.prepare_data(); dm.setup("fit")
    pcas = GASFPCAs.fit(dm.train_set)
    train_img = GASFDataset(dm.train_set, pcas, run_cfg.model.image_size)
    val_img = GASFDataset(dm.val_set, pcas, run_cfg.model.image_size)
    test_img = GASFDataset(dm.test_set, pcas, run_cfg.model.image_size)

    train_loader = DataLoader(train_img, batch_size=run_cfg.model.batch_size, shuffle=True,
                              num_workers=run_cfg.data.num_workers, collate_fn=gasf_collate)
    val_loader = DataLoader(val_img, batch_size=run_cfg.model.batch_size, shuffle=False,
                            num_workers=run_cfg.data.num_workers, collate_fn=gasf_collate)
    test_loader = DataLoader(test_img, batch_size=run_cfg.model.batch_size, shuffle=False,
                             num_workers=run_cfg.data.num_workers, collate_fn=gasf_collate)

    model = ResNetGASFModel(run_cfg.model)
    monitor, mode = _resolve_monitor(run_cfg, model)
    print(f"[resnet_gasf] monitoring {monitor!r} (mode={mode})")
    trainer = _make_trainer(run_cfg, monitor, mode)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


# ─── Entry ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_cfg = build_run_config(args)
    seed_everything(run_cfg.base.seed)

    dm = BambinoDataModule(run_cfg, csv_dir=args.data_dir)

    name = run_cfg.model.name
    if name == "logreg":
        run_logreg(run_cfg, dm)
    elif name == "minirocket":
        run_minirocket(run_cfg, dm)
    elif name in MOMENT_SKLEARN_HEAD_MAP:
        run_moment_sklearn(run_cfg, dm)
    elif name == "resnet_gasf":
        run_resnet_gasf(run_cfg, dm)
    elif name in {"inception_time", "film_cnn", "moment_mlp", "anomaly_detector"}:
        run_lightning_supervised(run_cfg, dm)
    else:
        raise ValueError(f"No runner wired for model={name}")


if __name__ == "__main__":
    main()
