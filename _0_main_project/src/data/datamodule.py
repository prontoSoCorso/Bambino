"""LightningDataModule for the BAMBINO dataset.

Responsibilities:

    * Load instances from CSV (or pre-pickled .pt blobs) once in `prepare_data`.
    * Subject-grouped split into train/val/test inside `setup`.
    * Apply normalisation according to `data_cfg.baseline_norm_mode`.
    * Optionally augment the TRAIN split (val/test never augmented).
    * Build DataLoaders with optional habituation-aware sampling.
"""
from __future__ import annotations

import os
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..configs import AugmentationConfig, DataConfig, RunConfig, TrainerConfig
from ..utils.samplers import build_sampler
from .augmentation import build_augmented_dataset
from .dataset import BambinoDataset
from .normalization import normalize_datasets
from .splits import split_dataset


def _collate(batch):
    """Custom collate: stacks tensor dicts, lists scalar metadata."""
    import torch

    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "meta":
            # Convert list-of-dict to dict-of-list
            out[k] = {mk: [m[mk] for m in vals] for mk in vals[0].keys()}
        elif k in ("x_post", "x_pre"):
            out[k] = {mod: torch.stack([d[mod] for d in vals], dim=0) for mod in vals[0].keys()}
        else:
            out[k] = torch.stack(vals, dim=0)
    return out


class BambinoDataModule(pl.LightningDataModule):
    """End-to-end data orchestration for one experiment.

    The Lightning lifecycle:

        prepare_data()  → load CSVs → BambinoDataset (called once, on rank 0)
        setup(stage)    → split by subject → normalise → augment train split
        {train,val,test}_dataloader() → DataLoader with appropriate sampler
    """

    def __init__(self, run_cfg: RunConfig, csv_dir: Optional[str] = None):
        super().__init__()
        self.run_cfg = run_cfg
        self.data_cfg: DataConfig = run_cfg.data
        self.aug_cfg: AugmentationConfig = run_cfg.augmentation
        self.trainer_cfg: TrainerConfig = run_cfg.trainer
        self.csv_dir = csv_dir or run_cfg.base.paths.data_root

        self._raw: Optional[BambinoDataset] = None
        self.train_set: Optional[BambinoDataset] = None
        self.val_set: Optional[BambinoDataset] = None
        self.test_set: Optional[BambinoDataset] = None

    # ──────────────────────────────────────────────────────────────────────
    def prepare_data(self) -> None:
        """Load instances into a single BambinoDataset.

        Called once per node by Lightning. The loader prefers, in order:

            1. A combined clean blob at `<data_dir>/bambino_clean.pt`
               (output of `scripts/migrate_legacy_data.py`).
            2. Per-split clean files (`training_set.pt`, `validation_set.pt`,
               `test_set.pt`) under the SAME directory — concatenated.
            3. CSV scan as a last resort (legacy path; useful for fresh
               OpenFace exports that were never pickled).

        If none of the above are found, a descriptive RuntimeError points the
        user at the migration script.
        """
        if not os.path.isdir(self.csv_dir):
            raise FileNotFoundError(
                f"Data directory does not exist: {self.csv_dir}. "
                "Update RunConfig.base.paths.data_root or provide csv_dir explicitly."
            )

        # Branch A: clean `.pt` files (preferred).
        combined = os.path.join(self.csv_dir, "bambino_clean.pt")
        per_split_clean = [
            os.path.join(self.csv_dir, n)
            for n in ("training_set.pt", "validation_set.pt", "test_set.pt")
        ]
        has_clean = os.path.isfile(combined) or any(os.path.isfile(p) for p in per_split_clean)

        if has_clean:
            self._raw = BambinoDataset.from_clean_dir(self.csv_dir, self.data_cfg)
            return

        # Branch B: CSV scan (no .pt files at all).
        csv_files = [n for n in os.listdir(self.csv_dir) if n.endswith(".csv")]
        if csv_files:
            self._raw = BambinoDataset.from_csv_dir(self.csv_dir, self.data_cfg)
            return

        raise RuntimeError(
            f"No usable data found in {self.csv_dir}.\n"
            "Expected one of:\n"
            "    1. Clean .pt files (bambino_clean.pt OR {training,validation,test}_set.pt)\n"
            "       produced by `python scripts/migrate_legacy_data.py`.\n"
            "    2. Raw OpenFace CSVs (*.csv) for the from_csv_dir loader.\n"
            "If you have legacy pickles holding DataUtils.* class instances, run:\n"
            "    python scripts/migrate_legacy_data.py --input <legacy_dir> --output "
            f"{self.csv_dir}"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Split, normalise, augment.

        Lightning calls this for stages {fit, validate, test, predict}. We do
        the full pipeline regardless of stage so the same DataModule serves
        all stages without redundant CSV parsing.
        """
        if self._raw is None:
            self.prepare_data()
        assert self._raw is not None

        train, val, test = split_dataset(self._raw, self.data_cfg.split, seed=self.run_cfg.base.seed)
        normalize_datasets(train, val, test, mode=self.data_cfg.baseline_norm_mode)

        # Augmentation applies to train only — val/test are pristine.
        if self.aug_cfg.n_aug_per_positive + self.aug_cfg.n_aug_per_negative > 0:
            train = build_augmented_dataset(train, self.aug_cfg)

        self.train_set = train
        self.val_set = val
        self.test_set = test

    # ──────────────────────────────────────────────────────────────────────
    def train_dataloader(self) -> DataLoader:
        """Train loader. Applies habituation-aware sampling when enabled.

        See `utils.samplers.build_sampler` for the composition rule:
            sample_weight = class_balance_weight × habituation_decay_weight
        """
        assert self.train_set is not None
        sampler = build_sampler(
            self.train_set,
            num_classes=self.run_cfg.base.num_classes,
            use_habituation_decay=self.trainer_cfg.use_habituation_decay_weights,
            decay_lambda=self.trainer_cfg.habituation_decay_lambda,
            balance=True,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.data_cfg.batch_size,
            sampler=sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=_collate,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None
        return DataLoader(
            self.val_set,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=_collate,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            collate_fn=_collate,
        )
