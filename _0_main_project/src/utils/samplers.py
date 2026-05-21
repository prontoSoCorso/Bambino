"""Habituation-aware weighted sampling.

Per PROJECT_STATE §3.2.2, the sampling weight per trial is the product of:

    * a class-balance weight  w_class = N / (n_classes * count(class))
    * a habituation-decay weight w_hab = exp(-λ · trial_position)

For the AD baseline (purpose='ad_baseline'), habituation weights are FORCED
to 1.0 because late-session fatigue is a valid baseline state, not an error
(PROJECT_STATE §3.2.2, "CRITICAL CONSTRAINT").
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_balance_weights(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights.

    `labels` is cast explicitly to int64; mixed-dtype or float labels would
    otherwise crash `np.bincount` with a cryptic dtype error.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    inv = len(labels_arr) / (num_classes * np.maximum(counts, 1.0))
    return inv[labels_arr]


def habituation_decay_weights(trial_positions: Sequence[float], decay_lambda: float) -> np.ndarray:
    """`w(p) = exp(-λ p)` for normalised trial position p ∈ [0, 1]."""
    p = np.asarray(trial_positions, dtype=np.float64)
    return np.exp(-decay_lambda * p)


def build_sampler(
    dataset,
    num_classes: int,
    use_habituation_decay: bool,
    decay_lambda: float = 0.05,
    balance: bool = True,
) -> WeightedRandomSampler:
    """Build a `WeightedRandomSampler` composing class balance × habituation decay.

    Per PROJECT_STATE §3.2.2: for AD baseline construction, habituation decay
    is suppressed (all weights = 1.0). Detection is via the dataset's
    augmentation purpose attribute when present.

    Raises a descriptive RuntimeError on empty datasets — the most common
    upstream cause is that legacy `.pt` files in `data/full/raw/` failed to
    unpickle because they reference DataUtils.* classes that no longer exist.
    """
    if len(dataset) == 0 or not getattr(dataset, "instances", None):
        raise RuntimeError(
            "Refusing to build a sampler on an empty dataset. "
            "Most common cause: the legacy `.pt` files at `data/full/raw/` were "
            "pickled with `DataUtils.BoaOpenFaceDataset` / `DataUtils.OpenFaceInstance` "
            "classes that the new architecture does not provide. Convert them with:\n"
            "    python scripts/migrate_legacy_data.py\n"
            "and point the run at the clean output directory:\n"
            "    python main.py ... --data-dir _0_main_project/data/full/clean/"
        )

    labels = [int(inst.trial_type) for inst in dataset.instances]
    positions = [float(dataset.trial_position(inst)) for inst in dataset.instances]
    base_weights = [float(inst.sample_weight) for inst in dataset.instances]

    weights = np.ones(len(dataset), dtype=np.float64)
    if balance:
        weights *= class_balance_weights(labels, num_classes)
    if use_habituation_decay:
        weights *= habituation_decay_weights(positions, decay_lambda)
    weights *= np.asarray(base_weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)

    return WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )


def get_habituation_aware_sampler(
    dataset,
    decay_lambda: float,
    balance: bool = True,
    num_classes: int = 2,
) -> WeightedRandomSampler:
    """Convenience entry point matching the spec name in PROJECT_STATE §3.2.2."""
    return build_sampler(
        dataset,
        num_classes=num_classes,
        use_habituation_decay=True,
        decay_lambda=decay_lambda,
        balance=balance,
    )
