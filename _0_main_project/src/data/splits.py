"""Subject-grouped dataset splitting.

Implements the strict subject-independent hold-out from PROJECT_STATE §1.2:
no infant ever appears in more than one of {train, val, test}.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..configs.data import SplitConfig
from .dataset import BambinoDataset


def subject_grouped_split(
    dataset: BambinoDataset,
    cfg: SplitConfig,
    seed: int = 2025,
) -> Dict[str, List[str]]:
    """Partition subject IDs into train/val/test, optionally class-stratified.

    Stratification is by the dominant trial type per subject, since the
    Stimulus/Control prior is fixed (~80/20) at the protocol level and
    therefore similar across subjects. We still stratify to avoid concentrating
    rare configurations (e.g. Female + young + few stimuli) in one split.
    """
    rng = np.random.default_rng(seed)
    subject_to_label: Dict[str, int] = {}
    for inst in dataset.instances:
        # 1 if subject has at least one stimulus trial (always true in practice,
        # but kept for completeness); falls back to majority class otherwise.
        subject_to_label.setdefault(inst.pt_id, inst.trial_type)

    subjects = sorted(subject_to_label.keys())
    if cfg.stratify_by_class:
        by_label: Dict[int, List[str]] = {}
        for s in subjects:
            by_label.setdefault(subject_to_label[s], []).append(s)
        train, val, test = [], [], []
        for _, group in by_label.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(round(n * cfg.train_frac))
            n_val = int(round(n * cfg.val_frac))
            train += shuffled[:n_train]
            val += shuffled[n_train:n_train + n_val]
            test += shuffled[n_train + n_val:]
    else:
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * cfg.train_frac))
        n_val = int(round(n * cfg.val_frac))
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

    return {"train": train, "val": val, "test": test}


def assert_no_subject_leakage(splits: Dict[str, Sequence[str]]) -> None:
    """Raise if any subject appears in more than one split."""
    counts = Counter()
    for ids in splits.values():
        counts.update(ids)
    leaks = {s: c for s, c in counts.items() if c > 1}
    if leaks:
        raise AssertionError(f"Subject leakage detected across splits: {leaks}")


def split_dataset(
    dataset: BambinoDataset,
    cfg: SplitConfig,
    seed: int = 2025,
) -> Tuple[BambinoDataset, BambinoDataset, BambinoDataset]:
    """Apply a subject-grouped split and return (train, val, test) datasets."""
    splits = subject_grouped_split(dataset, cfg, seed=seed)
    assert_no_subject_leakage(splits)
    train = dataset.filter_by_subjects(splits["train"])
    val = dataset.filter_by_subjects(splits["val"])
    test = dataset.filter_by_subjects(splits["test"])
    return train, val, test
