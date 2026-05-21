"""Subject-grouped split: no leakage allowed."""
from __future__ import annotations

import pytest

from src.configs import SplitConfig
from src.data.splits import (
    assert_no_subject_leakage,
    split_dataset,
    subject_grouped_split,
)


def test_no_subject_appears_in_two_splits(synthetic_dataset):
    cfg = SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2)
    splits = subject_grouped_split(synthetic_dataset, cfg, seed=0)
    # Should not raise
    assert_no_subject_leakage(splits)


def test_split_dataset_returns_disjoint_subject_sets(synthetic_dataset):
    cfg = SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2)
    train, val, test = split_dataset(synthetic_dataset, cfg, seed=0)
    train_subj = set(train.subject_ids)
    val_subj = set(val.subject_ids)
    test_subj = set(test.subject_ids)
    assert train_subj.isdisjoint(val_subj)
    assert train_subj.isdisjoint(test_subj)
    assert val_subj.isdisjoint(test_subj)


def test_assert_no_subject_leakage_raises_on_overlap():
    with pytest.raises(AssertionError):
        assert_no_subject_leakage({"train": ["A", "B"], "val": ["B"], "test": ["C"]})
