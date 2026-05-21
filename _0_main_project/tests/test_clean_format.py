"""Round-trip tests for the clean `.pt` format.

The clean format MUST NOT depend on any custom Python class to load. We
verify this by:

    1. Building synthetic OpenFaceInstance objects.
    2. Saving them to disk via `OpenFaceInstance.to_dict` + `torch.save`.
    3. Loading them back via `BambinoDataset.from_clean_pt`.
    4. Confirming every field round-trips and shape contracts hold.

We also confirm that the empty-dataset failure path in `samplers.build_sampler`
raises the descriptive RuntimeError pointing at the migration script.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.configs import DataConfig
from src.data.dataset import BambinoDataset
from src.data.instance import OpenFaceInstance
from src.utils.samplers import build_sampler


def test_to_dict_from_dict_round_trip(synthetic_instances):
    for src in synthetic_instances[:3]:
        d = src.to_dict()
        # Pure-Python types only.
        assert isinstance(d["pt_id"], str)
        assert isinstance(d["sex"], int)
        assert isinstance(d["age"], float)
        assert isinstance(d["gaze_info"], np.ndarray)
        rebuilt = OpenFaceInstance.from_dict(d)
        assert rebuilt.pt_id == src.pt_id
        assert rebuilt.trial_id == src.trial_id
        assert rebuilt.trial_type == src.trial_type
        for key in ("g", "h", "f"):
            np.testing.assert_array_equal(rebuilt.get_modality(key), src.get_modality(key))


def test_clean_pt_format_loads_without_custom_classes(synthetic_instances, data_cfg, tmp_path):
    """Save → reload via from_clean_pt; verify counts and per-trial shapes."""
    payload = {
        "instances": [s.to_dict() for s in synthetic_instances],
        "metadata": {"source_path": "synthetic", "frame_rate": 25,
                     "modality_dims": {"g": 8, "h": 13, "f": 17}, "split": "test"},
    }
    out = tmp_path / "clean_test.pt"
    torch.save(payload, out)

    ds = BambinoDataset.from_clean_pt(str(out), data_cfg)
    assert len(ds) == len(synthetic_instances)
    item = ds[0]
    assert item["x_pre"]["g"].shape == (data_cfg.window.pre_stim_frames, 8)
    assert item["x_post"]["g"].shape == (data_cfg.window.post_stim_frames, 8)


def test_from_clean_pt_rejects_wrong_format(data_cfg, tmp_path):
    """Missing 'instances' key → descriptive RuntimeError naming the script."""
    bogus = tmp_path / "bogus.pt"
    torch.save({"not_instances": []}, bogus)
    with pytest.raises(RuntimeError, match="migrate_legacy_data"):
        BambinoDataset.from_clean_pt(str(bogus), data_cfg)


def test_empty_dataset_sampler_raises_clean_error(data_cfg):
    """Building a sampler on an empty dataset must point at the migration."""
    empty = BambinoDataset([], data_cfg)
    with pytest.raises(RuntimeError, match="migrate_legacy_data"):
        build_sampler(empty, num_classes=2, use_habituation_decay=False)


def test_class_balance_weights_handles_int_cast():
    """`np.bincount` requires int64 — verify samplers cast even when given floats."""
    from src.utils.samplers import class_balance_weights
    weights = class_balance_weights([0.0, 1.0, 1.0, 0.0], num_classes=2)
    # Two classes, equal counts → equal inverse-frequency weights = 1.0
    assert np.allclose(weights, 1.0)
