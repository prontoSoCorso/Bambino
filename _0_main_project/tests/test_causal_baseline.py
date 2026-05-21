"""Tests for the causal-baseline contract (PROJECT_STATE §3.1).

If any of these fail, the AD pipeline has temporal leakage.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.configs.data import WindowConfig
from src.data.normalization import (
    apply_per_subject_baseline_normalization,
    apply_per_trial_baseline_normalization,
)
from src.models.manifold_utils import build_causal_baseline


def test_causal_baseline_excludes_future_trials(synthetic_instances, data_cfg):
    """Baseline for trial t must NOT contain any window from trials >= t."""
    pre_frames = data_cfg.window.pre_stim_frames

    # Pick subject S001, target trial t=2 (so trials 0, 1 are causal).
    target_pt = "S001"
    target_trial = 2.0

    bw = build_causal_baseline(
        synthetic_instances,
        target_pt_id=target_pt,
        target_trial_id=target_trial,
        pre_frames=pre_frames,
        include_silent_controls=True,
    )

    # Sanity: returns at least one window (trial 0 is silent control → +1
    # for its full clip and +1 for its pre-stim window = 2 windows for trial 0;
    # trial 1 is stimulus → +1 for pre-stim only = 1 window. Total >= 3.
    assert len(bw) >= 3, f"Expected >=3 baseline windows, got {len(bw)}"

    # Hard contract: rebuild future windows and confirm none of them appear.
    future_windows = []
    for inst in synthetic_instances:
        if inst.pt_id != target_pt or inst.trial_id < target_trial:
            continue
        full = np.concatenate([inst.gaze_info, inst.head_info, inst.face_info], axis=1)
        future_windows.append(full[:pre_frames])  # the pre-stim slice we'd be tempted to include

    for w in bw:
        for fw in future_windows:
            if w.shape == fw.shape and np.array_equal(w, fw):
                pytest.fail("Causal leakage: baseline contains future-trial pre-stim window.")


def test_causal_baseline_is_subject_local(synthetic_instances, data_cfg):
    """Baselines must NEVER mix in another subject's data."""
    pre_frames = data_cfg.window.pre_stim_frames
    target_pt = "S002"

    # Use a target trial so all S002 trials < t are valid sources.
    bw = build_causal_baseline(
        synthetic_instances,
        target_pt_id=target_pt,
        target_trial_id=10.0,
        pre_frames=pre_frames,
        include_silent_controls=True,
    )
    assert len(bw) > 0

    # Build the set of all OTHER subjects' pre-stim slices.
    other_pre_slices = []
    for inst in synthetic_instances:
        if inst.pt_id == target_pt:
            continue
        full = np.concatenate([inst.gaze_info, inst.head_info, inst.face_info], axis=1)
        other_pre_slices.append(full[:pre_frames])

    for w in bw:
        for o in other_pre_slices:
            if w.shape == o.shape and np.array_equal(w, o):
                pytest.fail("Cross-subject leakage: baseline contains another subject's window.")


def test_per_trial_baseline_normalization_is_causal(synthetic_dataset, data_cfg):
    """Per-trial baseline normalisation must use ONLY this trial's pre-stim slice.

    Verified by constructing a dataset where trials of the SAME subject have
    very different pre-stim distributions; per-trial norm should produce
    near-zero pre-stim mean PER TRIAL after the operation.
    """
    pre_frames = data_cfg.window.pre_stim_frames
    apply_per_trial_baseline_normalization(synthetic_dataset)

    for inst in synthetic_dataset.instances:
        for key in ("g", "h", "f"):
            arr = inst.get_modality(key)[:pre_frames]
            mean = arr.mean(axis=0)
            assert np.allclose(mean, 0.0, atol=1e-3), (
                f"Per-trial baseline mean not zero for {inst.pt_id}/{inst.trial_id}/{key}: "
                f"max |mean| = {np.max(np.abs(mean))}"
            )


def test_per_subject_baseline_normalization_uses_only_subject_data(synthetic_dataset, data_cfg):
    """After per-subject z-norm, EACH subject's concatenated baseline corpus
    has zero mean and unit std (channel-wise)."""
    pre_frames = data_cfg.window.pre_stim_frames
    apply_per_subject_baseline_normalization(synthetic_dataset)

    from collections import defaultdict
    by_subject = defaultdict(list)
    for inst in synthetic_dataset.instances:
        by_subject[inst.pt_id].append(inst)

    for pt_id, insts in by_subject.items():
        for key in ("g", "h", "f"):
            stacked = np.concatenate([inst.get_modality(key)[:pre_frames] for inst in insts], axis=0)
            assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-3), (
                f"Per-subject baseline mean not zero for {pt_id}/{key}"
            )
            # Std need not be exactly 1 because clipping at _EPS distorts low-var channels;
            # check it's at least close.
            std = stacked.std(axis=0)
            assert np.all(std > 0.5) and np.all(std < 1.5), (
                f"Per-subject baseline std out of [0.5, 1.5] for {pt_id}/{key}: {std}"
            )


def test_window_config_split_index_is_pre_frames():
    """The split index between pre and post is `pre_stim_frames` exactly.

    This is the invariant that the entire causal pipeline rests on. If this
    assertion ever fails, the AD baseline construction is wrong.
    """
    wc = WindowConfig(frame_rate=25, pre_stim_seconds=2.0, post_stim_seconds=10.0)
    assert wc.pre_stim_frames == 50
    assert wc.post_stim_frames == 250
    assert wc.total_frames == 300
    assert wc.stimulus_onset_index == wc.pre_stim_frames


def test_dataset_getitem_returns_pre_and_post_split_at_correct_index(synthetic_dataset, data_cfg):
    """`__getitem__` slices at exactly `pre_stim_frames`."""
    pre_frames = data_cfg.window.pre_stim_frames
    item = synthetic_dataset[0]
    assert "x_pre" in item and "x_post" in item
    for key in ("g", "h", "f"):
        assert item["x_pre"][key].shape[0] == pre_frames
        assert item["x_post"][key].shape[0] == data_cfg.window.post_stim_frames
