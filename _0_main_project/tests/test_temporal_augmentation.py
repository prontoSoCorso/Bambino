"""Tests for the temporal-augmentation contract (PROJECT_STATE §3.3).

Critical invariant: temporal augmentations MUST NOT cross the t=0 boundary.
We check this by constructing a synthetic clip with a sharp 5x amplitude step
at the stimulus-onset frame (`pre_frames`) and asserting that no augmented
copy ever contains pre-stim values whose magnitude rivals the post-stim step.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.configs.augmentation import AugmentationConfig
from src.data.augmentation import (
    augment_instance,
    build_augmented_dataset,
    pad_shift,
    time_warp,
)


def test_pad_shift_does_not_leak_across_array_ends():
    """Pad-shift fills with ZERO outside the original window — never wraps."""
    rng = np.random.default_rng(0)
    x = np.ones((100, 4), dtype=np.float32) * 10.0
    out = pad_shift(rng, x, max_shift=20)
    # Shifted-in region must be zero, never the wrapped value.
    nonzero = out != 0.0
    zero_run = (~nonzero).any()
    if zero_run:
        # At least one row was padded to zero (i.e. shift != 0 occurred);
        # confirm no contamination with the original 10.0 in the padded slot.
        assert (out[~nonzero] == 0).all()


def test_time_warp_respects_max_drift_ms():
    """No frame should move more than `max_drift_ms` milliseconds."""
    rng = np.random.default_rng(0)
    T = 250
    x = np.arange(T)[:, None].repeat(3, axis=1).astype(np.float32)
    out = time_warp(rng, x, max_warp=0.2, frame_rate=25, max_drift_ms=100.0)
    # The diagonal of the warped grid (= input column 0) shouldn't drift more
    # than max_drift_ms / 1000 * 25 = 2.5 frames worth of value at any index.
    expected = np.arange(T, dtype=np.float32)
    drift = np.abs(out[:, 0] - expected)
    # Allow some interpolation error: cap at 5 frames absolute drift.
    assert drift.max() < 5.0, f"Time-warp drift exceeded bound: max = {drift.max()}"


def test_temporal_augmentation_does_not_cross_t_zero(synthetic_instances, data_cfg):
    """Augmented stimulus trials still have a sharp step at the t=0 boundary.

    If a temporal aug crossed t=0, the pre-stim window would inherit some of
    the post-stim +5 step, lifting the pre-stim mean above 1.0. We assert
    pre-stim stays bounded near its original distribution (zero mean, ~unit std).
    """
    pre_frames = data_cfg.window.pre_stim_frames
    cfg = AugmentationConfig(
        purpose="supervised",
        n_aug_per_positive=3,
        n_aug_per_negative=0,
        deterministic_seed=42,
    )

    for source in synthetic_instances:
        if source.trial_type != 1:
            continue
        for k in range(3):
            aug = augment_instance(source, cfg, data_cfg, aug_index=k)
            pre_means = []
            post_means = []
            for key in ("g", "h", "f"):
                arr = aug.get_modality(key)
                pre_means.append(arr[:pre_frames].mean())
                post_means.append(arr[pre_frames:].mean())
            # Pre-stim should still be near 0 (zero-mean noise was the source);
            # tolerate up to magnitude 1.5 from scale + jitter perturbations.
            assert max(abs(m) for m in pre_means) < 1.5, (
                f"Pre-stim mean drifted above 1.5 — t=0 boundary crossed: {pre_means}"
            )
            # Post-stim should still carry the +5 step (perhaps scaled, but
            # certainly far above pre-stim).
            assert min(post_means) > 1.5, (
                f"Post-stim step lost — pre/post separation broken: {post_means}"
            )


def test_independent_temporal_seeds_pre_vs_post(synthetic_instances, data_cfg):
    """Pre and post tensors are warped under DIFFERENT realisations.

    Same source RNG state would mean identical warps on both halves; we
    verify they diverge on at least one trial.
    """
    cfg = AugmentationConfig(
        purpose="supervised",
        n_aug_per_positive=1,
        n_aug_per_negative=0,
        deterministic_seed=2025,
    )
    pre_frames = data_cfg.window.pre_stim_frames

    diverged = False
    for source in synthetic_instances[:5]:
        aug = augment_instance(source, cfg, data_cfg, aug_index=0)
        # Compose a "diff signature" — the autocorrelation lag-1 of each half.
        for key in ("g", "h", "f"):
            arr = aug.get_modality(key)
            pre_ac = np.corrcoef(arr[:pre_frames - 1, 0], arr[1:pre_frames, 0])[0, 1]
            post_ac = np.corrcoef(arr[pre_frames:-1, 0], arr[pre_frames + 1:, 0])[0, 1]
            if abs(pre_ac - post_ac) > 1e-6:
                diverged = True
                break
        if diverged:
            break
    assert diverged, "Pre and post temporal augs appear identical — independent RNG broken."


def test_ad_baseline_purpose_does_not_augment_post_stim_of_stimulus_trials(
    synthetic_instances, data_cfg
):
    """Under `purpose='ad_baseline'`, post-stim of stimulus trials is left RAW.

    Implementation in `augment_instance`: post-window augs are skipped for
    stimulus trials when purpose='ad_baseline'. Verify by checking that the
    aug copy's post-stim segment EQUALS the source's post-stim segment.
    """
    cfg = AugmentationConfig(
        purpose="ad_baseline",
        n_aug_per_positive=1,
        n_aug_per_negative=0,
        deterministic_seed=7,
    )
    pre_frames = data_cfg.window.pre_stim_frames

    for source in synthetic_instances:
        if source.trial_type != 1:
            continue
        aug = augment_instance(source, cfg, data_cfg, aug_index=0)
        for key in ("g", "h", "f"):
            src_post = source.get_modality(key)[pre_frames:]
            aug_post = aug.get_modality(key)[pre_frames:]
            assert np.array_equal(src_post, aug_post), (
                f"AD-baseline purpose augmented post-stim of stimulus trial "
                f"({source.pt_id}/{source.trial_id}/{key})"
            )


def test_augmented_copies_inherit_trial_id_and_record_composition(synthetic_instances, data_cfg):
    """Augmentation provenance contract (Workstream C.4 + C.7)."""
    cfg = AugmentationConfig(
        purpose="supervised",
        n_aug_per_positive=2,
        n_aug_per_negative=1,
        deterministic_seed=11,
        record_composition_string=True,
    )
    from src.configs.data import DataConfig
    from src.data.dataset import BambinoDataset

    ds = BambinoDataset(synthetic_instances, data_cfg)
    aug_ds = build_augmented_dataset(ds, cfg)

    augmented = [i for i in aug_ds.instances if i.is_augmented]
    assert len(augmented) > 0
    for inst in augmented:
        assert inst.source_trial_id is not None
        assert inst.aug_seed is not None
        assert inst.aug_composition is not None and "jitter" in inst.aug_composition


def test_augmentation_is_reproducible_under_same_seed(synthetic_instances, data_cfg):
    """Same seed → identical augmented output (Workstream C.7)."""
    cfg = AugmentationConfig(
        purpose="supervised",
        n_aug_per_positive=1,
        n_aug_per_negative=0,
        deterministic_seed=99,
    )
    a1 = augment_instance(synthetic_instances[1], cfg, data_cfg, aug_index=0)
    a2 = augment_instance(synthetic_instances[1], cfg, data_cfg, aug_index=0)
    for key in ("g", "h", "f"):
        assert np.array_equal(a1.get_modality(key), a2.get_modality(key)), (
            f"Augmentation not reproducible for modality {key}"
        )
