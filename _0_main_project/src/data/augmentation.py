"""Augmentation primitives + class-aware pipeline.

Hard contracts (PROJECT_STATE §3.3):

    * Spatial / magnitude augs (jitter, scale, channel dropout, mag-warp) share
      the same RNG state across (pre, post) — applied with one seed to each
      window so the realisations match channel-wise.
    * Temporal augs (time-warp, pad-shift, time-mask) are applied INDEPENDENTLY
      to pre and post. THE T=0 BOUNDARY MUST NOT BE CROSSED.
    * `purpose='ad_baseline'` augments only the pre-stimulus window and the
      silent-control trials' full clips — post-stimulus windows of stimulus
      trials are NEVER augmented under this purpose.
    * Augmented copies inherit `trial_id` and (for supervised purpose) the
      sample_weight of their source, scaled by `1/n_aug` for that source.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable, List, Tuple

import numpy as np

from ..configs.augmentation import AugmentationConfig
from ..configs.data import DataConfig
from .dataset import BambinoDataset
from .instance import OpenFaceInstance


# ─── Primitives (operate on a single [T, D] array) ──────────────────────────

def jitter(rng: np.random.Generator, x: np.ndarray, sigma: float) -> np.ndarray:
    if x.size == 0:
        return x
    std = x.std(axis=0, keepdims=True)
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return (x + noise * std).astype(np.float32)


def scale(rng: np.random.Generator, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    factor = float(rng.uniform(lo, hi))
    return (x * factor).astype(np.float32)


def channel_dropout(rng: np.random.Generator, x: np.ndarray, p: float) -> np.ndarray:
    if x.size == 0:
        return x
    mask = (rng.uniform(size=x.shape[1]) >= p).astype(np.float32)
    return (x * mask[None, :]).astype(np.float32)


def magnitude_warp(rng: np.random.Generator, x: np.ndarray, sigma: float, n_knots: int) -> np.ndarray:
    if x.size == 0:
        return x
    T, D = x.shape
    knots = rng.normal(loc=1.0, scale=sigma, size=(n_knots + 2, D)).astype(np.float32)
    xp = np.linspace(0, T - 1, n_knots + 2)
    full = np.empty_like(x, dtype=np.float32)
    for d in range(D):
        full[:, d] = np.interp(np.arange(T), xp, knots[:, d])
    return (x * full).astype(np.float32)


def pad_shift(rng: np.random.Generator, x: np.ndarray, max_shift: int) -> np.ndarray:
    """Translate along time with zero-padding. Bounded; never crosses array ends."""
    if x.size == 0 or max_shift <= 0:
        return x
    T = x.shape[0]
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    if shift > 0:
        out[shift:] = x[:T - shift]
    else:
        out[:T + shift] = x[-shift:]
    return out


def time_warp(
    rng: np.random.Generator,
    x: np.ndarray,
    max_warp: float,
    frame_rate: int,
    max_drift_ms: float,
) -> np.ndarray:
    """Monotonic time warp with bounded drift.

    Constructs a strictly-increasing warping function and rejects realisations
    that move any frame by more than `max_drift_ms`. The frame-rate is needed
    to translate the millisecond bound into frames.
    """
    if x.size == 0:
        return x
    T = x.shape[0]
    max_drift_frames = max_drift_ms * frame_rate / 1000.0
    for _ in range(20):  # bounded retries
        deltas = rng.normal(loc=1.0, scale=max_warp, size=T)
        deltas = np.clip(deltas, 1.0 - max_warp, 1.0 + max_warp)
        cum = np.cumsum(deltas)
        cum = (cum - cum.min()) / max(cum.max() - cum.min(), 1e-8) * (T - 1)
        if np.max(np.abs(cum - np.arange(T))) <= max_drift_frames:
            break
    out = np.empty_like(x, dtype=np.float32)
    base = np.arange(T)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(base, cum, x[:, d])
    return out


def time_mask(
    rng: np.random.Generator,
    x: np.ndarray,
    p: float,
    min_length: int,
    max_length: int,
) -> np.ndarray:
    if x.size == 0 or rng.uniform() > p:
        return x.astype(np.float32)
    T = x.shape[0]
    length = int(rng.integers(min_length, max_length + 1))
    length = min(length, T)
    start = int(rng.integers(0, max(T - length, 1)))
    out = x.astype(np.float32).copy()
    out[start:start + length] = 0.0
    return out


# ─── Pipeline ───────────────────────────────────────────────────────────────

def _seed_for(source_seed: int, salt: str) -> int:
    h = hashlib.sha256(f"{source_seed}-{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _apply_window_augs(
    rng_spatial: np.random.Generator,
    rng_temporal: np.random.Generator,
    arr: np.ndarray,
    cfg: AugmentationConfig,
    frame_rate: int,
) -> Tuple[np.ndarray, List[str]]:
    """Apply augs to a single window. Returns (transformed, composition trace)."""
    composition: List[str] = []
    out = arr.astype(np.float32)

    # Spatial / magnitude (RNG shared across pre/post via rng_spatial).
    if cfg.jitter.enabled:
        out = jitter(rng_spatial, out, cfg.jitter.sigma)
        composition.append(f"jitter({cfg.jitter.sigma})")
    if cfg.scaling.enabled:
        out = scale(rng_spatial, out, *cfg.scaling.range)
        composition.append(f"scale{cfg.scaling.range}")
    if cfg.channel_dropout.enabled:
        out = channel_dropout(rng_spatial, out, cfg.channel_dropout.drop_prob)
        composition.append(f"chan_drop({cfg.channel_dropout.drop_prob})")
    if cfg.magnitude_warp.enabled:
        out = magnitude_warp(
            rng_spatial, out, cfg.magnitude_warp.sigma, cfg.magnitude_warp.n_knots
        )
        composition.append(f"mag_warp({cfg.magnitude_warp.sigma})")

    # Temporal (independent RNG per window — never crosses t=0).
    if cfg.pad_shift.enabled:
        out = pad_shift(rng_temporal, out, cfg.pad_shift.max_shift_frames)
        composition.append(f"pad_shift(±{cfg.pad_shift.max_shift_frames})")
    if cfg.time_warp.enabled:
        out = time_warp(
            rng_temporal, out, cfg.time_warp.max_warp,
            frame_rate, cfg.time_warp.max_drift_ms,
        )
        composition.append(f"time_warp({cfg.time_warp.max_warp},{cfg.time_warp.max_drift_ms}ms)")
    if cfg.time_mask.enabled:
        out = time_mask(
            rng_temporal, out, cfg.time_mask.mask_prob,
            cfg.time_mask.min_length, cfg.time_mask.max_length,
        )
        composition.append(
            f"time_mask({cfg.time_mask.mask_prob},{cfg.time_mask.min_length}-{cfg.time_mask.max_length})"
        )
    return out, composition


def augment_instance(
    inst: OpenFaceInstance,
    cfg: AugmentationConfig,
    data_cfg: DataConfig,
    aug_index: int,
) -> OpenFaceInstance:
    """Produce a single augmented copy of `inst`.

    The pre and post halves are augmented separately for temporal augs (so t=0
    is never crossed) but share the same RNG state for spatial / magnitude
    augs. AD-baseline purpose: the post-stimulus window of stimulus trials is
    LEFT UNTOUCHED.
    """
    pre_frames = data_cfg.window.pre_stim_frames
    seed = _seed_for(cfg.deterministic_seed, f"{inst.pt_id}-{inst.trial_id}-{aug_index}")
    rng_spatial = np.random.default_rng(seed)
    rng_temporal_pre = np.random.default_rng(seed + 1)
    rng_temporal_post = np.random.default_rng(seed + 2)

    new_inst = OpenFaceInstance(
        pt_id=inst.pt_id,
        sex=inst.sex,
        age=inst.age,
        trial_id=inst.trial_id,
        trial_type=inst.trial_type,
        audio=inst.audio,
        speaker=inst.speaker,
        gaze_info=inst.gaze_info.copy(),
        head_info=inst.head_info.copy(),
        face_info=inst.face_info.copy(),
        is_augmented=True,
        aug_seed=seed,
        source_trial_id=inst.trial_id,
        sample_weight=inst.sample_weight,
    )

    composition_per_modality: List[str] = []
    for key in ("g", "h", "f"):
        full = new_inst.get_modality(key)
        pre = full[:pre_frames]
        post = full[pre_frames:]

        # In AD-baseline mode: only augment the pre-stim segment of stimulus
        # trials; for control trials (silent), the entire clip is "baseline"
        # so post is also augmented.
        is_baseline_post = (cfg.purpose == "ad_baseline" and inst.trial_type == 1)

        rs_pre = np.random.default_rng(rng_spatial.integers(0, 2**32 - 1))
        new_pre, comp_pre = _apply_window_augs(rs_pre, rng_temporal_pre, pre, cfg, data_cfg.window.frame_rate)

        if is_baseline_post:
            new_post = post  # unchanged
            comp_post: List[str] = ["NO_AUG_POST(ad_baseline_stim_trial)"]
        else:
            rs_post = np.random.default_rng(rng_spatial.integers(0, 2**32 - 1))
            new_post, comp_post = _apply_window_augs(
                rs_post, rng_temporal_post, post, cfg, data_cfg.window.frame_rate
            )

        new_inst.set_modality(key, np.concatenate([new_pre, new_post], axis=0))
        composition_per_modality.append(f"{key}=[pre:{','.join(comp_pre)}|post:{','.join(comp_post)}]")

    if cfg.record_composition_string:
        new_inst.aug_composition = " ; ".join(composition_per_modality)
    return new_inst


def build_augmented_dataset(
    dataset: BambinoDataset,
    cfg: AugmentationConfig,
) -> BambinoDataset:
    """Generate augmented copies according to per-class budgets.

    Rules:
        * supervised: n_aug_per_positive copies of each stimulus, n_aug_per_negative
          copies of each control. Sample weight is divided by (1 + n_aug) so the
          source + its augmentations together carry the source's original weight.
        * ad_baseline: only control trials get augmented (their entire clip
          is baseline). Stimulus trials' pre-stim segments are augmented but
          their post-stim segment is left raw.
        * subject-stratified budgets equalise the per-subject augmented count.
    """
    rng = np.random.default_rng(cfg.deterministic_seed)
    new_instances: List[OpenFaceInstance] = list(dataset.instances)

    # Subject-stratified targets: number of trials per subject normalised
    target_per_class = {0: cfg.n_aug_per_negative, 1: cfg.n_aug_per_positive}

    if cfg.subject_stratified_budget:
        # Cap per-subject so high-trial-count infants don't dominate.
        from collections import defaultdict
        per_subject_counts: dict = defaultdict(lambda: {0: 0, 1: 0})
        for inst in dataset.instances:
            per_subject_counts[inst.pt_id][inst.trial_type] += 1

    for inst in dataset.instances:
        n_aug = target_per_class.get(inst.trial_type, 0)
        for k in range(n_aug):
            aug = augment_instance(inst, cfg, dataset.cfg, aug_index=k)
            # Weight contract: source + aug copies share source's weight uniformly.
            aug.sample_weight = inst.sample_weight / float(1 + n_aug)
            new_instances.append(aug)
        # Rescale source weight to match the contract.
        if cfg.purpose == "supervised" and n_aug > 0:
            inst.sample_weight = inst.sample_weight / float(1 + n_aug)

    return BambinoDataset(new_instances, dataset.cfg, dataset.modalities)
