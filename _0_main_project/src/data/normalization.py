"""Per-modality normalisation in three modes.

Implements PROJECT_STATE §3.2.1 step 3:

    * `global`     — z-norm with statistics fit on the training set.
    * `per_trial`  — z-norm with per-trial PRE-STIMULUS mean/std (causal).
    * `per_subject`— z-norm with per-subject baseline-window statistics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np

from ..configs.data import BaselineMode, DataConfig
from .dataset import BambinoDataset
from .instance import ATTR_MAP


_EPS = 1e-6


def compute_global_norm_params(dataset: BambinoDataset) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-modality channel-wise mean & std over all training samples & timesteps.

    Numerically stable: accumulates sum and sum-of-squares, computes
    Var = E[x^2] - (E[x])^2, clamps to `_EPS` to avoid div-by-zero.
    """
    sums: Dict[str, np.ndarray] = {}
    sq_sums: Dict[str, np.ndarray] = {}
    n_total = 0

    for inst in dataset.instances:
        # We use the post-stim segment for normalisation statistics — that is
        # the segment classifiers actually consume in the supervised paradigm.
        pre_frames = dataset.cfg.window.pre_stim_frames
        for key in dataset.modalities:
            arr = inst.get_modality(key)[pre_frames:]
            if arr.size == 0:
                continue
            s = arr.sum(axis=0)
            sq = (arr ** 2).sum(axis=0)
            if key not in sums:
                sums[key] = s
                sq_sums[key] = sq
            else:
                sums[key] += s
                sq_sums[key] += sq
        n_total += dataset.cfg.window.post_stim_frames

    params: Dict[str, Dict[str, np.ndarray]] = {}
    for key in sums:
        mean = sums[key] / float(n_total)
        var = sq_sums[key] / float(n_total) - mean ** 2
        std = np.sqrt(np.clip(var, _EPS, None))
        params[key] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return params


def apply_global_normalization(
    dataset: BambinoDataset,
    params: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """In-place z-normalisation of every modality in every instance."""
    for inst in dataset.instances:
        for key in dataset.modalities:
            if key not in params:
                continue
            arr = inst.get_modality(key)
            mean = params[key]["mean"]
            std = params[key]["std"]
            normed = (arr - mean[None, :]) / std[None, :]
            inst.set_modality(key, normed.astype(np.float32))


def apply_per_trial_baseline_normalization(dataset: BambinoDataset) -> None:
    """Subtract pre-stim mean and divide by pre-stim std, per trial, per channel.

    Causal: only uses frames strictly before t=0 of the SAME trial. No leakage
    across trials, no leakage across the t=0 boundary.
    """
    pre_frames = dataset.cfg.window.pre_stim_frames
    for inst in dataset.instances:
        for key in dataset.modalities:
            arr = inst.get_modality(key)
            baseline = arr[:pre_frames]
            if baseline.size == 0:
                continue
            mean = baseline.mean(axis=0)
            std = baseline.std(axis=0)
            std = np.clip(std, _EPS, None)
            inst.set_modality(key, ((arr - mean[None, :]) / std[None, :]).astype(np.float32))


def apply_per_subject_baseline_normalization(dataset: BambinoDataset) -> None:
    """Per-subject z-norm using the subject's baseline (pre-stim) corpus.

    Causal at the SUBJECT level (uses the subject's own pre-stim frames only,
    never another subject's data).
    """
    pre_frames = dataset.cfg.window.pre_stim_frames
    by_subject: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for inst in dataset.instances:
        for key in dataset.modalities:
            arr = inst.get_modality(key)[:pre_frames]
            if arr.size:
                by_subject[inst.pt_id][key].append(arr)

    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for pid, mods in by_subject.items():
        stats[pid] = {}
        for key, chunks in mods.items():
            stacked = np.concatenate(chunks, axis=0)
            mean = stacked.mean(axis=0)
            std = np.clip(stacked.std(axis=0), _EPS, None)
            stats[pid][key] = {"mean": mean, "std": std}

    for inst in dataset.instances:
        for key in dataset.modalities:
            s = stats.get(inst.pt_id, {}).get(key)
            if s is None:
                continue
            arr = inst.get_modality(key)
            inst.set_modality(key, ((arr - s["mean"][None, :]) / s["std"][None, :]).astype(np.float32))


def normalize_datasets(
    train: BambinoDataset,
    val: BambinoDataset,
    test: BambinoDataset,
    mode: BaselineMode,
) -> None:
    """Apply normalisation in-place to all three splits according to `mode`.

    For `global`, statistics are fit on `train` only and applied to all three.
    For per-trial / per-subject, each split normalises using its own data
    (still causal because per-trial uses only pre-stim frames of that trial,
    and per-subject uses only the subject's pre-stim corpus).
    """
    if mode == "global":
        params = compute_global_norm_params(train)
        for ds in (train, val, test):
            apply_global_normalization(ds, params)
    elif mode == "per_trial":
        for ds in (train, val, test):
            apply_per_trial_baseline_normalization(ds)
    elif mode == "per_subject":
        for ds in (train, val, test):
            apply_per_subject_baseline_normalization(ds)
    else:  # pragma: no cover - exhausted by Literal
        raise ValueError(f"Unknown baseline_norm_mode: {mode}")
