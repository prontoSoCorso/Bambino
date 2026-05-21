"""Causal baseline construction (pure-numpy, no Lightning dependency).

Module name matches the spec in PROJECT_STATE §3.1 ("manifold_utils.py").
Kept independent of `anomaly_detector.py` so that the test suite and the
sklearn paths can validate the causal contract without requiring
`pytorch_lightning` to be installed.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..data.instance import OpenFaceInstance


def build_causal_baseline(
    instances: Sequence[OpenFaceInstance],
    target_pt_id: str,
    target_trial_id: float,
    pre_frames: int,
    include_silent_controls: bool,
) -> List[np.ndarray]:
    """Return all baseline windows for `target_pt_id` recorded BEFORE
    `target_trial_id`.

    Sources (per PROJECT_STATE §3.1):
        (a) the pre-stimulus 2 s segment of every trial with trial_id < target.
        (b) the FULL clip of every silent-control trial with trial_id < target,
            iff `include_silent_controls=True`.

    Each item is a (T_window, total_channels=38) float32 array, channels
    concatenated in canonical order (g, h, f).

    Causal guarantees:
        * Same subject only.
        * Trial id strictly less than target.
        * Pre-stim 2 s slice from each contributing trial (or full clip for
          silent controls).
    """
    out: List[np.ndarray] = []
    for inst in instances:
        if inst.pt_id != target_pt_id:
            continue
        if inst.trial_id >= target_trial_id:
            continue
        full = np.concatenate([inst.gaze_info, inst.head_info, inst.face_info], axis=1)
        out.append(full[:pre_frames])
        if include_silent_controls and inst.trial_type == 0:
            out.append(full)
    return out
