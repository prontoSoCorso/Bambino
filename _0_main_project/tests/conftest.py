"""Shared fixtures for the BAMBINO test suite."""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.configs import DataConfig
from src.data.dataset import BambinoDataset
from src.data.instance import OpenFaceInstance


def _make_instance(
    pt_id: str,
    trial_id: float,
    trial_type: int,
    pre_frames: int = 50,
    post_frames: int = 250,
    seed: int = 0,
) -> OpenFaceInstance:
    """Synthetic instance with deterministic per-channel waveforms.

    Pre-stim frames are zero-mean unit-variance noise. Post-stim frames are the
    same noise PLUS a step at frame `pre_frames` if `trial_type == 1`. This
    gives every test a predictable t=0 boundary signature.
    """
    rng = np.random.default_rng(seed)
    T = pre_frames + post_frames
    gaze = rng.normal(0, 1, size=(T, 8)).astype(np.float32)
    head = rng.normal(0, 1, size=(T, 13)).astype(np.float32)
    face = rng.normal(0, 1, size=(T, 17)).astype(np.float32)

    if trial_type == 1:
        # Step injection at t=0 to verify temporal augs do not cross it.
        gaze[pre_frames:] += 5.0
        head[pre_frames:] += 5.0
        face[pre_frames:] += 5.0

    return OpenFaceInstance(
        pt_id=pt_id,
        sex=0,
        age=4.0,
        trial_id=trial_id,
        trial_type=trial_type,
        audio="warble.wav",
        speaker=0,
        gaze_info=gaze,
        head_info=head,
        face_info=face,
    )


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig()


@pytest.fixture
def synthetic_instances() -> List[OpenFaceInstance]:
    """Three subjects × ~5 trials each. Mix of stimulus and control."""
    out: List[OpenFaceInstance] = []
    for s_idx, pt_id in enumerate(["S001", "S002", "S003"]):
        for t in range(5):
            trial_type = 1 if t > 0 else 0  # first trial is control
            out.append(_make_instance(pt_id, float(t), trial_type, seed=10 * s_idx + t))
    return out


@pytest.fixture
def synthetic_dataset(synthetic_instances, data_cfg) -> BambinoDataset:
    return BambinoDataset(synthetic_instances, data_cfg)
