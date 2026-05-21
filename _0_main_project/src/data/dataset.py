"""BAMBINO Torch dataset with pre/post-stimulus split exposure.

Replaces `DataUtils/BoaOpenFaceDataset.py`. Key differences from the legacy
implementation:

    1. `__getitem__` returns BOTH the post-stimulus tensor and the pre-stimulus
       tensor (Workstream B.1). Consumers select what they need.
    2. QC is anchored on the PRE-STIMULUS 50 frames (Workstream B.1, step 2).
    3. `trial_index_in_session` is exposed in the metadata vector (Workstream
       B.2.1) so models can encode habituation as a covariate.
    4. Subject-grouped split assignment is provided externally; the dataset is
       agnostic to which split it represents.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..configs.data import DataConfig
from .instance import OpenFaceInstance, ATTR_MAP


class BambinoDataset(Dataset):
    """Subject-trial dataset with explicit pre/post-stimulus exposure."""

    TRIAL_TYPES = ("control", "stimulus")

    def __init__(
        self,
        instances: List[OpenFaceInstance],
        cfg: DataConfig,
        modalities: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.modalities: Tuple[str, ...] = tuple(modalities) if modalities else cfg.modalities
        self.instances = instances
        self._build_session_indices()

    def _build_session_indices(self) -> None:
        """Per-subject, normalised trial-order index in [0, 1].

        Each (pt_id, trial_id) gets a position within that subject's sorted
        trial list, divided by max_trial_idx to land in [0, 1].
        """
        self._trial_order: Dict[Tuple[str, float], float] = {}
        by_subject: Dict[str, List[Tuple[float, OpenFaceInstance]]] = {}
        for inst in self.instances:
            by_subject.setdefault(inst.pt_id, []).append((inst.trial_id, inst))
        for pid, trials in by_subject.items():
            trials.sort(key=lambda x: x[0])
            n = max(len(trials) - 1, 1)
            for idx, (tid, _) in enumerate(trials):
                self._trial_order[(pid, tid)] = idx / n

    def trial_position(self, inst: OpenFaceInstance) -> float:
        return self._trial_order.get((inst.pt_id, inst.trial_id), 0.0)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int):
        inst = self.instances[idx]
        pre_frames = self.cfg.window.pre_stim_frames

        x_pre: Dict[str, torch.Tensor] = {}
        x_post: Dict[str, torch.Tensor] = {}
        for key in self.modalities:
            full = inst.get_modality(key)
            t_full = torch.as_tensor(full, dtype=torch.float32)
            x_pre[key] = t_full[:pre_frames]
            x_post[key] = t_full[pre_frames:]

        y = torch.tensor([inst.trial_type], dtype=torch.long)

        # Trial-order encoding (Workstream B.2.1)
        trial_pos = self.trial_position(inst)
        meta = {
            "pt_id": inst.pt_id,
            "trial_id": float(inst.trial_id),
            "trial_position": float(trial_pos),
            "age": float(inst.age),
            "sex": int(inst.sex),
            "speaker": int(inst.speaker) if inst.speaker is not None else -1,
            "audio": inst.audio,
            "is_augmented": bool(inst.is_augmented),
            "sample_weight": float(inst.sample_weight),
        }

        if self.cfg.expose_pre_stim:
            return {"x_post": x_post, "x_pre": x_pre, "y": y, "meta": meta}
        return {"x_post": x_post, "y": y, "meta": meta}

    # ─── Loading helpers ────────────────────────────────────────────────────
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        cfg: DataConfig,
        modalities: Optional[Sequence[str]] = None,
    ) -> "BambinoDataset":
        """Build the dataset from a wide CSV exported by the OpenFace pipeline.

        Applies QC anchored on the pre-stimulus window (PROJECT_STATE §3.2.1
        step 2): a trial is dropped if confidence ≤ `min_confidence` over the
        FIRST `qc_window_frames` frames of the clip. We rely on the legacy
        contract that t=0 of the stimulus sits at index `pre_stim_frames`, so
        the pre-stim segment IS the first 50 frames of the array.
        """
        df = pd.read_csv(csv_path)
        df["sex"] = df["sex"].map({"Boy": 1, "Girl": 0}).fillna(df["sex"])
        df["trial_type"] = df["trial_type"].map({
            cls.TRIAL_TYPES[0]: 0,
            cls.TRIAL_TYPES[1]: 1,
        }).fillna(df["trial_type"])

        instances: List[OpenFaceInstance] = []
        qc_window = cfg.qc.qc_window_frames
        min_conf = cfg.qc.min_confidence

        for pid in df["participant_id"].unique():
            for tid in df["trial_id"].unique():
                sub = df[(df["participant_id"] == pid) & (df["trial_id"] == tid)]
                if sub.empty:
                    continue
                if "low_confidence_for_trial" in sub.columns and bool(sub["low_confidence_for_trial"].iat[0]):
                    continue
                if "confidence" in sub.columns:
                    if (sub["confidence"].iloc[:qc_window] <= min_conf).all():
                        continue
                instances.append(OpenFaceInstance.from_trial_dataframe(sub))

        return cls(instances, cfg, modalities)

    @classmethod
    def from_csv_dir(
        cls,
        directory: str,
        cfg: DataConfig,
        modalities: Optional[Sequence[str]] = None,
    ) -> "BambinoDataset":
        """Load every CSV under `directory` and concatenate."""
        all_instances: List[OpenFaceInstance] = []
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".csv"):
                continue
            ds = cls.from_csv(os.path.join(directory, fname), cfg, modalities)
            all_instances.extend(ds.instances)
        return cls(all_instances, cfg, modalities)

    @classmethod
    def from_clean_pt(
        cls,
        path: str,
        cfg: DataConfig,
        modalities: Optional[Sequence[str]] = None,
    ) -> "BambinoDataset":
        """Load a clean `.pt` blob produced by `scripts/migrate_legacy_data.py`.

        The blob is `{'instances': [dict, ...], 'metadata': {...}}` — no custom
        Python classes required to unpickle it. Each per-instance dict is
        rehydrated through `OpenFaceInstance.from_dict`.
        """
        blob = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(blob, dict) or "instances" not in blob:
            raise RuntimeError(
                f"{path} is not a clean BAMBINO blob. Expected dict with key 'instances'; "
                f"got top-level type {type(blob).__name__}. Re-run the migration bridge: "
                f"`python scripts/migrate_legacy_data.py`."
            )
        instances = [OpenFaceInstance.from_dict(d) for d in blob["instances"]]
        return cls(instances, cfg, modalities)

    @classmethod
    def from_clean_dir(
        cls,
        directory: str,
        cfg: DataConfig,
        modalities: Optional[Sequence[str]] = None,
        prefer_combined: bool = True,
    ) -> "BambinoDataset":
        """Load every clean `.pt` in `directory` and concatenate their instances.

        If `prefer_combined=True` and `bambino_clean.pt` exists, that file is
        used directly (single source-of-truth for re-splitting). Otherwise
        the per-split files (`training_set.pt`, `validation_set.pt`,
        `test_set.pt`) are concatenated.
        """
        combined = os.path.join(directory, "bambino_clean.pt")
        if prefer_combined and os.path.isfile(combined):
            return cls.from_clean_pt(combined, cfg, modalities)

        all_instances: List[OpenFaceInstance] = []
        for fname in ("training_set.pt", "validation_set.pt", "test_set.pt"):
            p = os.path.join(directory, fname)
            if os.path.isfile(p):
                ds = cls.from_clean_pt(p, cfg, modalities)
                all_instances.extend(ds.instances)
        if not all_instances:
            raise RuntimeError(
                f"No clean *.pt files found under {directory}. "
                "Did you run `python scripts/migrate_legacy_data.py`?"
            )
        return cls(all_instances, cfg, modalities)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"instances": self.instances, "cfg": self.cfg, "modalities": self.modalities}, f)

    @classmethod
    def load(cls, path: str, cfg: Optional[DataConfig] = None) -> "BambinoDataset":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return cls(blob["instances"], cfg or blob["cfg"], blob.get("modalities"))

    # ─── Convenience accessors ──────────────────────────────────────────────
    @property
    def subject_ids(self) -> List[str]:
        return sorted({inst.pt_id for inst in self.instances})

    @property
    def labels(self) -> np.ndarray:
        return np.array([inst.trial_type for inst in self.instances], dtype=np.int64)

    def filter_by_subjects(self, subjects: Sequence[str]) -> "BambinoDataset":
        keep = [inst for inst in self.instances if inst.pt_id in set(subjects)]
        return BambinoDataset(keep, self.cfg, self.modalities)
