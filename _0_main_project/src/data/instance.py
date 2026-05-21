"""Trial instance: a single (subject, trial) record with pre/post split.

This replaces `DataUtils/OpenFaceInstance.py` from the legacy code. The crucial
new contract is:

    * The 12 s clip is split at index `pre_stim_frames` into `(pre, post)`.
    * Both windows are exposed; downstream code chooses which to consume.
    * NaNs are interpolated -> back/forward filled -> zero-filled (preserved
      from legacy behaviour for compatibility with existing CSV exports).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


GAZE_COLS = list(range(17, 25))  # 8 channels
HEAD_COLS = list(range(25, 38))  # 13 channels
FACE_COLS = list(range(38, 55))  # 17 channels

DIM_DICT = {"g": 8, "h": 13, "f": 17}
ATTR_MAP = {"g": "gaze_info", "h": "head_info", "f": "face_info"}


@dataclass
class OpenFaceInstance:
    """A single trial. Modality arrays are float32, shape [T, dim]."""
    pt_id: str
    sex: int
    age: float
    trial_id: float
    trial_type: int  # 0=control, 1=stimulus
    audio: str
    speaker: Optional[int]

    gaze_info: np.ndarray
    head_info: np.ndarray
    face_info: np.ndarray

    # Augmentation provenance (Workstream C.7)
    is_augmented: bool = False
    aug_seed: Optional[int] = None
    aug_composition: Optional[str] = None
    source_trial_id: Optional[float] = None
    sample_weight: float = 1.0

    def to_dict(self) -> dict:
        """Serialise to a pure-Python dict (no custom-class dependency).

        Used by the migration bridge to write `data/full/clean/*.pt` files
        that load on any future architecture without needing this dataclass
        in scope.
        """
        return {
            "pt_id": str(self.pt_id),
            "sex": int(self.sex),
            "age": float(self.age),
            "trial_id": float(self.trial_id),
            "trial_type": int(self.trial_type),
            "audio": str(self.audio),
            "speaker": int(self.speaker) if self.speaker is not None else None,
            "gaze_info": np.asarray(self.gaze_info, dtype=np.float32),
            "head_info": np.asarray(self.head_info, dtype=np.float32),
            "face_info": np.asarray(self.face_info, dtype=np.float32),
            # Augmentation provenance — only round-tripped if present.
            "is_augmented": bool(self.is_augmented),
            "aug_seed": self.aug_seed,
            "aug_composition": self.aug_composition,
            "source_trial_id": self.source_trial_id,
            "sample_weight": float(self.sample_weight),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpenFaceInstance":
        """Reconstruct from a pure-dict serialisation produced by `to_dict`
        OR by the legacy migration bridge (`scripts/migrate_legacy_data.py`).

        Tolerant to missing optional fields (provenance, augmentation flags).
        """
        speaker = d.get("speaker")
        if speaker is not None:
            try:
                speaker = int(speaker)
            except (TypeError, ValueError):
                speaker = None
        return cls(
            pt_id=str(d["pt_id"]),
            sex=int(d.get("sex", 0)),
            age=float(d.get("age", 0.0)),
            trial_id=float(d["trial_id"]),
            trial_type=int(d["trial_type"]),
            audio=str(d.get("audio", "")),
            speaker=speaker,
            gaze_info=np.asarray(d["gaze_info"], dtype=np.float32),
            head_info=np.asarray(d["head_info"], dtype=np.float32),
            face_info=np.asarray(d["face_info"], dtype=np.float32),
            is_augmented=bool(d.get("is_augmented", False)),
            aug_seed=d.get("aug_seed"),
            aug_composition=d.get("aug_composition"),
            source_trial_id=d.get("source_trial_id"),
            sample_weight=float(d.get("sample_weight", 1.0)),
        )

    @classmethod
    def from_trial_dataframe(cls, trial_data: pd.DataFrame) -> "OpenFaceInstance":
        """Build from a per-trial slice of the original wide CSV."""
        arr = trial_data.to_numpy()
        raw_audio = arr[0, 6]
        audio = str(raw_audio)[:-4].replace("_", " ") if isinstance(raw_audio, str) else str(raw_audio)
        raw_spk = str(arr[0, 7]).lower() if arr[0, 7] is not None else "unknown"
        speaker = 0 if raw_spk == "left" else (1 if raw_spk == "right" else None)

        gaze = _interp_columns(arr, GAZE_COLS)
        head = _interp_columns(arr, HEAD_COLS)
        face = _interp_columns(arr, FACE_COLS)

        return cls(
            pt_id=str(arr[0, 0]),
            sex=int(arr[0, 1]),
            age=float(arr[0, 3]),
            trial_id=float(arr[0, 4]),
            trial_type=int(arr[0, 5]),
            audio=audio,
            speaker=speaker,
            gaze_info=gaze,
            head_info=head,
            face_info=face,
        )

    def get_modality(self, key: str) -> np.ndarray:
        return getattr(self, ATTR_MAP[key])

    def set_modality(self, key: str, arr: np.ndarray) -> None:
        setattr(self, ATTR_MAP[key], arr)

    def split_pre_post(self, pre_frames: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Return {'pre': {modality: arr}, 'post': {modality: arr}}.

        The split is sliced from the START of the array, which we contractually
        treat as the pre-stimulus segment per the OpenFace pipeline (the legacy
        code already loads the 12 s clip with t=0 at index `pre_frames`).
        """
        out = {"pre": {}, "post": {}}
        for key in ("g", "h", "f"):
            full = self.get_modality(key)
            out["pre"][key] = full[:pre_frames]
            out["post"][key] = full[pre_frames:]
        return out


def _interp_columns(arr: np.ndarray, cols) -> np.ndarray:
    """Linear interpolation -> back/forward fill -> zero-fill, returning float32."""
    cols_data = []
    for c in cols:
        s = pd.Series(arr[:, c].astype(np.float32))
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.bfill().ffill().fillna(0.0)
        cols_data.append(s.to_numpy(dtype=np.float32))
    out = np.stack(cols_data, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def categorize_age(age: float) -> Optional[int]:
    if 3.0 <= age < 5.5:
        return 0
    if 5.5 <= age < 7.5:
        return 1
    return None
