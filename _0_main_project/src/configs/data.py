"""Data-loading and windowing configuration.

Encodes the temporal contract documented in PROJECT_STATE.md §1.2 / §3.2:
    * 25 fps capture
    * 12 s trial = 2 s pre-stimulus + 10 s post-stimulus
    * QC filter anchored on the pre-stimulus 50 frames
    * Causal baseline: only frames strictly before the current trial's onset
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

BaselineMode = Literal["global", "per_trial", "per_subject"]
ScalerType = Literal["zscore", "robust", "none"]


@dataclass(frozen=True)
class WindowConfig:
    """Temporal windowing of a single trial.

    The post-stimulus segment is the classifier input under the supervised
    paradigm; the pre-stimulus segment is the per-trial baseline for the
    anomaly-detection paradigm.
    """
    frame_rate: int = 25
    pre_stim_seconds: float = 2.0
    post_stim_seconds: float = 10.0

    @property
    def pre_stim_frames(self) -> int:
        return int(round(self.pre_stim_seconds * self.frame_rate))

    @property
    def post_stim_frames(self) -> int:
        return int(round(self.post_stim_seconds * self.frame_rate))

    @property
    def total_frames(self) -> int:
        return self.pre_stim_frames + self.post_stim_frames

    @property
    def stimulus_onset_index(self) -> int:
        """Frame index where t=0 (stimulus onset) sits in the 12 s window."""
        return self.pre_stim_frames


@dataclass(frozen=True)
class QualityControlConfig:
    """OpenFace confidence-based trial-rejection thresholds."""
    min_confidence: float = 0.5
    # Window of frames the confidence threshold is checked against. Anchored
    # on the pre-stimulus segment per PROJECT_STATE §3.2.1 step 2.
    qc_anchor: Literal["pre_stim", "first_frames"] = "pre_stim"
    qc_window_frames: int = 50  # 2 s @ 25 fps


@dataclass(frozen=True)
class SplitConfig:
    """Subject-grouped split fractions. Subjects are partitioned, never trials."""
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    stratify_by_class: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Composite data-pipeline configuration."""
    raw_csv_glob: str = "*.csv"
    modalities: Tuple[str, ...] = ("g", "h", "f")
    window: WindowConfig = field(default_factory=WindowConfig)
    qc: QualityControlConfig = field(default_factory=QualityControlConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    baseline_norm_mode: BaselineMode = "global"
    scaler: ScalerType = "zscore"

    # Whether to expose the pre-stimulus window in the dataset getter
    # (Workstream B.1). When False, only post-stimulus frames are returned.
    expose_pre_stim: bool = True

    # Trial-order encoding (Workstream B.2.1). `none` disables the feature.
    trial_order_feature: Literal["none", "scalar", "scalar_log", "broadcast_channel"] = "scalar"

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
