"""Augmentation configuration.

Encodes the Workstream C contract from PROJECT_STATE.md §3.3:

    * Spatial / magnitude augmentations share the same RNG state across the
      paired (pre, post) windows to preserve spatial continuity.
    * Temporal augmentations (warp, shift) are applied INDEPENDENTLY to pre
      and post tensors. They MUST NOT cross the t=0 boundary, otherwise the
      anomaly-detection baseline premise is violated.
    * `purpose` distinguishes supervised augmentation from AD-baseline-only
      augmentation. AD baseline corpora augment only the pre-stimulus and
      silent-control windows; post-stimulus windows are scored, never aug'd.
    * Augmented copies inherit `trial_id` and (for supervised models) the
      habituation-decay weight of their source trial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

Purpose = Literal["supervised", "ad_baseline"]


@dataclass(frozen=True)
class JitterConfig:
    enabled: bool = True
    sigma: float = 0.05  # Gaussian std as fraction of channel std


@dataclass(frozen=True)
class ScalingConfig:
    enabled: bool = True
    range: Tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class PadShiftConfig:
    """Temporal translation with zero-padding. Independent per side of t=0."""
    enabled: bool = True
    max_shift_frames: int = 25  # ±1 s @ 25 fps


@dataclass(frozen=True)
class TimeWarpConfig:
    """Nonlinear monotonic time stretching, bounded so any frame moves by
    at most `max_drift_ms`."""
    enabled: bool = True
    max_warp: float = 0.2
    # Soft replacement for the legacy "preserve first 0.6 s" hard constraint.
    max_drift_ms: float = 100.0


@dataclass(frozen=True)
class TimeMaskingConfig:
    enabled: bool = True
    mask_prob: float = 0.5
    min_length: int = 5
    max_length: int = 20


@dataclass(frozen=True)
class ChannelDropoutConfig:
    """Zero a random subset of the 38 channels (Workstream C.6)."""
    enabled: bool = True
    drop_prob: float = 0.1


@dataclass(frozen=True)
class MagnitudeWarpConfig:
    """Smooth multiplicative envelope (Workstream C.6)."""
    enabled: bool = True
    sigma: float = 0.1
    n_knots: int = 4


@dataclass(frozen=True)
class AugmentationConfig:
    """Top-level augmentation contract."""
    purpose: Purpose = "supervised"
    n_aug_per_positive: int = 1
    n_aug_per_negative: int = 3
    subject_stratified_budget: bool = True

    # Reproducibility (Workstream C.7)
    deterministic_seed: int = 2025
    record_composition_string: bool = True

    jitter: JitterConfig = field(default_factory=JitterConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    pad_shift: PadShiftConfig = field(default_factory=PadShiftConfig)
    time_warp: TimeWarpConfig = field(default_factory=TimeWarpConfig)
    time_mask: TimeMaskingConfig = field(default_factory=TimeMaskingConfig)
    channel_dropout: ChannelDropoutConfig = field(default_factory=ChannelDropoutConfig)
    magnitude_warp: MagnitudeWarpConfig = field(default_factory=MagnitudeWarpConfig)
