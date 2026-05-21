"""Data layer.

`BambinoDataModule` is intentionally imported lazily — it depends on
PyTorch Lightning, which is an optional install for users running only the
sklearn / numpy paths (e.g. the test suite).
"""
from importlib import import_module
from typing import TYPE_CHECKING

from .augmentation import (
    augment_instance,
    build_augmented_dataset,
    channel_dropout,
    jitter,
    magnitude_warp,
    pad_shift,
    scale,
    time_mask,
    time_warp,
)
from .dataset import BambinoDataset
from .features import build_feature_matrix, per_channel_descriptors, trial_feature_vector
from .instance import OpenFaceInstance, categorize_age
from .normalization import (
    apply_global_normalization,
    apply_per_subject_baseline_normalization,
    apply_per_trial_baseline_normalization,
    compute_global_norm_params,
    normalize_datasets,
)
from .splits import assert_no_subject_leakage, split_dataset, subject_grouped_split

if TYPE_CHECKING:  # pragma: no cover
    from .datamodule import BambinoDataModule


def __getattr__(name: str):
    if name == "BambinoDataModule":
        return import_module(".datamodule", __name__).BambinoDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BambinoDataModule",
    "BambinoDataset",
    "OpenFaceInstance",
    "apply_global_normalization",
    "apply_per_subject_baseline_normalization",
    "apply_per_trial_baseline_normalization",
    "assert_no_subject_leakage",
    "augment_instance",
    "build_augmented_dataset",
    "build_feature_matrix",
    "categorize_age",
    "channel_dropout",
    "compute_global_norm_params",
    "jitter",
    "magnitude_warp",
    "normalize_datasets",
    "pad_shift",
    "per_channel_descriptors",
    "scale",
    "split_dataset",
    "subject_grouped_split",
    "time_mask",
    "time_warp",
    "trial_feature_vector",
]
