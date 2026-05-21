"""Project utilities (plotting, metrics, samplers, seeding)."""
from .metrics import bootstrap_ci, core_metrics, habituation_bucketed_metrics
from .plotting import (
    ACCENT,
    CLASS_COLORS,
    PALETTE,
    PRIMARY,
    REFERENCE,
    SECONDARY,
    TERTIARY_A,
    TERTIARY_B,
    apply_default_style,
    get_class_colors,
    plot_anomaly_score_distribution,
    plot_confusion_matrix,
    plot_habituation_buckets,
    plot_roc_curve,
    plot_training_history,
    plot_umap_2d,
)
from .samplers import build_sampler, get_habituation_aware_sampler
from .seeding import seed_everything

__all__ = [
    "ACCENT",
    "CLASS_COLORS",
    "PALETTE",
    "PRIMARY",
    "REFERENCE",
    "SECONDARY",
    "TERTIARY_A",
    "TERTIARY_B",
    "apply_default_style",
    "bootstrap_ci",
    "build_sampler",
    "core_metrics",
    "get_class_colors",
    "get_habituation_aware_sampler",
    "habituation_bucketed_metrics",
    "plot_anomaly_score_distribution",
    "plot_confusion_matrix",
    "plot_habituation_buckets",
    "plot_roc_curve",
    "plot_training_history",
    "plot_umap_2d",
    "seed_everything",
]
