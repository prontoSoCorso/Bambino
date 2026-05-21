"""Per-model dataclass configurations.

Mirrors the active-model inventory in PROJECT_STATE.md §2.1. Each model has a
self-contained config; the orchestrator in `main.py` picks one and hands it to
the matching LightningModule (or sklearn wrapper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple  # noqa: F401

ModelName = Literal[
    "logreg",
    "minirocket",
    "inception_time",
    "film_cnn",
    "resnet_gasf",
    "moment_mlp",
    "moment_histgb",
    "moment_logreg",
    "moment_tabpfn",
    "moment_pca_tabpfn",
    "anomaly_detector",
]


@dataclass(frozen=True)
class LogRegConfig:
    """L1-penalised logistic regression on hand-crafted descriptors."""
    name: ModelName = "logreg"
    penalty: Literal["l1", "l2", "elasticnet"] = "l1"
    C: float = 1.0
    solver: str = "saga"
    max_iter: int = 5000
    class_weight: Optional[str] = "balanced"
    feature_set: Literal["full", "stats_only"] = "full"


@dataclass(frozen=True)
class MiniRocketConfig:
    """MiniRocket transform + gradient-boosted classifier."""
    name: ModelName = "minirocket"
    num_kernels: int = 10000
    classifier: Literal["histgb", "xgboost"] = "histgb"
    max_depth: int = 6
    n_estimators: int = 300
    use_metadata: bool = True


@dataclass(frozen=True)
class InceptionTimeConfig:
    """Deep ensemble of inception modules (Lightning)."""
    name: ModelName = "inception_time"
    nb_filters: int = 16
    depth: int = 3
    kernel_size: int = 39
    bottleneck_size: int = 8
    use_residual: bool = True
    dropout: float = 0.6

    # Trial-order encoding mode for sequence inputs (PROJECT_STATE §3.2.2).
    trial_order_mode: Literal["none", "broadcast_channel", "metadata"] = "broadcast_channel"

    # Optional pre-stimulus context (PROJECT_STATE §3.2.1 step 4).
    use_pre_stim_context: bool = False

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    batch_size: int = 16
    patience: int = 10


@dataclass(frozen=True)
class FiLMCNNConfig:
    """1D CNN conditioned on (age, sex) via FiLM blocks."""
    name: ModelName = "film_cnn"
    channels: Tuple[int, int] = (8, 16)
    kernel_size: int = 7
    dropout: float = 0.7
    head_dim: int = 16
    target_len: int = 250

    lr: float = 1e-3
    weight_decay: float = 0.05
    max_epochs: int = 40
    batch_size: int = 16
    patience: int = 10


@dataclass(frozen=True)
class ResNetGASFConfig:
    """ImageNet-pretrained ResNet-18 on GASF-encoded modality projections."""
    name: ModelName = "resnet_gasf"
    image_size: int = 224
    pca_per_modality: bool = True
    backbone_frozen: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 50
    batch_size: int = 32
    patience: int = 10


@dataclass(frozen=True)
class MomentConfig:
    """MOMENT foundation model + classifier head."""
    name: ModelName = "moment_mlp"
    backbone: str = "AutonLab/MOMENT-1-large"
    backbone_frozen: bool = True
    target_len: int = 512
    head: Literal["mlp", "histgb", "logreg", "tabpfn", "pca_tabpfn"] = "mlp"
    pca_components: int = 100
    contrast_pre_post: bool = False  # post_emb - pre_emb (PROJECT_STATE §3.2.1)

    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    batch_size: int = 8
    patience: int = 6


@dataclass(frozen=True)
class AnomalyDetectorConfig:
    """Subject-conditioned autoencoder over the per-subject baseline manifold.

    Implements the causal-baseline contract from PROJECT_STATE §3.1: at trial
    t for infant i, the model is conditioned on data from trials strictly < t.
    """
    name: ModelName = "anomaly_detector"
    encoder: Literal["conv1d_ae", "lstm_ae", "moment_ae"] = "conv1d_ae"

    latent_dim: int = 64
    subject_embed_dim: int = 16
    use_film_conditioning: bool = True

    # Trial-bucket of post-stim windows used at evaluation
    eval_buckets: Tuple[str, ...] = ("early", "mid", "late")

    # Causal baseline construction (Workstream A.1, A.2)
    baseline_window_seconds: float = 2.0
    include_silent_controls_in_baseline: bool = True
    causal_strict: bool = True  # raises if any leakage attempt is detected

    lr: float = 1e-3
    weight_decay: float = 1e-5
    mask_ratio: float = 0.3
    max_epochs: int = 100
    batch_size: int = 16
    patience: int = 10
