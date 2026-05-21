"""Configuration dataclasses for the BAMBINO pipeline.

Composition is explicit: each `RunConfig` carries the concrete dataclass
instances for data, augmentation, trainer, and the chosen model. There is no
global mutable state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .augmentation import AugmentationConfig
from .base import BaseConfig, ModalityConfig, PathConfig
from .data import DataConfig, QualityControlConfig, SplitConfig, WindowConfig
from .models import (
    AnomalyDetectorConfig,
    FiLMCNNConfig,
    InceptionTimeConfig,
    LogRegConfig,
    MiniRocketConfig,
    MomentConfig,
    ResNetGASFConfig,
)
from .trainer import TrainerConfig

ModelConfig = Union[
    LogRegConfig,
    MiniRocketConfig,
    InceptionTimeConfig,
    FiLMCNNConfig,
    ResNetGASFConfig,
    MomentConfig,
    AnomalyDetectorConfig,
]


@dataclass(frozen=True)
class RunConfig:
    """A single experiment specification."""
    run_id: str
    base: BaseConfig = field(default_factory=BaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=InceptionTimeConfig)


__all__ = [
    "AnomalyDetectorConfig",
    "AugmentationConfig",
    "BaseConfig",
    "DataConfig",
    "FiLMCNNConfig",
    "InceptionTimeConfig",
    "LogRegConfig",
    "ModalityConfig",
    "ModelConfig",
    "MiniRocketConfig",
    "MomentConfig",
    "PathConfig",
    "QualityControlConfig",
    "ResNetGASFConfig",
    "RunConfig",
    "SplitConfig",
    "TrainerConfig",
    "WindowConfig",
]
