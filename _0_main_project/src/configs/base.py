"""Project-wide static settings.

Paths, seeds, modality channel counts, and device selection. Everything that
does not change between experiments lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEGACY_REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout for the new pipeline.

    Layout under `_0_main_project/data/full/`:
        * `raw/`   — legacy `.pt` pickles (input to the migration bridge).
        * `clean/` — class-free `.pt` dicts, the runtime input format.

    `data_root` defaults to `clean/` because that is what the new pipeline
    consumes. `legacy_raw_root` is exposed for the migration script.
    """
    project_root: str = PROJECT_ROOT
    data_root: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "full", "clean")
    )
    legacy_raw_root: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "full", "raw")
    )
    legacy_repo_data_root: str = field(
        default_factory=lambda: os.path.join(LEGACY_REPO_ROOT, "data")
    )
    results_root: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "results")
    )
    tb_log_root: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "results", "tb_logs")
    )
    csv_log_root: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "results", "csv_logs")
    )


@dataclass(frozen=True)
class ModalityConfig:
    """Channel counts per OpenFace modality. Total = 38.

    g: gaze (8), h: head pose (13), f: facial action units (17).
    """
    dims: Dict[str, int] = field(
        default_factory=lambda: {"g": 8, "h": 13, "f": 17}
    )

    @property
    def total_channels(self) -> int:
        return sum(self.dims.values())

    @property
    def keys(self) -> tuple:
        return tuple(self.dims.keys())


@dataclass(frozen=True)
class BaseConfig:
    """Top-level static settings shared across runs."""
    seed: int = 2025
    num_classes: int = 2
    paths: PathConfig = field(default_factory=PathConfig)
    modalities: ModalityConfig = field(default_factory=ModalityConfig)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def multi_gpu(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
