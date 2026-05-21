"""PyTorch Lightning Trainer configuration.

We log to local TensorBoard + CSV only — no W&B, no MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class TrainerConfig:
    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int = 1
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    max_epochs: int = 50
    check_val_every_n_epoch: int = 1
    gradient_clip_val: Optional[float] = 1.0
    deterministic: bool = True

    # Early stopping. `monitor=None` lets the trainer ask the model for
    # its own preferred metric via `Model.monitor_metric()` — this is how
    # the anomaly detector (which logs `val/novelty_auc`) and supervised
    # classifiers (which log `val/balanced_accuracy`) coexist without a
    # hard-coded global metric.
    early_stopping: bool = True
    monitor: Optional[str] = None
    monitor_mode: Optional[Literal["min", "max"]] = None
    patience: int = 10

    # Logging
    log_every_n_steps: int = 10
    save_top_k: int = 1

    # Habituation-aware sampling toggle (Workstream B.2.2)
    use_habituation_decay_weights: bool = False
    habituation_decay_lambda: float = 0.05
