"""Base LightningModule with shared metric logging.

Subclasses implement `forward` and override `_compute_loss`. The base handles:

    * `training_step`   — forward, loss, optional sample-weight loss reduction
    * `validation_step` — forward, log val metrics + accumulate for epoch-end
    * `test_step`       — forward, accumulate for epoch-end, no logging
    * `_log_metrics_at_epoch_end` — balanced accuracy / AUROC / F1

The expectation is that subclasses produce LOGITS of shape (B, num_classes).
For binary tasks, num_classes=2 and we softmax → use [:, 1] as positive score.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.metrics import core_metrics


class BaseClassifier(pl.LightningModule):
    """Shared training scaffold for binary classifiers.

    Subclasses MAY override `monitor_metric` / `monitor_mode` to declare which
    val metric the EarlyStopping + ModelCheckpoint callbacks should track.
    Defaults are `val/balanced_accuracy` (mode=max) which is correct for every
    supervised binary classifier. The anomaly detector overrides these.
    """

    monitor_metric: str = "val/balanced_accuracy"
    monitor_mode: str = "max"

    def __init__(self, num_classes: int = 2, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self._val_buffer: List[Dict[str, Any]] = []
        self._test_buffer: List[Dict[str, Any]] = []

    # ── To be implemented by subclasses ─────────────────────────────────────
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:  # noqa: D401
        """Return logits of shape (B, num_classes)."""
        raise NotImplementedError

    def _extract_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        return batch["y"].squeeze(-1).long()

    def _extract_sample_weights(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        meta = batch.get("meta", {})
        if "sample_weight" in meta:
            return torch.tensor(meta["sample_weight"], dtype=torch.float32, device=self.device)
        return None

    # ── Lightning hooks ─────────────────────────────────────────────────────
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Forward, weighted CE loss, log train/loss.

        We use sample weights from the batch metadata when present (these
        compose class-balance × habituation-decay × augmentation-share weights
        per PROJECT_STATE §3.2.2). Reduction is manual: per-sample CE multiplied
        by per-sample weight, then mean.
        """
        logits = self.forward(batch)
        target = self._extract_target(batch)
        per_sample = F.cross_entropy(logits, target, reduction="none")

        weights = self._extract_sample_weights(batch)
        if weights is not None and weights.shape == per_sample.shape:
            loss = (per_sample * weights).mean()
        else:
            loss = per_sample.mean()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Forward, no grad. Buffer logits + targets for epoch-end aggregation."""
        logits = self.forward(batch)
        target = self._extract_target(batch)
        loss = F.cross_entropy(logits, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_buffer.append({
            "logits": logits.detach().cpu(),
            "target": target.detach().cpu(),
            "trial_position": torch.tensor(batch["meta"]["trial_position"]),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_buffer:
            return
        logits = torch.cat([b["logits"] for b in self._val_buffer], dim=0).numpy()
        target = torch.cat([b["target"] for b in self._val_buffer], dim=0).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        m = core_metrics(target, preds, probs)
        for k, v in m.items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "balanced_accuracy"))
        self._val_buffer.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.forward(batch)
        target = self._extract_target(batch)
        self._test_buffer.append({
            "logits": logits.detach().cpu(),
            "target": target.detach().cpu(),
            "trial_position": torch.tensor(batch["meta"]["trial_position"]),
        })

    def on_test_epoch_end(self) -> None:
        if not self._test_buffer:
            return
        logits = torch.cat([b["logits"] for b in self._test_buffer], dim=0).numpy()
        target = torch.cat([b["target"] for b in self._test_buffer], dim=0).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        m = core_metrics(target, preds, probs)
        for k, v in m.items():
            self.log(f"test/{k}", float(v))
        self._test_buffer.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/balanced_accuracy"},
        }


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    """Return positive-class probability for shape-(N, 2) logits."""
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return p[:, 1]
