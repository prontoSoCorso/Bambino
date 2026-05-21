"""Subject-conditioned causal anomaly detector.

Implements the architectural blueprint in PROJECT_STATE §3.1.

Causal contract — re-stated explicitly:

    To score a post-stimulus window of trial t for infant i, we may only
    ingest data with `(pt_id == i AND trial_id < t)`. NEVER `>= t`. NEVER
    cross subjects. This guarantee is enforced by `build_causal_baseline()`
    and asserted by `_assert_causal()`.

The model is a 1D convolutional autoencoder (default) with FiLM conditioning
on a per-subject embedding. A single backbone amortises across infants
(no per-infant fine-tune), so the conditioning vector is the only
subject-specific knob at inference.

The novelty score is the per-trial mean squared reconstruction error on the
post-stimulus window. Higher score = more deviation from baseline = more
likely a "stimulus response" under the BOA logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import AnomalyDetectorConfig, ModalityConfig
from ..data.dataset import BambinoDataset
from ..data.instance import OpenFaceInstance
from .components import FiLM
from .manifold_utils import build_causal_baseline  # re-exported for back-compat


# ─── Architecture ───────────────────────────────────────────────────────────
class _ConvAE(nn.Module):
    """Symmetric 1D conv autoencoder with FiLM conditioning."""

    def __init__(self, in_channels: int, latent_dim: int, cond_dim: int):
        super().__init__()
        self.enc1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.enc2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.enc3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)
        self.film_enc = FiLM(cond_dim, latent_dim)

        self.dec1 = nn.Conv1d(latent_dim, 64, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.dec3 = nn.Conv1d(32, in_channels, kernel_size=7, padding=3)
        self.film_dec = FiLM(cond_dim, 64)

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = self.enc3(h)
        h = self.film_enc(h, cond)
        return h

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = self.film_dec(h, cond)
        h = F.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, cond)
        return self.decode(z, cond)


class SubjectConditionedAnomalyDetector(pl.LightningModule):
    """Subject-conditioned baseline-manifold model.

    Training inputs are baseline windows ONLY (pre-stim + silent controls).
    The model learns to reconstruct them; high reconstruction error on a
    post-stim window at inference time = anomaly score.

    Per PROJECT_STATE §3.2.2 ("CRITICAL CONSTRAINT"), training samples here
    carry FLAT weight (1.0) regardless of their position in the session.
    Habituation-decay weights are NEVER applied to AD baseline training.

    The supervised models monitor `val/balanced_accuracy`; the AD model has
    no class label at training time, so it monitors its own `val/novelty_auc`
    (proxy AUROC of reconstruction error vs. nominal label, computed at
    epoch end). Trainer callbacks consult these attributes via
    `_resolve_monitor_metric()` in main.py.
    """

    monitor_metric: str = "val/novelty_auc"
    monitor_mode: str = "max"

    def __init__(
        self,
        model_cfg: AnomalyDetectorConfig,
        modality_cfg: ModalityConfig,
        num_subjects: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["modality_cfg"])
        self.cfg = model_cfg
        self.modality_cfg = modality_cfg

        self.subject_embed = nn.Embedding(num_subjects, model_cfg.subject_embed_dim)
        self.ae = _ConvAE(
            in_channels=modality_cfg.total_channels,
            latent_dim=model_cfg.latent_dim,
            cond_dim=model_cfg.subject_embed_dim,
        )

        # Subject-id → integer mapping (filled by DataModule before fit).
        self._subject_to_idx: Dict[str, int] = {}

        self._val_buffer: List[Dict[str, Any]] = []
        self._test_buffer: List[Dict[str, Any]] = []

    # ── Subject mapping ─────────────────────────────────────────────────────
    def register_subjects(self, subjects: Sequence[str]) -> None:
        self._subject_to_idx = {s: i for i, s in enumerate(sorted(subjects))}

    def _subject_indices(self, pt_ids: Sequence[str]) -> torch.Tensor:
        idxs = [self._subject_to_idx.get(s, 0) for s in pt_ids]
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    # ── Lightning hooks ─────────────────────────────────────────────────────
    def _stack_window(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        chunks = [x_dict[k].transpose(1, 2) for k in self.modality_cfg.keys]
        return torch.cat(chunks, dim=1)  # (B, C, T)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Reconstruction-only training on PRE-STIMULUS windows.

        We deliberately use `x_pre` here (not `x_post`). Silent-control trials
        are mixed in by the DataModule — for them, the dataset returns the
        full clip and we still take the pre-stim half to keep input shape
        homogeneous. Equal sample weight per PROJECT_STATE §3.2.2.
        """
        x = self._stack_window(batch["x_pre"])
        pt_ids = batch["meta"]["pt_id"]
        cond = self.subject_embed(self._subject_indices(pt_ids))

        recon = self.ae(x, cond)
        loss = F.mse_loss(recon, x)
        self.log("train/recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Score POST-STIMULUS windows; novelty = per-trial mean MSE.

        We compare post-window reconstruction error against the binary label
        (Stimulus=1, Control=0) only as a proxy AUROC. Real evaluation is
        bucketed by trial position (see `on_validation_epoch_end`).
        """
        x_post = self._stack_window(batch["x_post"])
        pt_ids = batch["meta"]["pt_id"]
        cond = self.subject_embed(self._subject_indices(pt_ids))
        recon = self.ae(x_post, cond)
        per_trial_mse = ((recon - x_post) ** 2).mean(dim=(1, 2))
        target = batch["y"].squeeze(-1).long()
        self._val_buffer.append({
            "score": per_trial_mse.detach().cpu(),
            "target": target.detach().cpu(),
            "trial_position": torch.tensor(batch["meta"]["trial_position"]),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_buffer:
            return
        scores = torch.cat([b["score"] for b in self._val_buffer]).numpy()
        targets = torch.cat([b["target"] for b in self._val_buffer]).numpy()
        positions = torch.cat([b["trial_position"] for b in self._val_buffer]).numpy()
        # Proxy AUROC over all val trials.
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(targets, scores))
        except ValueError:
            auc = float("nan")
        self.log("val/novelty_auc", auc, prog_bar=True)

        # Early-bucket AUROC = the metric the journal needs to clear 0.5
        early = positions < 1 / 3
        if early.sum() > 5 and len(np.unique(targets[early])) == 2:
            try:
                auc_e = float(roc_auc_score(targets[early], scores[early]))
            except ValueError:
                auc_e = float("nan")
            self.log("val/novelty_auc_early", auc_e, prog_bar=True)
        self._val_buffer.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x_post = self._stack_window(batch["x_post"])
        pt_ids = batch["meta"]["pt_id"]
        cond = self.subject_embed(self._subject_indices(pt_ids))
        recon = self.ae(x_post, cond)
        per_trial_mse = ((recon - x_post) ** 2).mean(dim=(1, 2))
        self._test_buffer.append({
            "score": per_trial_mse.detach().cpu(),
            "target": batch["y"].squeeze(-1).long().detach().cpu(),
            "trial_position": torch.tensor(batch["meta"]["trial_position"]),
        })

    def on_test_epoch_end(self) -> None:
        if not self._test_buffer:
            return
        scores = torch.cat([b["score"] for b in self._test_buffer]).numpy()
        targets = torch.cat([b["target"] for b in self._test_buffer]).numpy()
        from sklearn.metrics import roc_auc_score
        try:
            self.log("test/novelty_auc", float(roc_auc_score(targets, scores)))
        except ValueError:
            self.log("test/novelty_auc", float("nan"))
        self._test_buffer.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
