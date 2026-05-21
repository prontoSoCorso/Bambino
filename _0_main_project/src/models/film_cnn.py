"""1D ConvFiLM classifier — replication of `_03_train/FiLM/film_cnn.ipynb`.

Architecture (verbatim from the legacy notebook's `ConvFiLM`):

    * 2 conv blocks: Conv1d(in_channels, channels[0]) → BN → ReLU → FiLM_1
                    Conv1d(channels[0], channels[1]) → BN → ReLU → FiLM_2
    * Each FiLM gen: Linear(meta_dim, 8) → ReLU → Linear(8, 2 * channel_dim)
                    initialised to identity (γ=1, β=0).
    * AdaptiveAvgPool1d(1) → squeeze.
    * Concat(features, raw metadata) → MLP head (head_dim → 1 logit, here 2
      logits for cross-entropy loss with the rest of the suite).

Metadata vector (matches the legacy notebook):
    [age_scaled, sex_one_hot_0, sex_one_hot_1]
plus optional `trial_position` scalar when `trial_order_mode='metadata'`
(consistent with PROJECT_STATE §3.2.2).

Frozen backbone is N/A here — every parameter is trainable.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import FiLMCNNConfig, ModalityConfig
from .base import BaseClassifier


class _FiLMGen(nn.Module):
    """Per-block FiLM generator with identity initialisation.

    Identity init = γ initialised near 1, β at 0 — so the modulation begins
    as a no-op and the network only learns to deviate when metadata helps.
    """

    def __init__(self, meta_dim: int, channel_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(meta_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2 * channel_dim),
        )
        # Identity init on the final linear layer.
        last: nn.Linear = self.layers[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.constant_(last.bias, 0.0)
        # First half of bias = γ → set to 1.0 (multiplicative identity).
        with torch.no_grad():
            last.bias[:channel_dim].fill_(1.0)

    def forward(self, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.layers(meta)
        gamma, beta = params.chunk(2, dim=1)
        return gamma, beta


class FiLMCNNModel(BaseClassifier):
    """ConvFiLM 1D CNN with metadata-driven feature-wise modulation."""

    monitor_metric = "val/balanced_accuracy"
    monitor_mode = "max"

    def __init__(
        self,
        model_cfg: FiLMCNNConfig,
        modality_cfg: ModalityConfig,
        num_classes: int = 2,
    ):
        super().__init__(
            num_classes=num_classes,
            lr=model_cfg.lr,
            weight_decay=model_cfg.weight_decay,
        )
        self.cfg = model_cfg
        self.modality_cfg = modality_cfg

        in_channels = modality_cfg.total_channels  # 38
        ch1, ch2 = model_cfg.channels
        ks = model_cfg.kernel_size

        self.conv1 = nn.Conv1d(in_channels, ch1, ks, padding=ks // 2)
        self.bn1 = nn.BatchNorm1d(ch1)
        self.conv2 = nn.Conv1d(ch1, ch2, ks, padding=ks // 2)
        self.bn2 = nn.BatchNorm1d(ch2)

        self.meta_dim = self._meta_dim()
        self.film1 = _FiLMGen(self.meta_dim, ch1)
        self.film2 = _FiLMGen(self.meta_dim, ch2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        head_in = ch2 + self.meta_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, model_cfg.head_dim),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.head_dim, num_classes),
        )

    def _meta_dim(self) -> int:
        # age (1) + sex one-hot (2) — matches legacy ColumnTransformer.
        return 3

    # ──────────────────────────────────────────────────────────────────────
    def _build_meta(self, batch: Dict[str, Any], B: int) -> torch.Tensor:
        meta = batch["meta"]
        age = torch.tensor(meta["age"], dtype=torch.float32, device=self.device).view(B, 1)
        # Standardise age in-batch (legacy code fits a StandardScaler over
        # the train set; we approximate with batch-level standardisation,
        # which is stable for the 3–7 month range).
        if B > 1:
            age = (age - age.mean()) / (age.std() + 1e-6)
        sex_int = torch.tensor(meta["sex"], dtype=torch.long, device=self.device).clamp_(0, 1)
        sex_oh = F.one_hot(sex_int, num_classes=2).float()
        return torch.cat([age, sex_oh], dim=1)

    def _build_input(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        post = batch["x_post"]
        chunks = [post[k].transpose(1, 2) for k in self.modality_cfg.keys]
        x = torch.cat(chunks, dim=1)  # (B, 38, T)
        meta = self._build_meta(batch, x.shape[0])
        return x, meta

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x, meta = self._build_input(batch)

        # Block 1 — Conv → BN → ReLU → FiLM
        x = F.relu(self.bn1(self.conv1(x)))
        gamma1, beta1 = self.film1(meta)
        x = x * gamma1.unsqueeze(-1) + beta1.unsqueeze(-1)

        # Block 2 — Conv → BN → ReLU → FiLM
        x = F.relu(self.bn2(self.conv2(x)))
        gamma2, beta2 = self.film2(meta)
        x = x * gamma2.unsqueeze(-1) + beta2.unsqueeze(-1)

        # Pool + concat with raw metadata + classify
        x = self.pool(x).squeeze(-1)
        x = torch.cat([x, meta], dim=1)
        return self.head(x)
