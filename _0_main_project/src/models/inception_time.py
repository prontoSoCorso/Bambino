"""InceptionTime + metadata-fusion classifier (PyTorch Lightning).

Mirrors `_03_train/InceptionTime/inceptionTime.ipynb`:

    * `InceptionModule` — bottleneck (1x1) → three parallel convs at kernel
      sizes (k, k//2 + 1, k//4 + 1) → maxpool branch via 1x1 conv → concat
      (4 × nb_filters) → BN → ReLU. Outputs are length-truncated to the
      original T to absorb the rounding caused by `padding=k//2` on odd k.
    * `InceptionBlock` — `depth` stacked InceptionModules with an optional
      residual shortcut (1x1 conv + BN, ReLU after add).
    * `InceptionTimeWithMeta` — global average pool over time, concat with
      a metadata vector (age scalar + sex one-hot 2 + trial-order scalar),
      MLP head with high dropout, single logit (here: 2 logits for CE loss).

Trial-order encoding (PROJECT_STATE §3.2.2) is exposed via
`cfg.trial_order_mode`:

    * `"none"`              — disabled.
    * `"metadata"`          — appended to the post-GAP fusion vector.
    * `"broadcast_channel"` — broadcast as an extra input channel (38 → 39).

Pre-stim context (PROJECT_STATE §3.2.1 step 4) is enabled by
`cfg.use_pre_stim_context`: the pre-stim window is linearly resampled to
match the post-stim length and stacked along the channel axis (38 → 76).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import InceptionTimeConfig, ModalityConfig
from .base import BaseClassifier


# ─── Inception primitives ────────────────────────────────────────────────────
class InceptionModule(nn.Module):
    """Single InceptionTime module with 4 parallel branches.

    Branches (3 conv kernels k, k/2, k/4 + maxpool) are concatenated channel-
    wise; output channel count is `4 * nb_filters`. All branches output the
    same temporal length as the input by padding-and-truncating.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        kernel_size: int,
        bottleneck_size: int,
    ):
        super().__init__()
        self.use_bottleneck = bottleneck_size is not None and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            input_channels = bottleneck_size
        else:
            self.bottleneck = nn.Identity()
            input_channels = in_channels

        # Legacy ratios — keep odd to avoid padding-induced shape errors.
        k1 = kernel_size
        k2 = kernel_size // 2 + 1
        k3 = kernel_size // 4 + 1
        self.conv1 = nn.Conv1d(input_channels, nb_filters, k1, padding=k1 // 2, bias=False)
        self.conv2 = nn.Conv1d(input_channels, nb_filters, k2, padding=k2 // 2, bias=False)
        self.conv3 = nn.Conv1d(input_channels, nb_filters, k3, padding=k3 // 2, bias=False)
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = x.shape[-1]
        x_bottle = self.bottleneck(x)
        out1 = self.conv1(x_bottle)[..., :target_len]
        out2 = self.conv2(x_bottle)[..., :target_len]
        out3 = self.conv3(x_bottle)[..., :target_len]
        out_pool = self.conv_pool(self.maxpool(x))[..., :target_len]
        out = torch.cat([out1, out2, out3, out_pool], dim=1)
        return self.act(self.bn(out))


class InceptionBlock(nn.Module):
    """`depth` stacked InceptionModules with an optional residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        kernel_size: int,
        bottleneck_size: int,
        depth: int,
        use_residual: bool,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.depth = depth
        self.modules_list = nn.ModuleList()
        for d in range(depth):
            curr_in = in_channels if d == 0 else nb_filters * 4
            self.modules_list.append(
                InceptionModule(curr_in, nb_filters, kernel_size, bottleneck_size)
            )
        if use_residual:
            self.shortcut = nn.Conv1d(in_channels, nb_filters * 4, kernel_size=1, bias=False)
            self.bn_shortcut = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d in range(self.depth):
            x = self.modules_list[d](x)
        if self.use_residual:
            res = self.bn_shortcut(self.shortcut(res))
            x = F.relu(x + res)
        return x


# ─── LightningModule ─────────────────────────────────────────────────────────
class InceptionTimeModel(BaseClassifier):
    """Deep ensemble of inception modules + metadata-fusion head."""

    monitor_metric = "val/balanced_accuracy"
    monitor_mode = "max"

    def __init__(
        self,
        model_cfg: InceptionTimeConfig,
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

        in_channels = modality_cfg.total_channels
        if model_cfg.trial_order_mode == "broadcast_channel":
            in_channels += 1
        if model_cfg.use_pre_stim_context:
            in_channels += modality_cfg.total_channels

        self.backbone = InceptionBlock(
            in_channels=in_channels,
            nb_filters=model_cfg.nb_filters,
            kernel_size=model_cfg.kernel_size,
            bottleneck_size=model_cfg.bottleneck_size,
            depth=model_cfg.depth,
            use_residual=model_cfg.use_residual,
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        feat_dim = model_cfg.nb_filters * 4
        meta_dim = self._meta_dim()
        head_in = feat_dim + meta_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(64, num_classes),
        )

    def _meta_dim(self) -> int:
        # age (1) + sex one-hot (2) + optional trial-order scalar (1)
        d = 3
        if self.cfg.trial_order_mode == "metadata":
            d += 1
        return d

    # ──────────────────────────────────────────────────────────────────────
    def _build_meta_vector(self, batch: Dict[str, Any], B: int) -> torch.Tensor:
        meta = batch["meta"]
        age = torch.tensor(meta["age"], dtype=torch.float32, device=self.device).view(B, 1)
        sex_int = torch.tensor(meta["sex"], dtype=torch.long, device=self.device)
        sex_oh = F.one_hot(sex_int.clamp_(0, 1), num_classes=2).float()
        feats = [age, sex_oh]
        if self.cfg.trial_order_mode == "metadata":
            tp = torch.tensor(meta["trial_position"], dtype=torch.float32, device=self.device).view(B, 1)
            feats.append(tp)
        return torch.cat(feats, dim=1)

    def _build_input(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        post_dict = batch["x_post"]
        chunks = [post_dict[k].transpose(1, 2) for k in self.modality_cfg.keys]
        x = torch.cat(chunks, dim=1)  # (B, sum_D, T_post)
        B, _, T_post = x.shape

        if self.cfg.use_pre_stim_context and "x_pre" in batch:
            pre_chunks = [batch["x_pre"][k].transpose(1, 2) for k in self.modality_cfg.keys]
            pre = torch.cat(pre_chunks, dim=1)
            pre_resampled = F.interpolate(pre, size=T_post, mode="linear", align_corners=False)
            x = torch.cat([x, pre_resampled], dim=1)

        if self.cfg.trial_order_mode == "broadcast_channel":
            tp = torch.tensor(batch["meta"]["trial_position"], dtype=x.dtype, device=x.device)
            tp = tp.view(-1, 1, 1).expand(-1, 1, T_post)
            x = torch.cat([x, tp], dim=1)

        meta = self._build_meta_vector(batch, B)
        return x, meta

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x, meta = self._build_input(batch)
        h = self.backbone(x)
        h = self.gap(h).squeeze(-1)
        h = torch.cat([h, meta], dim=1)
        return self.head(h)
