"""Reusable neural building blocks (Inception block, FiLM, MLP head)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """1D inception block (Fawaz et al. 2020)."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        kernel_sizes=(10, 20, 40),
        bottleneck_size: int = 8,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_bottleneck = bottleneck_size > 0 and in_channels > 1

        bottleneck_in = bottleneck_size if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_size, kernel_size=1, padding=0, bias=False
            )

        self.conv_list = nn.ModuleList([
            nn.Conv1d(bottleneck_in, nb_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        out_channels = nb_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)

        if use_residual:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        conv_outs = [conv(x) for conv in self.conv_list]
        pool_out = self.conv_pool(self.maxpool(x_in))
        out = torch.cat(conv_outs + [pool_out], dim=1)
        out = F.relu(self.bn(out))
        if self.use_residual:
            res = self.shortcut_bn(self.shortcut(x_in))
            out = F.relu(out + res)
        return out


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta from a conditioning vector."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g_b = self.linear(cond)
        gamma, beta = g_b.chunk(2, dim=-1)
        # x: (B, C, T) — broadcast (B, C) → (B, C, 1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class MLPHead(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
