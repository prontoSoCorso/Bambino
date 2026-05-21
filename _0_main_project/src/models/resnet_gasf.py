"""ResNet-18 (frozen ImageNet backbone) on GASF-encoded modality projections.

End-to-end migration of `_03_train/resnet_on_images/{image_creation.py,resnet.ipynb}`,
done IN MEMORY (no PNG round-trip):

    1. Per-modality PCA(1) fit on the training set — projects each of the 3
       modality time-series to a 1-D scalar series.
    2. Concatenate the 3 PC-1 series → (T, 3) tensor per trial.
    3. Per-trial MinMax-scale to [-1, 1] (GASF input requirement).
    4. Compute Gramian Angular Summation Field (GASF) per channel:
            φ_t = arccos(x_t)
            G[i, j] = cos(φ_i + φ_j) = x_i x_j − sqrt(1−x_i²)·sqrt(1−x_j²)
       Stack the 3 GASF matrices as RGB image channels → (3, T, T).
    5. Resize to 224×224 (linear interpolation), normalise with ImageNet
       mean/std, feed to a frozen ResNet-18 backbone.
    6. New head: Dropout(0.5) → Linear(num_ftrs, 32) → ReLU → Dropout(0.3)
       → Linear(32, num_classes) — only this head is trained (legacy spec).

This is a LightningModule; the GASF dataset is built once during
`prepare_data` style and passed into the train/val/test loaders. Because
the GASF transform depends on a per-modality PCA fit on the TRAIN split,
we hold the fitted PCAs on the model so they can be reused at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ..configs import ModalityConfig, ResNetGASFConfig
from ..data.dataset import BambinoDataset


# ─── GASF transform ─────────────────────────────────────────────────────────
def compute_gasf(x: np.ndarray) -> np.ndarray:
    """Gramian Angular Summation Field for a single 1-D series.

    `x` MUST be in [-1, 1]. Returns (T, T) float32 matrix.
    """
    x_clip = np.clip(x, -1.0, 1.0)
    sin_phi = np.sqrt(1.0 - x_clip ** 2)
    return (np.outer(x_clip, x_clip) - np.outer(sin_phi, sin_phi)).astype(np.float32)


def _minmax_per_trial(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel MinMax-scale to [-1, 1] over the time axis. (T, C) → (T, C)."""
    lo = arr.min(axis=0, keepdims=True)
    hi = arr.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, eps)
    return (2.0 * (arr - lo) / span - 1.0).astype(np.float32)


def _resize_2d(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize a (C, T, T) float image to (C, size, size) via torch."""
    t = torch.from_numpy(img).unsqueeze(0)
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze(0).numpy()


# ─── Per-modality PCA fitting on the train split ────────────────────────────
@dataclass
class GASFPCAs:
    """Fitted PCAs (one per modality), ready for `apply`."""
    pcas: Dict[str, PCA]

    @classmethod
    def fit(cls, train_ds: BambinoDataset, modalities=("g", "h", "f")) -> "GASFPCAs":
        """Fit PCA(1) per modality on the post-stim segments of `train_ds`."""
        pre_frames = train_ds.cfg.window.pre_stim_frames
        per_mod: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
        for inst in train_ds.instances:
            for m in modalities:
                per_mod[m].append(inst.get_modality(m)[pre_frames:])
        out: Dict[str, PCA] = {}
        for m in modalities:
            stacked = np.concatenate(per_mod[m], axis=0)
            pca = PCA(n_components=1, random_state=0)
            pca.fit(stacked)
            out[m] = pca
        return cls(pcas=out)

    def apply(self, inst, pre_frames: int) -> np.ndarray:
        """Transform a single instance's post-stim window → (T, 3) PC-1 stack."""
        chunks = []
        for m in ("g", "h", "f"):
            arr = inst.get_modality(m)[pre_frames:]
            chunks.append(self.pcas[m].transform(arr))  # (T, 1)
        return np.hstack(chunks)  # (T, 3)


# ─── In-memory GASF dataset ─────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class GASFDataset(torch.utils.data.Dataset):
    """In-memory GASF images for one BambinoDataset split.

    Images are kept as float32 tensors in [0, 1] then normalised with ImageNet
    statistics on the fly. We don't write PNGs (legacy `image_creation.py` did)
    because there's no benefit when the dataset fits in RAM.
    """

    def __init__(
        self,
        bambino_ds: BambinoDataset,
        pcas: GASFPCAs,
        image_size: int,
    ):
        super().__init__()
        self.bambino_ds = bambino_ds
        self.pcas = pcas
        self.image_size = image_size
        self.pre_frames = bambino_ds.cfg.window.pre_stim_frames
        self._cache: List[Optional[np.ndarray]] = [None] * len(bambino_ds)

    def __len__(self) -> int:
        return len(self.bambino_ds)

    def _build_image(self, idx: int) -> np.ndarray:
        """(3, H, W) float32 tensor in [0, 1] — ImageNet-norm applied later."""
        inst = self.bambino_ds.instances[idx]
        pcs = self.pcas.apply(inst, self.pre_frames)        # (T, 3)
        scaled = _minmax_per_trial(pcs)                      # (T, 3) ∈ [-1, 1]
        channels = [compute_gasf(scaled[:, c]) for c in range(3)]
        img = np.stack(channels, axis=0).astype(np.float32)  # (3, T, T) ∈ [-1, 1]
        img = (img + 1.0) / 2.0                              # → [0, 1]
        img = _resize_2d(img, self.image_size)               # (3, H, W)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache[idx] is None:
            self._cache[idx] = self._build_image(idx)
        img = self._cache[idx]
        img_norm = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        inst = self.bambino_ds.instances[idx]
        return {
            "image": torch.from_numpy(img_norm.astype(np.float32)),
            "y": torch.tensor([inst.trial_type], dtype=torch.long),
            "meta": {
                "pt_id": inst.pt_id,
                "trial_id": float(inst.trial_id),
                "trial_position": float(self.bambino_ds.trial_position(inst)),
                "age": float(inst.age),
                "sex": int(inst.sex),
                "sample_weight": float(inst.sample_weight),
            },
        }


def gasf_collate(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    meta = {k: [b["meta"][k] for b in batch] for k in batch[0]["meta"]}
    return {"image": images, "y": y, "meta": meta}


# ─── ResNet-18 LightningModule ──────────────────────────────────────────────
class ResNetGASFModel(pl.LightningModule):
    """Frozen ResNet-18 backbone + small trainable head.

    Backbone weights are ImageNet (`weights=ResNet18_Weights.DEFAULT`) and
    held FROZEN. Only the replacement classification head receives gradient.
    """

    monitor_metric: str = "val/balanced_accuracy"
    monitor_mode: str = "max"

    def __init__(self, model_cfg: ResNetGASFConfig, num_classes: int = 2):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.num_classes = num_classes

        # Lazy-import torchvision so the import doesn't crash environments
        # without torchvision installed.
        from torchvision import models
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            backbone = models.resnet18(weights=None)

        if model_cfg.backbone_frozen:
            for p in backbone.parameters():
                p.requires_grad_(False)

        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        self.net = backbone

        self._val_buffer: List[Dict[str, Any]] = []
        self._test_buffer: List[Dict[str, Any]] = []

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.net(batch["image"])

    def training_step(self, batch, _idx):
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        loss = F.cross_entropy(logits, target)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _idx):
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        loss = F.cross_entropy(logits, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_buffer.append({"logits": logits.detach().cpu(), "target": target.detach().cpu()})

    def on_validation_epoch_end(self):
        if not self._val_buffer:
            return
        from ..utils.metrics import core_metrics
        logits = torch.cat([b["logits"] for b in self._val_buffer]).numpy()
        target = torch.cat([b["target"] for b in self._val_buffer]).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        for k, v in core_metrics(target, preds, probs).items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "balanced_accuracy"))
        self._val_buffer.clear()

    def test_step(self, batch, _idx):
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        self._test_buffer.append({"logits": logits.detach().cpu(), "target": target.detach().cpu()})

    def on_test_epoch_end(self):
        if not self._test_buffer:
            return
        from ..utils.metrics import core_metrics
        logits = torch.cat([b["logits"] for b in self._test_buffer]).numpy()
        target = torch.cat([b["target"] for b in self._test_buffer]).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        for k, v in core_metrics(target, preds, probs).items():
            self.log(f"test/{k}", float(v))
        self._test_buffer.clear()

    def configure_optimizers(self):
        """Optimise ONLY the head — backbone is frozen."""
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "ResNetGASFModel has no trainable parameters — verify the new "
                "head was not also frozen."
            )
        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.max_epochs, eta_min=1e-6,
        )
        return {"optimizer": opt, "lr_scheduler": sched}


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]
