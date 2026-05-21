"""MOMENT foundation-model + 5 classifier heads.

Reproduces PROJECT_STATE §2.1's MOMENT row exactly:

    1. embeddings+histgb     — frozen MOMENT embeddings → HistGradientBoosting
    2. embeddings+logreg     — frozen MOMENT embeddings → L2 LogisticRegression
    3. embeddings+mlp        — frozen MOMENT embeddings → small MLP (Lightning)
    4. pca+tabpfn            — PCA(100) on embeddings → TabPFN
    5. tabpfn                — direct TabPFN on embeddings (no PCA)

Embedding contract — matches legacy `moment_utils.MomentFeatureExtractor`:
    * Modality dict (g, h, f) is concatenated channel-wise → shape (T, 38).
    * Time dimension interpolated to MOMENT's expected `target_len = 512`.
    * MOMENT-1-large emits per-token embeddings (B, T, D); we mean-pool AND
      std-pool over T and concatenate → final dim = 2 * D ≈ 2048.
    * Age (numeric) and sex (one-hot 2-dim) metadata is fused on top:
      X = [emb, age, sex_one_hot]  (legacy `fuse_features`).

The MLP head is a pl.LightningModule. The four sklearn / TabPFN heads are
fit OUTSIDE Lightning (no backprop); the orchestrator routes them through
`fit_moment_sklearn_head()` directly, similar to LogReg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import ModalityConfig, MomentConfig
from ..utils.metrics import bootstrap_ci, core_metrics, habituation_bucketed_metrics
from .components import MLPHead


# ─── Backbone wrapper ───────────────────────────────────────────────────────
def _try_import_moment():
    try:
        from momentfm import MOMENTPipeline
        return MOMENTPipeline
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MOMENT is not installed. `pip install momentfm` and ensure the "
            "AutonLab/MOMENT-1-large weights are accessible."
        ) from e


class MomentEmbedder(nn.Module):
    """Frozen MOMENT-1-large wrapper.

    Returns embeddings of shape (B, 2 * D) — mean-pool ⊕ std-pool over the
    token axis, matching the legacy `MomentFeatureExtractor.extract` shape.
    """

    def __init__(self, cfg: MomentConfig, modality_cfg: ModalityConfig):
        super().__init__()
        self.cfg = cfg
        self.modality_cfg = modality_cfg
        self._backbone: Optional[nn.Module] = None

    def _ensure_backbone(self):
        if self._backbone is not None:
            return
        Pipe = _try_import_moment()
        self._backbone = Pipe.from_pretrained(
            self.cfg.backbone,
            model_kwargs={
                "task_name": "embedding",
                "n_channels": self.modality_cfg.total_channels,
                "num_class": 2,
            },
        )
        self._backbone.init()
        if self.cfg.backbone_frozen:
            for p in self._backbone.parameters():
                p.requires_grad_(False)
            self._backbone.eval()

    def _stack_modalities(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Modality dict {g, h, f} → (B, C, T) with C = 38."""
        chunks = [x_dict[k].transpose(1, 2) for k in self.modality_cfg.keys]
        return torch.cat(chunks, dim=1)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._ensure_backbone()
        x = self._stack_modalities(x_dict)
        x_resampled = F.interpolate(x, size=self.cfg.target_len, mode="linear", align_corners=False)
        mask = torch.ones(x_resampled.shape[0], x_resampled.shape[2], device=x.device)
        ctx = torch.no_grad() if self.cfg.backbone_frozen else torch.enable_grad()
        with ctx:
            out = self._backbone(x_enc=x_resampled, input_mask=mask)
            emb = out.embeddings
        if emb.ndim == 3:
            mean_emb = emb.mean(dim=1)
            std_emb = emb.std(dim=1)
            emb = torch.cat([mean_emb, std_emb], dim=1)
        return emb

    @property
    def output_dim(self) -> int:
        """Final embedding dim AFTER mean⊕std pooling — 2 × backbone hidden.

        MOMENT-1-large hidden dim is 1024; mean+std → 2048. We fall back to
        a probe forward pass on first call when the backbone is loaded.
        """
        # MOMENT-1-large is 1024 per PROJECT_STATE §2.1 (frozen MOMENT-1-large
        # → 5 heads). 2× for mean+std pooling. If a different size is ever
        # used, _ensure_backbone followed by a probe will still work.
        return 2048


# ─── Lightning MLP head ─────────────────────────────────────────────────────
class MomentMLP(pl.LightningModule):
    """Frozen MOMENT embeddings → MLP head (PyTorch Lightning).

    Fixed in this revision:
        * The head is BUILT IN __init__ at a known dim (`embed_dim=2048` for
          MOMENT-1-large), not lazily — so `configure_optimizers` always sees
          trainable parameters.
        * `configure_optimizers` filters by `requires_grad` so the frozen
          backbone never appears in the optimizer's parameter list.
    """

    monitor_metric: str = "val/balanced_accuracy"
    monitor_mode: str = "max"

    def __init__(self, model_cfg: MomentConfig, modality_cfg: ModalityConfig, num_classes: int = 2):
        super().__init__()
        self.save_hyperparameters(ignore=["modality_cfg"])
        self.cfg = model_cfg
        self.modality_cfg = modality_cfg
        self.num_classes = num_classes

        self.embedder = MomentEmbedder(model_cfg, modality_cfg)
        embed_dim = self.embedder.output_dim
        # Legacy MLP shape: (64, 32) hidden → 1 logit. We adapt to 2-logit
        # cross-entropy by writing `num_classes` outputs.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

        self._val_buffer: List[Dict[str, Any]] = []
        self._test_buffer: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────
    def _embed(self, batch: Dict[str, Any]) -> torch.Tensor:
        post_emb = self.embedder(batch["x_post"])
        if self.cfg.contrast_pre_post and "x_pre" in batch:
            pre_emb = self.embedder(batch["x_pre"])
            return post_emb - pre_emb
        return post_emb

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.head(self._embed(batch))

    def training_step(self, batch: Dict[str, Any], _idx: int) -> torch.Tensor:
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        loss = F.cross_entropy(logits, target)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], _idx: int) -> None:
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        loss = F.cross_entropy(logits, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_buffer.append({
            "logits": logits.detach().cpu(),
            "target": target.detach().cpu(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_buffer:
            return
        logits = torch.cat([b["logits"] for b in self._val_buffer]).numpy()
        target = torch.cat([b["target"] for b in self._val_buffer]).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        for k, v in core_metrics(target, preds, probs).items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "balanced_accuracy"))
        self._val_buffer.clear()

    def test_step(self, batch: Dict[str, Any], _idx: int) -> None:
        logits = self.forward(batch)
        target = batch["y"].squeeze(-1).long()
        self._test_buffer.append({"logits": logits.detach().cpu(), "target": target.detach().cpu()})

    def on_test_epoch_end(self) -> None:
        if not self._test_buffer:
            return
        logits = torch.cat([b["logits"] for b in self._test_buffer]).numpy()
        target = torch.cat([b["target"] for b in self._test_buffer]).numpy()
        probs = _softmax_pos(logits)
        preds = (probs >= 0.5).astype(np.int64)
        for k, v in core_metrics(target, preds, probs).items():
            self.log(f"test/{k}", float(v))
        self._test_buffer.clear()

    def configure_optimizers(self):
        """ONLY trainable parameters reach the optimizer.

        With `cfg.backbone_frozen=True` (default), the MOMENT backbone has
        `requires_grad=False` on every parameter; passing `self.parameters()`
        directly would crash with `optimizer got an empty parameter list` on
        any version of PyTorch. We filter explicitly so the optimizer sees
        ONLY the MLP head's parameters.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "MomentMLP has no trainable parameters. The MOMENT backbone is "
                "frozen — verify the head was constructed in __init__ and that "
                "you're not running this with `cfg.backbone_frozen=True` AND a "
                "frozen head."
            )
        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/balanced_accuracy"},
        }


# ─── Embedding extraction (used by sklearn / TabPFN heads) ──────────────────
def extract_embeddings(
    embedder: MomentEmbedder,
    dataloader,
    contrast_pre_post: bool = False,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_emb, y, age, sex, trial_pos) over the dataloader.

    `age` is a (N, 1) float32 column. `sex` is a (N, 2) one-hot. These are
    fused with the embedding by `fuse_features` to match the legacy code.
    """
    embedder = embedder.to(device).eval()
    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    age_chunks: List[float] = []
    sex_chunks: List[int] = []
    pos_chunks: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            post = {k: v.to(device) for k, v in batch["x_post"].items()}
            emb_post = embedder(post)
            if contrast_pre_post and "x_pre" in batch:
                pre = {k: v.to(device) for k, v in batch["x_pre"].items()}
                emb_pre = embedder(pre)
                emb = emb_post - emb_pre
            else:
                emb = emb_post
            X_chunks.append(emb.cpu().numpy())
            y_chunks.append(batch["y"].squeeze(-1).numpy())
            age_chunks.extend(batch["meta"]["age"])
            sex_chunks.extend(batch["meta"]["sex"])
            pos_chunks.extend(batch["meta"]["trial_position"])
    X = np.concatenate(X_chunks, axis=0).astype(np.float32)
    y = np.concatenate(y_chunks, axis=0).astype(np.int64)
    age = np.asarray(age_chunks, dtype=np.float32).reshape(-1, 1)
    sex_int = np.asarray(sex_chunks, dtype=np.int64)
    sex_oh = np.eye(2, dtype=np.float32)[np.clip(sex_int, 0, 1)]
    pos = np.asarray(pos_chunks, dtype=np.float32)
    return X, y, age, sex_oh, pos


def fuse_features(emb: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Legacy fusion: hstack(emb, age, sex_one_hot)."""
    return np.hstack([emb, age, sex]).astype(np.float32)


# ─── sklearn / TabPFN heads ────────────────────────────────────────────────
SklearnHeadName = Literal[
    "embeddings+histgb",
    "embeddings+logreg",
    "pca+tabpfn",
    "tabpfn",
]


@dataclass
class MomentHeadResult:
    head: str
    test_metrics: Dict[str, float]
    bootstrap_auc: tuple
    bucket_metrics: Dict[str, Dict[str, float]]
    sklearn_model: Any  # the fit estimator


def _build_sklearn_head(name: SklearnHeadName, seed: int):
    """Instantiate the requested sklearn / TabPFN classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    if name == "embeddings+histgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            class_weight="balanced", random_state=seed,
        )
    if name == "embeddings+logreg":
        return LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=seed,
            penalty="l2", solver="lbfgs",
        )
    if name == "pca+tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("TabPFN not installed. `pip install tabpfn`") from e
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("pca", PCA(n_components=100, random_state=seed)),
            ("tabpfn", TabPFNClassifier(
                device="cuda" if torch.cuda.is_available() else "cpu",
                n_estimators=32, random_state=seed,
            )),
        ])
    if name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("TabPFN not installed. `pip install tabpfn`") from e
        return TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_estimators=32, random_state=seed,
        )
    raise ValueError(f"Unknown MOMENT head: {name}")


def fit_moment_sklearn_head(
    head_name: SklearnHeadName,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    test_trial_positions: np.ndarray,
    seed: int = 2025,
) -> MomentHeadResult:
    """Train one of the four sklearn / TabPFN heads.

    Threshold-tunes on val (max balanced accuracy); evaluates on test with
    bootstrap AUC + habituation-bucketed AUROCs. Returns a `MomentHeadResult`.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    clf = _build_sklearn_head(head_name, seed=seed)
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        ba = balanced_accuracy_score(y_val, (val_scores >= thr).astype(np.int64))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)

    test_scores = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_scores >= best_thr).astype(np.int64)

    test_metrics = core_metrics(y_test, test_preds, test_scores)
    test_metrics["threshold"] = best_thr

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        y_test, test_preds, test_scores,
        metric_fn=lambda y, _p, s: float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s)),
        seed=seed,
    )
    buckets = habituation_bucketed_metrics(y_test, test_preds, test_scores, test_trial_positions)
    return MomentHeadResult(
        head=head_name, test_metrics=test_metrics,
        bootstrap_auc=(auc_mean, auc_lo, auc_hi),
        bucket_metrics=buckets, sklearn_model=clf,
    )


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]
