"""Model layer.

Lightning-dependent classes are imported lazily so the sklearn paths and the
test suite can run without `pytorch_lightning` installed. The pure-numpy
`build_causal_baseline` helper at `manifold_utils` is always importable.
"""
from importlib import import_module
from typing import TYPE_CHECKING

from .components import FiLM, InceptionBlock, MLPHead
from .logistic_regression import LogRegResult, fit_logreg
from .manifold_utils import build_causal_baseline
from .minirocket import MiniRocketResult, fit_minirocket

if TYPE_CHECKING:  # pragma: no cover
    from .anomaly_detector import SubjectConditionedAnomalyDetector
    from .base import BaseClassifier
    from .film_cnn import FiLMCNNModel
    from .inception_time import InceptionTimeModel
    from .moment_heads import (
        MomentEmbedder,
        MomentHeadResult,
        MomentMLP,
        extract_embeddings,
        fit_moment_sklearn_head,
        fuse_features,
    )
    from .resnet_gasf import (
        GASFDataset,
        GASFPCAs,
        ResNetGASFModel,
        gasf_collate,
    )


_LAZY = {
    "BaseClassifier": (".base", "BaseClassifier"),
    "InceptionTimeModel": (".inception_time", "InceptionTimeModel"),
    "FiLMCNNModel": (".film_cnn", "FiLMCNNModel"),
    "ResNetGASFModel": (".resnet_gasf", "ResNetGASFModel"),
    "GASFDataset": (".resnet_gasf", "GASFDataset"),
    "GASFPCAs": (".resnet_gasf", "GASFPCAs"),
    "gasf_collate": (".resnet_gasf", "gasf_collate"),
    "MomentEmbedder": (".moment_heads", "MomentEmbedder"),
    "MomentMLP": (".moment_heads", "MomentMLP"),
    "MomentHeadResult": (".moment_heads", "MomentHeadResult"),
    "extract_embeddings": (".moment_heads", "extract_embeddings"),
    "fuse_features": (".moment_heads", "fuse_features"),
    "fit_moment_sklearn_head": (".moment_heads", "fit_moment_sklearn_head"),
    "SubjectConditionedAnomalyDetector": (".anomaly_detector", "SubjectConditionedAnomalyDetector"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        return getattr(import_module(mod_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
