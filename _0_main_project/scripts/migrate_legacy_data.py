"""Legacy → clean data migration bridge.

The legacy `.pt` files in `data/full/raw/` were saved as pickled instances of
the legacy classes:

    DataUtils.BoaOpenFaceDataset.BoaOpenFaceDataset
    DataUtils.OpenFaceDataset.OpenFaceDataset
    DataUtils.OpenFaceInstance.OpenFaceInstance

Re-importing them under the new architecture would force us to drag the legacy
namespace into `src/`, polluting the clean rewrite. Instead, this script:

    1. Installs minimal STUB classes at the legacy module paths so pickle's
       unpickler can find them. Stubs override `__setstate__` / `__init__` to
       absorb the pickled state without running the legacy constructor.
    2. Loads each legacy `.pt` file and walks `obj.instances`, extracting the
       raw fields (pt_id, sex, age, trial_id, trial_type, audio, speaker, and
       the gaze/head/face arrays) into pure-dict form.
    3. Writes the result back as a `torch.save` of a plain dict — no custom
       Python classes, safely loadable from any future architecture.

Default paths:
    INPUT  = `<repo>/_0_main_project/data/full/raw/`     (legacy pickles in-place)
    OUTPUT = `<repo>/_0_main_project/data/full/clean/`   (clean dicts)

Output structure per split:
    {
        'instances': [
            {
                'pt_id': str, 'sex': int, 'age': float,
                'trial_id': float, 'trial_type': int,
                'audio': str, 'speaker': int|None,
                'gaze_info': np.ndarray[T, 8],
                'head_info': np.ndarray[T, 13],
                'face_info': np.ndarray[T, 17],
            },
            ...
        ],
        'metadata': {
            'source_path': str,
            'frame_rate': 25,
            'modality_dims': {'g': 8, 'h': 13, 'f': 17},
            'split': str,                     # 'train' | 'val' | 'test'
            'n_instances': int,
            'n_subjects': int,
        },
    }

Plus a combined file `bambino_clean.pt` concatenating all splits' instances —
the new pipeline re-splits this with its own subject-grouped logic.
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
import types
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ─── 1. Stub installation ───────────────────────────────────────────────────
class _LegacyStub:
    """Generic stub. Pickle restoration calls `__new__` then writes state.

    We deliberately accept any keyword arguments in __init__ so that even if
    a future legacy class uses `copyreg.__reduce__` paths that re-invoke
    `__init__`, the construction still succeeds.
    """
    def __init__(self, *args, **kwargs):
        # Absorb any positional / keyword args silently.
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            # (slots, dict) tuple form
            self.__dict__.update(state[1])
        else:
            self.__dict__["_pickled_state"] = state


_LEGACY_CLASS_NAMES = {
    "OpenFaceInstance",
    "OpenFaceDataset",
    "BoaOpenFaceDataset",
    "ToyOpenFaceDataset",
}


def install_legacy_stubs() -> None:
    """Install stubs at every plausible module path the legacy code may have used.

    Different vintages of the legacy code pickled the same classes from
    different namespaces depending on whether the script ran as `__main__`,
    via a notebook, or imported the `DataUtils.*` modules. We pre-populate
    `__main__`, `DataUtils`, and the per-file submodule paths so the standard
    pickle resolver finds the class regardless of provenance. Anything we
    miss is caught by `_TolerantUnpickler.find_class` below.
    """
    targets = [
        ("DataUtils", _LEGACY_CLASS_NAMES),
        ("DataUtils.OpenFaceInstance", {"OpenFaceInstance"}),
        ("DataUtils.OpenFaceDataset", {"OpenFaceDataset"}),
        ("DataUtils.BoaOpenFaceDataset", {"BoaOpenFaceDataset"}),
        ("DataUtils.ToyOpenFaceDataset", {"ToyOpenFaceDataset"}),
        ("__main__", _LEGACY_CLASS_NAMES),
    ]

    if "DataUtils" not in sys.modules:
        pkg = types.ModuleType("DataUtils")
        pkg.__path__ = []  # treat as package
        sys.modules["DataUtils"] = pkg

    for module_name, class_names in targets:
        if module_name not in sys.modules:
            mod = types.ModuleType(module_name)
            sys.modules[module_name] = mod
            if module_name.startswith("DataUtils."):
                short = module_name.rsplit(".", 1)[-1]
                setattr(sys.modules["DataUtils"], short, mod)
        mod = sys.modules[module_name]
        for cname in class_names:
            if not hasattr(mod, cname):
                cls = type(cname, (_LegacyStub,), {})
                cls.__module__ = module_name
                cls.__qualname__ = cname
                setattr(mod, cname, cls)

    # Defensive `config` stub — the legacy __init__ refers to `settings`.
    if "config" not in sys.modules:
        config_mod = types.ModuleType("config")

        class _Settings:
            seed = 2025
            DATASETS_ROOT = ""

            @staticmethod
            def seed_everything(_seed):
                return None

            @staticmethod
            def get_dataset_path(*_a, **_kw):
                return ""

        class _Config02Norm:
            input_dataset_type = "augmented"
            output_dataset_type = "augmented_normalized"

        config_mod.settings = _Settings()
        config_mod.Config_02_normalization = _Config02Norm
        sys.modules["config"] = config_mod


# ─── Tolerant unpickler ─────────────────────────────────────────────────────
class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler that synthesises a stub for any missing class.

    PyTorch's `torch.load` lets us inject a custom `pickle_module`. We exploit
    that to override `find_class` so we never crash on `AttributeError`/
    `ModuleNotFoundError` for unknown legacy classes — instead we mint a
    `_LegacyStub` subclass on the fly, register it at the requested module
    path, and return it. The pickled state is then restored into the stub's
    `__dict__` exactly as the original code would have.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            if module not in sys.modules:
                m = types.ModuleType(module)
                # Mark as package so dotted children resolve.
                m.__path__ = []  # type: ignore[attr-defined]
                sys.modules[module] = m
            mod = sys.modules[module]
            cls = type(name, (_LegacyStub,), {})
            cls.__module__ = module
            cls.__qualname__ = name
            setattr(mod, name, cls)
            return cls


class _TolerantPickleModule:
    """Drop-in replacement for the `pickle` module, for `torch.load`."""
    Unpickler = _TolerantUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _TolerantUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return _TolerantUnpickler(io.BytesIO(data), **kwargs).load()


# ─── 2. Extraction ──────────────────────────────────────────────────────────
def _coerce_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_str(v: Any, default: str = "") -> str:
    return str(v) if v is not None else default


def _coerce_array(v: Any, expected_dim: int) -> np.ndarray:
    """Force `v` into a float32 numpy array of shape [T, expected_dim].

    Accepts numpy arrays or torch tensors; raises if the channel count is wrong.
    """
    if v is None:
        return np.zeros((0, expected_dim), dtype=np.float32)
    if hasattr(v, "numpy"):
        try:
            v = v.detach().cpu().numpy()
        except AttributeError:
            v = v.numpy()
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"Channel-count mismatch: expected {expected_dim}, got {arr.shape[1]} "
            f"(array shape {arr.shape})"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def instance_to_dict(inst: Any) -> Dict[str, Any]:
    """Convert one legacy stub `OpenFaceInstance` to a clean dict.

    Tolerant to missing fields — the legacy code occasionally pickled
    instances mid-pipeline with partial state.
    """
    d = inst.__dict__
    return {
        "pt_id": _coerce_str(d.get("pt_id"), ""),
        "sex": _coerce_int(d.get("sex"), default=0),
        "age": _coerce_float(d.get("age")),
        "trial_id": _coerce_float(d.get("trial_id")),
        "trial_type": _coerce_int(d.get("trial_type"), default=0),
        "audio": _coerce_str(d.get("audio"), ""),
        "speaker": _coerce_int(d.get("speaker"), default=None),
        "gaze_info": _coerce_array(d.get("gaze_info"), 8),
        "head_info": _coerce_array(d.get("head_info"), 13),
        "face_info": _coerce_array(d.get("face_info"), 17),
    }


def _load_legacy_blob(legacy_pt_path: str):
    """Load a legacy `.pt` file using whichever serialiser produced it.

    The legacy code mixed two serialisation paths:
        * `torch.save(obj, f)` — wraps pickle in a torch envelope.
        * `pickle.dump(obj, f)` (via `OpenFaceDataset._save`) — raw pickle stream.

    Both write a `.pt` extension. We try torch.load first; on `Invalid magic
    number` we fall back to plain pickle.load. Both paths use the TOLERANT
    unpickler that synthesises stubs for any class it cannot find.
    """
    try:
        return torch.load(
            legacy_pt_path,
            weights_only=False,
            map_location="cpu",
            pickle_module=_TolerantPickleModule,
        )
    except (RuntimeError, pickle.UnpicklingError) as e:
        msg = str(e).lower()
        if "magic number" not in msg and "invalid load key" not in msg:
            raise
        # Fall through to raw-pickle path.
    with open(legacy_pt_path, "rb") as f:
        return _TolerantUnpickler(f).load()


def extract_split(legacy_pt_path: str) -> List[Dict[str, Any]]:
    """Unpickle a legacy `.pt` and return a list of clean dicts."""
    blob = _load_legacy_blob(legacy_pt_path)
    instances = getattr(blob, "instances", None)
    if instances is None and isinstance(blob, dict):
        instances = blob.get("instances")
    if instances is None:
        raise RuntimeError(
            f"Legacy file {legacy_pt_path} does not expose `.instances`. "
            f"Top-level type: {type(blob).__name__}; attributes: {list(getattr(blob, '__dict__', {}).keys())}"
        )
    return [instance_to_dict(i) for i in instances]


# ─── 3. Save & combine ──────────────────────────────────────────────────────
def save_clean_split(out_path: str, instances: List[Dict[str, Any]], split: str, source: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_subjects = len({i["pt_id"] for i in instances})
    payload = {
        "instances": instances,
        "metadata": {
            "source_path": source,
            "frame_rate": 25,
            "modality_dims": {"g": 8, "h": 13, "f": 17},
            "split": split,
            "n_instances": len(instances),
            "n_subjects": n_subjects,
        },
    }
    torch.save(payload, out_path)
    print(f"  → wrote {out_path}: {len(instances)} instances, {n_subjects} subjects")


def write_combined(out_path: str, all_instances: List[Dict[str, Any]], sources: List[str]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_subjects = len({i["pt_id"] for i in all_instances})
    payload = {
        "instances": all_instances,
        "metadata": {
            "source_paths": sources,
            "frame_rate": 25,
            "modality_dims": {"g": 8, "h": 13, "f": 17},
            "split": "combined",
            "n_instances": len(all_instances),
            "n_subjects": n_subjects,
        },
    }
    torch.save(payload, out_path)
    print(f"  → wrote combined {out_path}: {len(all_instances)} instances, {n_subjects} subjects")


# ─── CLI ───────────────────────────────────────────────────────────────────
DEFAULT_INPUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "full", "raw",
))
DEFAULT_OUTPUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "full", "clean",
))

DEFAULT_SPLIT_FILES = [
    ("training_set.pt", "train"),
    ("validation_set.pt", "val"),
    ("test_set.pt", "test"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Directory holding the legacy *.pt files")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory to write clean *.pt files")
    p.add_argument(
        "--combined-name",
        default="bambino_clean.pt",
        help="Filename for the combined (re-splittable) clean blob",
    )
    p.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip writing the combined `bambino_clean.pt` (per-split files only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    install_legacy_stubs()

    if not os.path.isdir(args.input):
        raise SystemExit(f"Input directory does not exist: {args.input}")

    all_instances: List[Dict[str, Any]] = []
    sources: List[str] = []

    for fname, split in DEFAULT_SPLIT_FILES:
        src = os.path.join(args.input, fname)
        if not os.path.isfile(src):
            print(f"  ! skipping missing file: {src}")
            continue
        print(f"Migrating {src} ({split}) …")
        instances = extract_split(src)
        out = os.path.join(args.output, fname)
        save_clean_split(out, instances, split=split, source=src)
        all_instances.extend(instances)
        sources.append(src)

    if not all_instances:
        raise SystemExit(
            f"No legacy instances were extracted from {args.input}. "
            "Check that the directory contains training_set.pt / validation_set.pt / test_set.pt."
        )

    if not args.skip_combined:
        combined_path = os.path.join(args.output, args.combined_name)
        write_combined(combined_path, all_instances, sources)

    print(f"\nDone. Clean files in: {args.output}")
    print(
        "Point the new pipeline at this directory:\n"
        f"  python main.py --model anomaly_detector --run-id ad_v1 --data-dir {args.output}"
    )


if __name__ == "__main__":
    main()
