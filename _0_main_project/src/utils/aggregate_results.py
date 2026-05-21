"""Aggregate every completed run under `results/` into comparative plots.

Each run leaves artifacts in one of two formats:

    Lightning runs (inception_time, moment, anomaly_detector):
        results/csv_logs/<run_id>/version_<n>/metrics.csv  ← one row per epoch
        results/tb_logs/<run_id>/version_<n>/              ← TensorBoard
        results/<run_id>/checkpoints/                       ← best ckpt

    Sklearn runs (logreg):
        results/<run_id>/metrics.json
        {
            "test": {"balanced_accuracy": ..., "roc_auc": ..., ...},
            "bootstrap_auc": [mean, lo, hi],
            "bucket": {"early": {...}, "mid": {...}, "late": {...}}
        }

This module sweeps both shapes, normalises them into a single per-run record,
and renders a set of comparative figures into `results/aggregated_grid/`.
ALL plots draw exclusively from the project palette enforced in
`utils.plotting`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import (
    ACCENT,
    PRIMARY,
    REFERENCE,
    SECONDARY,
    TERTIARY_A,
    TERTIARY_B,
    apply_default_style,
)


# Family → palette mapping (used to colour bars by model family). Every
# colour here is sourced verbatim from the project palette enforced in
# `utils.plotting`; do not introduce new hex strings.
FAMILY_COLORS = {
    "logreg": SECONDARY,            # Blue   — sklearn baseline
    "minirocket": TERTIARY_B,       # Sand   — kernel transform + GBM
    "inception_time": PRIMARY,      # Wine   — deep TS
    "film_cnn": ACCENT,             # Rose   — metadata-conditioned CNN
    "resnet_gasf": TERTIARY_A,      # Teal   — vision domain
    "moment": TERTIARY_A,           # Teal   — foundation model (any head)
    "anomaly_detector": ACCENT,     # Rose   — research direction
    "other": REFERENCE,             # Grey   — uncategorised
}

# Metrics surfaced in the comparison table / plots.
PRIMARY_METRICS = ["balanced_accuracy", "roc_auc", "f1", "specificity"]


@dataclass
class RunRecord:
    run_id: str
    family: str  # logreg / inception_time / moment / anomaly_detector / minirocket / other
    test_metrics: Dict[str, float] = field(default_factory=dict)
    bootstrap_auc: Optional[List[float]] = None  # [mean, lo, hi]
    buckets: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def color(self) -> str:
        return FAMILY_COLORS.get(self.family, REFERENCE)


# ─── Family inference ───────────────────────────────────────────────────────
def infer_family(run_id: str) -> str:
    """Heuristic: substring-match `run_id` against known family tokens.

    Order matters: more specific tokens first so `moment_logreg` doesn't fall
    into the `logreg` family (it's a MOMENT head, family=`moment`). Same for
    `resnet_gasf` (vision branch, not a generic resnet token).
    """
    rid = run_id.lower()
    # MOMENT heads first — they may contain `logreg` in the run-id.
    if "moment" in rid:
        return "moment"
    if "resnet" in rid or "gasf" in rid:
        return "resnet_gasf"
    if "film" in rid:
        return "film_cnn"
    if "incept" in rid:
        return "inception_time"
    if rid.startswith("ad_") or rid.endswith("_ad") or "_ad_" in rid or rid == "ad":
        return "anomaly_detector"
    if "logreg" in rid:
        return "logreg"
    if "mini" in rid or "rocket" in rid:
        return "minirocket"
    return "other"


# ─── Loaders ────────────────────────────────────────────────────────────────
def _last_test_row(metrics_csv: str) -> Optional[pd.Series]:
    """Return the LAST row of a Lightning CSV that contains any `test/*` cols.

    Lightning logs each metric in its own row when called from `Trainer.test()`,
    so we collapse the test-section rows into a single record.
    """
    try:
        df = pd.read_csv(metrics_csv)
    except Exception:
        return None
    test_cols = [c for c in df.columns if c.startswith("test/")]
    if not test_cols:
        return None
    test_df = df[test_cols].dropna(how="all")
    if test_df.empty:
        return None
    return test_df.iloc[-1]


def load_lightning_run(run_id: str, results_dir: str) -> Optional[RunRecord]:
    """Read the latest CSV log for `<run_id>`. Returns None if no test row.

    The anomaly detector logs `test/novelty_auc` (no class predictions); we
    surface it under the `roc_auc` key so it shows up alongside supervised
    models on the comparative plots.
    """
    pattern = os.path.join(results_dir, "csv_logs", run_id, "version_*", "metrics.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None

    latest = max(candidates, key=os.path.getmtime)
    row = _last_test_row(latest)
    if row is None:
        return None
    test_metrics: Dict[str, float] = {}
    for col, val in row.items():
        if col.startswith("test/") and not pd.isna(val):
            test_metrics[col[len("test/"):]] = float(val)
    # Anomaly detector parity: surface novelty_auc as roc_auc when no
    # supervised AUC was produced.
    if "roc_auc" not in test_metrics and "novelty_auc" in test_metrics:
        test_metrics["roc_auc"] = test_metrics["novelty_auc"]
    if not test_metrics:
        return None
    return RunRecord(
        run_id=run_id,
        family=infer_family(run_id),
        test_metrics=test_metrics,
    )


def load_sklearn_run(run_id: str, results_dir: str) -> Optional[RunRecord]:
    """Read `<run_id>/metrics.json` for sklearn-path runs (LogReg)."""
    path = os.path.join(results_dir, run_id, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    test_metrics = {k: float(v) for k, v in blob.get("test", {}).items()
                    if isinstance(v, (int, float))}
    return RunRecord(
        run_id=run_id,
        family=infer_family(run_id),
        test_metrics=test_metrics,
        bootstrap_auc=blob.get("bootstrap_auc"),
        buckets={k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                 for k, v in blob.get("bucket", {}).items()},
    )


def discover_runs(results_dir: str) -> List[RunRecord]:
    """Walk `results_dir` and collect every distinct run, by either path."""
    seen: Dict[str, RunRecord] = {}

    # Lightning runs: one folder per run_id under csv_logs/
    csv_logs_dir = os.path.join(results_dir, "csv_logs")
    if os.path.isdir(csv_logs_dir):
        for run_id in sorted(os.listdir(csv_logs_dir)):
            rec = load_lightning_run(run_id, results_dir)
            if rec is not None:
                seen[run_id] = rec

    # Sklearn runs: one folder per run_id at top of results/, holding metrics.json
    for entry in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, entry)
        if not os.path.isdir(full) or entry in {"csv_logs", "tb_logs", "aggregated_grid"}:
            continue
        if entry in seen:
            continue
        rec = load_sklearn_run(entry, results_dir)
        if rec is not None:
            seen[entry] = rec

    return list(seen.values())


# ─── Plot helpers ───────────────────────────────────────────────────────────
def _save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bar(records: List[RunRecord], metric: str, out_path: str) -> None:
    """Horizontal bar chart of `metric` across all runs, coloured by family."""
    apply_default_style()
    records = [r for r in records if metric in r.test_metrics]
    if not records:
        return
    records = sorted(records, key=lambda r: r.test_metrics[metric])
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.32 * len(records))))
    y = np.arange(len(records))
    values = [r.test_metrics[metric] for r in records]
    colors = [r.color for r in records]
    ax.barh(y, values, color=colors, edgecolor=REFERENCE, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r.run_id for r in records], fontsize=8)
    ax.axvline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"Test {metric} across all runs")

    # Family legend
    families = sorted({r.family for r in records})
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS.get(f, REFERENCE)) for f in families]
    ax.legend(handles, families, loc="lower right", frameon=False, title="Family")

    _save(fig, out_path)


def plot_balanced_acc_vs_auroc(records: List[RunRecord], out_path: str) -> None:
    """Scatter: x = balanced accuracy, y = ROC AUC, colour = family."""
    apply_default_style()
    records = [r for r in records
               if "balanced_accuracy" in r.test_metrics and "roc_auc" in r.test_metrics]
    if not records:
        return
    fig, ax = plt.subplots()
    for r in records:
        ax.scatter(
            r.test_metrics["balanced_accuracy"],
            r.test_metrics["roc_auc"],
            color=r.color,
            s=60,
            edgecolors="white",
            linewidths=0.6,
            label=r.family,
        )
        ax.annotate(r.run_id, (r.test_metrics["balanced_accuracy"], r.test_metrics["roc_auc"]),
                    fontsize=7, alpha=0.75, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.5, color=REFERENCE, linestyle="--", lw=1)
    ax.axvline(0.5, color=REFERENCE, linestyle="--", lw=1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("ROC AUC")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Balanced accuracy vs ROC AUC — all runs")

    # Dedup family legend
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab); h2.append(h); l2.append(lab)
    ax.legend(h2, l2, frameon=False, title="Family")
    _save(fig, out_path)


def plot_auc_comparison_grouped(records: List[RunRecord], out_path: str) -> None:
    """Grouped AUC comparison: bars by family, with the chance line.

    Replaces the historical "all models on one ROC curve" plot — we don't
    have FPR/TPR curves available from sklearn paths, so an AUC-mean bar
    chart grouped by family is the directly comparable equivalent.
    """
    apply_default_style()
    records = [r for r in records if "roc_auc" in r.test_metrics]
    if not records:
        return
    by_family: Dict[str, List[RunRecord]] = {}
    for r in records:
        by_family.setdefault(r.family, []).append(r)
    families = sorted(by_family.keys())

    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(families)), 4.5))
    xs, ys, colors, labels = [], [], [], []
    pos = 0
    for fam in families:
        for r in sorted(by_family[fam], key=lambda x: x.run_id):
            xs.append(pos)
            ys.append(r.test_metrics["roc_auc"])
            colors.append(FAMILY_COLORS.get(fam, REFERENCE))
            labels.append(r.run_id)
            pos += 1
        pos += 1  # gap between families
    ax.bar(xs, ys, color=colors, edgecolor=REFERENCE, linewidth=0.5)
    ax.axhline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Test ROC AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Test AUROC across all runs (grouped by family)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS.get(f, REFERENCE)) for f in families]
    ax.legend(handles, families, loc="lower right", frameon=False, title="Family")
    _save(fig, out_path)


def plot_bootstrap_auc_intervals(records: List[RunRecord], out_path: str) -> None:
    """Whisker plot of bootstrap AUC CIs for runs that produced them (logreg)."""
    apply_default_style()
    records = [r for r in records if r.bootstrap_auc is not None and len(r.bootstrap_auc) >= 3]
    if not records:
        return
    records = sorted(records, key=lambda r: r.bootstrap_auc[0])
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(records))))
    y = np.arange(len(records))
    means = [r.bootstrap_auc[0] for r in records]
    los = [r.bootstrap_auc[0] - r.bootstrap_auc[1] for r in records]
    his = [r.bootstrap_auc[2] - r.bootstrap_auc[0] for r in records]
    ax.errorbar(
        means, y, xerr=[los, his],
        fmt="o", color=PRIMARY, ecolor=SECONDARY,
        elinewidth=2, capsize=3, markersize=6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([r.run_id for r in records])
    ax.axvline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    ax.set_xlabel("Bootstrap AUROC (mean ± 95% CI)")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Bootstrap AUROC intervals")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out_path)


def plot_habituation_buckets_grid(records: List[RunRecord], out_path: str) -> None:
    """For runs that report bucket metrics, plot AUROC by early/mid/late."""
    apply_default_style()
    records = [r for r in records if r.buckets]
    if not records:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bucket_names = ["early", "mid", "late"]
    bucket_colors = [TERTIARY_A, TERTIARY_B, ACCENT]
    width = 0.25
    x = np.arange(len(records))

    for i, name in enumerate(bucket_names):
        vals = [r.buckets.get(name, {}).get("roc_auc", float("nan")) for r in records]
        ax.bar(x + (i - 1) * width, vals, width, color=bucket_colors[i], label=name, edgecolor=REFERENCE)
    ax.axhline(0.5, color=REFERENCE, linestyle="--", lw=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([r.run_id for r in records], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0, 1)
    ax.set_title("AUROC by session bucket (early ⅓ / mid ⅓ / late ⅓)")
    ax.legend(frameon=False)
    _save(fig, out_path)


# ─── Summary CSV ────────────────────────────────────────────────────────────
def write_summary_csv(records: List[RunRecord], out_path: str) -> None:
    rows = []
    for r in records:
        row = {"run_id": r.run_id, "family": r.family}
        for m in PRIMARY_METRICS:
            row[m] = r.test_metrics.get(m)
        if r.bootstrap_auc is not None and len(r.bootstrap_auc) >= 3:
            row["auc_boot_mean"] = r.bootstrap_auc[0]
            row["auc_boot_low"] = r.bootstrap_auc[1]
            row["auc_boot_high"] = r.bootstrap_auc[2]
        for bucket_name in ("early", "mid", "late"):
            row[f"{bucket_name}_auc"] = r.buckets.get(bucket_name, {}).get("roc_auc")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


# ─── CLI ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True, help="Project results/ directory.")
    p.add_argument("--output-dir", required=True, help="Where to write the aggregated grid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.results_dir):
        raise SystemExit(f"Results directory does not exist: {args.results_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    records = discover_runs(args.results_dir)
    if not records:
        print(f"[!] No completed runs found under {args.results_dir}.")
        return

    print(f"Found {len(records)} runs:")
    for r in records:
        print(f"  • {r.family:18s} {r.run_id:30s} → {r.test_metrics}")

    # Per-metric bars
    for metric in PRIMARY_METRICS:
        plot_metric_bar(records, metric, os.path.join(args.output_dir, f"compare_{metric}.png"))

    plot_balanced_acc_vs_auroc(records, os.path.join(args.output_dir, "scatter_bal_acc_vs_auroc.png"))
    plot_auc_comparison_grouped(records, os.path.join(args.output_dir, "auc_comparison_grouped.png"))
    plot_bootstrap_auc_intervals(records, os.path.join(args.output_dir, "bootstrap_auc_intervals.png"))
    plot_habituation_buckets_grid(records, os.path.join(args.output_dir, "habituation_buckets.png"))
    write_summary_csv(records, os.path.join(args.output_dir, "summary.csv"))

    print(f"\n→ Comparative plots and summary.csv written to {args.output_dir}")


if __name__ == "__main__":
    main()
