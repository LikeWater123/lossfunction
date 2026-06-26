#!/usr/bin/env python3
"""Aggregate experiment results from runs/ into result tables and plots.

Scans ``runs/<dataset>_<model>_<loss>_seed<seed>/history.json`` files,
extracts final Top-1 / ECE, builds 4 result tables (one per dataset×model
combo), and writes:
  - ``runs/summary.json``: nested dict of all results
  - ``runs/results_table.csv``: flat CSV
  - ``runs/results_table.png``: bar chart + table figure

Usage:
    python scripts/aggregate_results.py [--runs-dir runs] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSS_ORDER = ["ce", "focal", "poly", "lace_multi", "f_multi",
              "mba_ce", "mba_f", "mba_ps"]
LOSS_LABELS = {
    "ce": "CE",
    "focal": "Focal",
    "poly": "PolyLoss",
    "lace_multi": "LACE-Multi",
    "f_multi": "f-Multi",
    "mba_ce": "MBA-CE",
    "mba_f": "MBA-f",
    "mba_ps": "MBA-PS",
}
DATASETS = ["cifar10", "cifar100"]
MODELS = ["resnet56", "vit"]


def load_history(path: Path) -> Optional[Dict[str, Any]]:
    """Load a history.json file, returning None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def extract_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """Extract final Top-1, ECE, and best Top-1 from a history dict."""
    test_top1 = history.get("test_top1", [])
    test_ece = history.get("test_ece", [])
    epochs = history.get("epoch", [])
    n = len(epochs) if epochs else 0
    final_top1 = float(test_top1[-1]) if test_top1 else float("nan")
    final_ece = float(test_ece[-1]) if test_ece else float("nan")
    best_top1 = float(max(test_top1)) if test_top1 else float("nan")
    return {
        "top1": final_top1,
        "ece": final_ece,
        "best_top1": best_top1,
        "epochs_completed": n,
    }


def scan_runs(runs_dir: Path, seed: int) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Scan runs directory and build nested results dict.

    Returns: {dataset: {model: {loss: {top1, ece, best_top1, epochs_completed}}}}
    """
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for loss in LOSS_ORDER:
                run_dir = runs_dir / f"{dataset}_{model}_{loss}_seed{seed}"
                history = load_history(run_dir / "history.json")
                if history is None:
                    results[dataset][model][loss] = {
                        "top1": float("nan"), "ece": float("nan"),
                        "best_top1": float("nan"), "epochs_completed": 0,
                    }
                else:
                    results[dataset][model][loss] = extract_metrics(history)
    return results


def print_tables(results: Dict) -> None:
    """Print 4 result tables to stdout."""
    for dataset in DATASETS:
        for model in MODELS:
            print(f"\n=== {dataset} + {model} ===")
            print(f"{'Loss':<15} {'Top-1 (%)':>10} {'ECE (%)':>10} {'Best (%)':>10} {'Epochs':>8}")
            print("-" * 55)
            for loss in LOSS_ORDER:
                m = results[dataset][model].get(loss, {})
                top1 = m.get("top1", float("nan"))
                ece = m.get("ece", float("nan"))
                best = m.get("best_top1", float("nan"))
                epochs = m.get("epochs_completed", 0)
                top1_s = f"{top1*100:.2f}" if not np.isnan(top1) else "N/A"
                ece_s = f"{ece*100:.2f}" if not np.isnan(ece) else "N/A"
                best_s = f"{best*100:.2f}" if not np.isnan(best) else "N/A"
                print(f"{LOSS_LABELS[loss]:<15} {top1_s:>10} {ece_s:>10} {best_s:>10} {epochs:>8}")


def write_csv(results: Dict, path: Path) -> None:
    """Write flat CSV of all results."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "loss", "top1", "ece", "best_top1", "epochs"])
        for dataset in DATASETS:
            for model in MODELS:
                for loss in LOSS_ORDER:
                    m = results[dataset][model].get(loss, {})
                    writer.writerow([
                        dataset, model, loss,
                        f"{m.get('top1', float('nan')):.6f}",
                        f"{m.get('ece', float('nan')):.6f}",
                        f"{m.get('best_top1', float('nan')):.6f}",
                        m.get("epochs_completed", 0),
                    ])
    print(f"Wrote {path}")


def write_summary_json(results: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {path}")


def plot_results(results: Dict, path: Path) -> None:
    """Generate a bar chart comparing Top-1 across all 32 experiments."""
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for loss in LOSS_ORDER:
                m = results[dataset][model].get(loss, {})
                top1 = m.get("top1", float("nan"))
                rows.append((dataset, model, loss, top1))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.5), 6))
    colors = {"ce": "#888888", "focal": "#666666", "poly": "#444444",
              "lace_multi": "#4488cc", "f_multi": "#88cc44",
              "mba_ce": "#cc4444", "mba_f": "#cc8844", "mba_ps": "#cc44cc"}
    labels = [f"{r[0]}\n{r[1]}\n{LOSS_LABELS[r[2]]}" for r in rows]
    vals = [r[3] * 100 if not np.isnan(r[3]) else 0 for r in rows]
    bar_colors = [colors.get(r[2], "#888888") for r in rows]
    bars = ax.bar(range(n), vals, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("MBA Loss Family: Top-1 Accuracy Comparison (100 epochs, seed 0, GPU)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to aggregate.")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    results = scan_runs(runs_dir, args.seed)
    print_tables(results)
    write_csv(results, runs_dir / "results_table.csv")
    write_summary_json(results, runs_dir / "summary.json")
    plot_results(results, runs_dir / "results_table.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
