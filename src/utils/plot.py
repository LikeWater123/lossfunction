"""Plotting helpers: training curves, learnable-param trajectories, results tables.

Uses the matplotlib ``Agg`` backend so figures render headless on CPU servers.
All functions save to ``output_path`` with ``dpi=120, bbox_inches='tight'`` and
close the figure afterwards to free memory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_training_curves", "plot_param_trajectory", "plot_results_table"]


def _safe_epochs(history: Dict[str, Any], key: str) -> List[int]:
    """Return the x-axis (epochs) for a history series, defaulting to range."""
    n = len(history.get(key, []))
    if "epoch" in history:
        return list(history["epoch"][:n])
    return list(range(n))


def plot_training_curves(history: Dict[str, Any], output_path: str) -> None:
    """Plot train/test loss, top-1 accuracy, ECE, and lr vs epoch (4 panels).

    ``history`` is the dict persisted as ``history.json`` by the trainer. Any
    missing series is simply skipped on its panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _panel(ax, series_pairs, title, ylabel):
        for key, label in series_pairs:
            if key in history:
                ax.plot(_safe_epochs(history, key), history[key], label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
        ax.grid(True, alpha=0.3)

    _panel(axes[0, 0], [("train_loss", "train"), ("test_loss", "test")], "Loss", "Loss")
    _panel(axes[0, 1], [("train_top1", "train"), ("test_top1", "test")],
           "Top-1 Accuracy", "Accuracy")
    _panel(axes[1, 0], [("train_ece", "train"), ("test_ece", "test")], "ECE", "ECE")

    # LR panel (single series).
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(_safe_epochs(history, "lr"), history["lr"], label="lr", color="tab:green")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _split_pairs(seq: Sequence[Tuple[int, Any]]) -> Tuple[List[int], List[Any]]:
    return [int(e) for e, _ in seq], [v for _, v in seq]


def plot_param_trajectory(traj: Dict[str, Any], output_path: str) -> None:
    """Plot learnable-parameter trajectories vs epoch.

    ``traj`` keys: ``'eps'``, ``'gamma'``, ``'a'``, ``'b'``, ``'c'``, ``'alpha'``.
    Each value is either ``None`` (parameter absent for this loss) or a list of
    ``(epoch, value)`` tuples, where ``value`` is an ``np.ndarray`` of shape
    ``(C,)`` for per-class params (eps/a/b/c) or a float for scalar params
    (gamma/alpha).

    Generates up to 3 panels:
        1. eps: mean over classes with min/max shaded region.
        2. gamma + alpha: scalar trajectories.
        3. a/b/c: per-class means (MBA-PS only).
    Panels for absent parameters are skipped.
    """
    eps_data = traj.get("eps")
    scalar_pairs: List[Tuple[str, List[Tuple[int, float]]]] = []
    for name in ("gamma", "alpha"):
        d = traj.get(name)
        if d:
            scalar_pairs.append((name, d))
    abc_pairs: List[Tuple[str, List[Tuple[int, np.ndarray]]]] = []
    for name in ("a", "b", "c"):
        d = traj.get(name)
        if d:
            abc_pairs.append((name, d))

    panels: List[str] = []
    if eps_data:
        panels.append("eps")
    if scalar_pairs:
        panels.append("scalar")
    if abc_pairs:
        panels.append("abc")

    if not panels:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No learnable parameters tracked",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(panels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, panel in enumerate(panels):
        ax = axes[i // cols][i % cols]
        if panel == "eps":
            epochs, arrs = _split_pairs(eps_data)
            mat = np.stack(arrs)  # (T, C)
            mean = mat.mean(axis=1)
            lo = mat.min(axis=1)
            hi = mat.max(axis=1)
            ax.plot(epochs, mean, label="mean")
            ax.fill_between(epochs, lo, hi, alpha=0.25, label="min/max")
            ax.set_title("eps (per-class)")
        elif panel == "scalar":
            for name, d in scalar_pairs:
                epochs, vals = _split_pairs(d)
                ax.plot(epochs, vals, label=name, marker="o", markersize=3)
            ax.set_title("Scalar parameters")
        elif panel == "abc":
            for name, d in abc_pairs:
                epochs, arrs = _split_pairs(d)
                mat = np.stack(arrs)
                ax.plot(epochs, mat.mean(axis=1), label=f"{name} mean", marker="o", markersize=3)
            ax.set_title("MBA-PS a/b/c (mean over classes)")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes.
    for j in range(len(panels), rows * cols):
        axes[j // cols][j % cols].set_axis_off()

    fig.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_results_table(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                      output_path: str) -> None:
    """Given nested results dict, produce a bar chart + table figure.

    Input format: ``{'cifar10': {'resnet56': {'ce': {'top1': 0.92, 'ece': 0.05}, ...}}}``.
    Saves a figure with a Top-1 bar chart and a rendered results table.
    """
    rows: List[Tuple[str, str, str, float, float]] = []
    for dataset in sorted(results.keys()):
        for model in sorted(results[dataset].keys()):
            for loss in sorted(results[dataset][model].keys()):
                m = results[dataset][model][loss]
                rows.append((dataset, model, loss,
                             float(m.get("top1", float("nan"))),
                             float(m.get("ece", float("nan")))))
    n = len(rows)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.5), 8),
                             gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    labels = [f"{r[0]}\n{r[1]}\n{r[2]}" for r in rows]
    top1_vals = [r[3] for r in rows]
    bars = ax.bar(range(n), top1_vals, color="tab:blue")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Top-1 Accuracy"); ax.set_title("Top-1 Accuracy Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    if top1_vals:
        valid = [v for v in top1_vals if not np.isnan(v)]
        ymax = max(valid) if valid else 1.0
        ax.set_ylim(0, max(1.0, ymax * 1.05))
        for bar, v in zip(bars, top1_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1]; ax.axis("off")
    cell_text = [[r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}"] for r in rows]
    table = ax.table(cellText=cell_text,
                     colLabels=["Dataset", "Model", "Loss", "Top-1", "ECE"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
