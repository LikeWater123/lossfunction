"""Evaluation metrics: Top-1 accuracy, Expected Calibration Error, confusion matrix.

All accumulators accept ``(logits, targets)`` batches with ``logits`` of shape
``(B, C)`` and ``targets`` of shape ``(B,)`` long tensor, matching the loss
module convention. CPU-only: no device transfers are made here.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Accuracy", "ECE", "compute_metrics"]


class Accuracy:
    """Top-1 accuracy accumulator. ``update(logits, targets)``; ``compute()`` -> float."""

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=-1)
        self.correct += int((preds == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self) -> float:
        return self.correct / max(self.total, 1)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class ECE:
    """Expected Calibration Error with ``num_bins`` equal-width bins.

    Bins samples by their max softmax probability and computes
    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|``. Standard 15-bin setting.
    """

    def __init__(self, num_bins: int = 15) -> None:
        self.num_bins = int(num_bins)
        self.bin_correct = np.zeros(self.num_bins, dtype=np.float64)
        self.bin_conf = np.zeros(self.num_bins, dtype=np.float64)
        self.bin_count = np.zeros(self.num_bins, dtype=np.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = F.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        correct = (preds == targets).to(probs.dtype)
        # Bin index in [0, num_bins-1]; right edge maps to last bin.
        bin_idx = (conf * self.num_bins).long().clamp(0, self.num_bins - 1)
        bi = bin_idx.detach().cpu().numpy()
        np.add.at(self.bin_correct, bi, correct.detach().cpu().numpy())
        np.add.at(self.bin_conf, bi, conf.detach().cpu().numpy())
        np.add.at(self.bin_count, bi, 1.0)

    def compute(self) -> float:
        total = self.bin_count.sum()
        if total == 0:
            return 0.0
        safe = np.maximum(self.bin_count, 1.0)
        acc = self.bin_correct / safe
        avg_conf = self.bin_conf / safe
        return float(np.sum((self.bin_count / total) * np.abs(acc - avg_conf)))

    def reset(self) -> None:
        self.bin_correct[:] = 0.0
        self.bin_conf[:] = 0.0
        self.bin_count[:] = 0.0

def compute_metrics(model, loader, device, num_classes: int) -> Dict[str, object]:
    """Compute Top-1 accuracy, ECE, and confusion matrix in one pass.

    Iterates ``loader`` once under ``torch.no_grad()``. Returns a dict with
    ``'top1'`` (float), ``'ece'`` (float), and ``'confusion_matrix'``
    (``np.ndarray`` of shape ``(C, C)`` int64).
    """
    model.eval()
    acc = Accuracy()
    ece = ECE(num_bins=15)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            acc.update(logits, targets)
            ece.update(logits, targets)
            preds = logits.argmax(dim=-1)
            t_np = targets.detach().cpu().numpy().reshape(-1)
            p_np = preds.detach().cpu().numpy().reshape(-1)
            np.add.at(confusion, (t_np, p_np), 1)
    return {"top1": acc.compute(), "ece": ece.compute(), "confusion_matrix": confusion}
