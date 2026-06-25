"""Evaluation entry point for the MBA loss-family experiments.

CLI::

    python src/evaluate.py --config <path> --checkpoint <path> [--output <path>]

Loads a trained checkpoint, rebuilds the model and test dataset, computes
Top-1 accuracy, ECE, and the confusion matrix on the test set, prints a
summary, and (if ``--output`` is given) writes the metrics to a JSON file.
Uses CUDA when available, otherwise falls back to CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-train-cache")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 8))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_cifar_loaders  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def evaluate(config: Dict[str, Any], checkpoint_path: str,
             output_path: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild the model, load the checkpoint, and evaluate on the test set.

    Args:
        config: The training config dict (used to rebuild model + dataset).
        checkpoint_path: Path to a ``.pt`` checkpoint saved by the trainer.
        output_path: If given, write the metrics dict here as JSON.

    Returns:
        Dict with ``top1``, ``ece``, ``confusion_matrix`` (as nested lists),
        ``epoch`` (the checkpoint epoch), and the resolved ``config``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config["dataset"]
    model_name = config["model"]
    batch_size = int(config.get("batch_size", 128))

    _, test_loader, num_classes = get_cifar_loaders(
        name=dataset, batch_size=batch_size,
        root=config.get("data_root", "./data"),
        num_workers=int(config.get("num_workers", 4)),
        augment=False, distributed=False, cutout=False,
    )

    model = build_model(model_name, num_classes=num_classes).to(device)

    print(f"[eval] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    epoch = int(ckpt.get("epoch", -1))
    print(f"[eval] checkpoint epoch={epoch} "
          f"stored metrics={ckpt.get('metrics', {})}")

    metrics = compute_metrics(model, test_loader, device, num_classes)
    confusion = metrics["confusion_matrix"]
    result: Dict[str, Any] = {
        "epoch": epoch,
        "top1": float(metrics["top1"]),
        "ece": float(metrics["ece"]),
        "confusion_matrix": confusion.tolist(),
        "checkpoint": os.path.abspath(checkpoint_path),
        "config": config,
    }

    print(f"[eval] top1={result['top1']*100:.2f}% ece={result['ece']:.4f} "
          f"confusion shape={confusion.shape}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] wrote {output_path}")
    return result


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the CIFAR test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config used for training.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint to evaluate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write metrics JSON.")
    args = parser.parse_args(argv)

    # Reuse the trainer's config loader (handles `defaults:` inheritance).
    from src.train import load_config
    config = load_config(args.config)

    result = evaluate(config, args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("confusion_matrix", "config")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
