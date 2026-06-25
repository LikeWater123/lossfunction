"""Config-driven training pipeline for the MBA loss-family experiments.

CLI::

    python src/train.py --config <path> [--override key=value ...]

Loads a YAML config (optionally inheriting from a ``defaults:`` base), applies
``--override key=value`` overrides (dotted keys, YAML-parsed values), runs the
training loop, and writes ``history.json``, ``param_trajectory.json``,
``best.pt``, ``final.pt``, and plots into ``output_dir``. Uses CUDA when
available, otherwise falls back to CPU; CPU threads are capped to
``min(8, os.cpu_count())`` to avoid contention.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-train-cache")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

torch.set_num_threads(int(os.environ.get("MBA_NUM_THREADS", min(8, os.cpu_count() or 8))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.cifar import get_cifar_loaders  # noqa: E402
from src.methods import build_loss, get_loss_names  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.metrics import Accuracy, ECE  # noqa: E402
from src.utils.plot import plot_param_trajectory, plot_training_curves  # noqa: E402


def _resolve_path(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config, recursively merging any ``defaults:`` base.

    A ``defaults: <path>`` key causes the referenced file (resolved relative to
    the current config's directory) to be loaded first and the current config
    merged on top. ``loss_kwargs`` and ``model_kwargs`` are deep-merged; other
    keys are shallow-merged (current overrides base).
    """
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    defaults_ref = cfg.pop("defaults", None)
    if defaults_ref:
        base_dir = os.path.dirname(config_path)
        base = load_config(_resolve_path(base_dir, defaults_ref))
        merged: Dict[str, Any] = dict(base)
        for k in ("loss_kwargs", "model_kwargs"):
            if isinstance(base.get(k), dict) and isinstance(cfg.get(k), dict):
                merged_k = dict(base[k]); merged_k.update(cfg[k])
                merged[k] = merged_k
                cfg.pop(k, None)
        merged.update(cfg)
        cfg = merged
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``key=value`` overrides with dotted-path traversal. Values are
    parsed with ``yaml.safe_load`` so ``1``, ``1.0``, ``true`` coerce sensibly."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' missing '=' (expected key=value).")
        key, raw = ov.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        keys = key.strip().split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU


def build_optimizer(name: str, model: nn.Module, loss_params: List[nn.Parameter],
                    lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build optimizer with a separate (no-WD) param group for loss params."""
    groups: List[Dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay},
    ]
    if loss_params:
        groups.append({"params": loss_params, "lr": lr, "weight_decay": 0.0})
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'; expected sgd|adamw|adam.")


def build_scheduler(name: str, optimizer: torch.optim.Optimizer,
                    epochs: int) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5)
    if name in ("none", "constant", ""):
        return None
    raise ValueError(f"Unknown scheduler '{name}'; expected cosine|step|plateau.")


def _loss_accepts_step_ratio(loss_fn: nn.Module) -> bool:
    """True if ``loss_fn.forward`` accepts a ``step_ratio`` kwarg (MBAPS)."""
    try:
        sig = inspect.signature(loss_fn.forward)
    except (ValueError, TypeError):
        return False
    return "step_ratio" in sig.parameters


def _call_loss(loss_fn: nn.Module, logits: torch.Tensor, targets: torch.Tensor,
               step_ratio: float, accepts: bool) -> torch.Tensor:
    if accepts:
        return loss_fn(logits, targets, step_ratio=step_ratio)
    return loss_fn(logits, targets)


def _snapshot_loss_params(loss_fn: nn.Module, epoch: int) -> Dict[str, Tuple[int, Any]]:
    """Snapshot learnable / hyperparameter values at ``epoch`` for trajectory.

    Returns keys among ``eps``, ``gamma``, ``a``, ``b``, ``c``, ``alpha`` that
    are present. ``eps/a/b/c`` -> ``(epoch, np.ndarray)``; ``gamma/alpha`` ->
    ``(epoch, float)``. Learnable ``gamma`` records the effective (softplus) value.
    """
    snap: Dict[str, Tuple[int, Any]] = {}
    eps_attr = getattr(loss_fn, "eps", None)
    if isinstance(eps_attr, nn.Parameter):
        snap["eps"] = (epoch, eps_attr.detach().cpu().numpy().copy())
    gamma_attr = getattr(loss_fn, "gamma", None)
    if gamma_attr is not None:
        if isinstance(gamma_attr, nn.Parameter):
            val = float(F.softplus(gamma_attr.detach()).item())
        elif torch.is_tensor(gamma_attr):
            val = float(gamma_attr.item())
        else:
            val = float(gamma_attr)
        snap["gamma"] = (epoch, val)
    for n in ("a", "b", "c"):
        attr = getattr(loss_fn, n, None)
        if isinstance(attr, nn.Parameter):
            snap[n] = (epoch, attr.detach().cpu().numpy().copy())
    alpha_attr = getattr(loss_fn, "alpha", None)
    if isinstance(alpha_attr, (int, float)) and not isinstance(alpha_attr, bool):
        snap["alpha"] = (epoch, float(alpha_attr))
    return snap


def _init_trajectory(loss_fn: nn.Module) -> Dict[str, Optional[List[Tuple[int, Any]]]]:
    traj: Dict[str, Optional[List[Tuple[int, Any]]]] = {
        "eps": None, "gamma": None, "a": None, "b": None, "c": None, "alpha": None,
    }
    for k in _snapshot_loss_params(loss_fn, -1):
        traj[k] = []
    return traj


def _serialize_trajectory(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to lists for JSON serialisation."""
    out: Dict[str, Any] = {}
    for k, v in traj.items():
        if v is None:
            out[k] = None
        else:
            out[k] = [(int(e), a.tolist() if hasattr(a, "tolist") else float(a))
                      for e, a in v]
    return out


def _save_checkpoint(path: str, model: nn.Module, loss_fn: nn.Module,
                     optimizer: torch.optim.Optimizer, epoch: int,
                     config: Dict[str, Any], metrics: Dict[str, float]) -> None:
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "loss_state_dict": loss_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full training loop described by ``config``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config.get("seed", 0)))

    dataset, model_name, loss_name = config["dataset"], config["model"], config["loss"]
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])
    optimizer_name = config["optimizer"]
    scheduler_name = config.get("scheduler", "cosine")
    eval_interval = int(config.get("eval_interval", 1))
    log_interval = int(config.get("log_interval", 50))
    save_checkpoint = bool(config.get("save_checkpoint", True))
    checkpoint_best_only = bool(config.get("checkpoint_best_only", True))

    output_dir = config.get("output_dir") or (
        f"runs/{dataset}_{model_name}_{loss_name}_seed{config.get('seed', 0)}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[train] dataset={dataset} model={model_name} loss={loss_name} "
          f"epochs={epochs} lr={lr} wd={weight_decay} optim={optimizer_name} "
          f"sched={scheduler_name} output_dir={output_dir} device={device} "
          f"threads={torch.get_num_threads()}")

    train_loader, test_loader, num_classes = get_cifar_loaders(
        name=dataset, batch_size=batch_size, root=config.get("data_root", "./data"),
        num_workers=int(config.get("num_workers", 4)), augment=True,
        distributed=False, cutout=bool(config.get("cutout", False)),
    )

    model = build_model(model_name, num_classes=num_classes).to(device)
    print(f"[train] model params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    loss_kwargs = dict(config.get("loss_kwargs", {}) or {})
    if loss_name == "mba_ps":
        loss_kwargs.setdefault("num_epochs", epochs)
    loss_fn = build_loss(loss_name, num_classes=num_classes, **loss_kwargs).to(device)
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = build_optimizer(optimizer_name, model, loss_params, lr, weight_decay)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs)
    accepts = _loss_accepts_step_ratio(loss_fn)
    print(f"[train] loss kwargs={loss_kwargs} params={len(loss_params)} "
          f"accepts_step_ratio={accepts}")

    history: Dict[str, List[Any]] = {"epoch": [], "train_loss": [], "train_top1": [],
        "test_loss": [], "test_top1": [], "test_ece": [], "lr": []}
    trajectory = _init_trajectory(loss_fn)
    best_top1, best_epoch = -1.0, -1

    for epoch in range(epochs):
        step_ratio = (epoch + 1) / epochs
        model.train()
        run_acc = Accuracy()
        run_loss_sum, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = _call_loss(loss_fn, logits, targets, step_ratio, accepts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = int(targets.numel())
            run_loss_sum += float(loss.item()) * bs
            n_seen += bs
            run_acc.update(logits, targets)
            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{run_loss_sum/max(n_seen,1):.4f}",
                    "acc": f"{run_acc.compute()*100:.2f}",
                })
        pbar.close()
        train_loss = run_loss_sum / max(n_seen, 1)
        train_top1 = run_acc.compute()

        # Single-pass evaluation: test_loss + top1 + ECE.
        test_top1 = test_ece = test_loss = 0.0
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            model.eval()
            t_acc, t_ece = Accuracy(), ECE(num_bins=15)
            t_loss_sum, t_n = 0.0, 0
            with torch.no_grad():
                for batch in test_loader:
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    logits = model(images)
                    loss = _call_loss(loss_fn, logits, targets, step_ratio, accepts)
                    bs = int(targets.numel())
                    t_loss_sum += float(loss.item()) * bs
                    t_n += bs
                    t_acc.update(logits, targets)
                    t_ece.update(logits, targets)
            test_top1, test_ece = t_acc.compute(), t_ece.compute()
            test_loss = t_loss_sum / max(t_n, 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        for k, v in [("epoch", epoch), ("train_loss", train_loss),
                     ("train_top1", train_top1), ("test_loss", test_loss),
                     ("test_top1", test_top1), ("test_ece", test_ece), ("lr", cur_lr)]:
            history[k].append(v)

        for k, v in _snapshot_loss_params(loss_fn, epoch).items():
            if trajectory[k] is None:
                trajectory[k] = []
            trajectory[k].append(v)

        print(f"[epoch {epoch+1:3d}/{epochs}] train_loss={train_loss:.4f} "
              f"train_top1={train_top1*100:.2f} test_top1={test_top1*100:.2f} "
              f"test_ece={test_ece:.4f} lr={cur_lr:.5g}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_top1)
            else:
                scheduler.step()

        if save_checkpoint:
            if test_top1 > best_top1:
                best_top1, best_epoch = test_top1, epoch
                _save_checkpoint(os.path.join(output_dir, "best.pt"),
                                 model, loss_fn, optimizer, epoch, config,
                                 {"top1": test_top1, "ece": test_ece})
            elif not checkpoint_best_only:
                _save_checkpoint(os.path.join(output_dir, f"epoch_{epoch}.pt"),
                                 model, loss_fn, optimizer, epoch, config,
                                 {"top1": test_top1, "ece": test_ece})

    if save_checkpoint:
        _save_checkpoint(os.path.join(output_dir, "final.pt"),
                         model, loss_fn, optimizer, epochs - 1, config,
                         {"top1": test_top1, "ece": test_ece})

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "param_trajectory.json"), "w") as f:
        json.dump(_serialize_trajectory(trajectory), f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    try:
        plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))
        plot_param_trajectory(dict(trajectory),
                              os.path.join(output_dir, "param_trajectory.png"))
    except Exception as e:  # plotting is best-effort
        print(f"[train] plotting failed: {e}")

    print(f"\n[train] done. best_top1={best_top1*100:.2f} @ epoch {best_epoch+1} "
          f"final_top1={test_top1*100:.2f} final_ece={test_ece:.4f}\n"
          f"  output_dir={output_dir}")
    return {"best_top1": best_top1, "best_epoch": best_epoch,
            "final_top1": test_top1, "final_ece": test_ece,
            "output_dir": output_dir}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a model with the MBA loss-family experiments pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (may use `defaults:` for inheritance).")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Dotted-path key=value overrides (parsed as YAML).")
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), list(args.override))

    if not config.get("output_dir"):
        config["output_dir"] = (
            f"runs/{config['dataset']}_{config['model']}_{config['loss']}_"
            f"seed{config.get('seed', 0)}")

    valid_losses = set(get_loss_names())
    if config["loss"].lower() not in valid_losses:
        raise ValueError(
            f"Unknown loss '{config['loss']}'. Supported: {sorted(valid_losses)}")

    print(json.dumps(train(config), indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
