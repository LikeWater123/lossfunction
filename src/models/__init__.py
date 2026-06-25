"""Unified model registry.

Provides ``build_model(name, num_classes)`` to construct any registered
backbone by name. Currently registered models:

    - 'resnet56'  -> ResNet-56 for CIFAR (primary CNN)
    - 'resnet20'  -> ResNet-20 for CIFAR (ablation)
    - 'resnet44'  -> ResNet-44 for CIFAR (ablation)
    - 'resnet110' -> ResNet-110 for CIFAR (ablation)
    - 'vit'       -> small Vision Transformer for CIFAR (primary ViT)

All registered models accept ``(B, 3, 32, 32)`` input and return
``(B, num_classes)`` logits.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .cnn.resnet import (
    ResNetCIFAR,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
)
from .vit.vit import VisionTransformer, vit_small_cifar

ModelFactory = Callable[..., nn.Module]

MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "resnet56": resnet56,
    "resnet44": resnet44,
    "resnet32": resnet32,
    "resnet20": resnet20,
    "resnet110": resnet110,
    "vit": vit_small_cifar,
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Construct a model by registry name.

    Args:
        name: One of the keys in ``MODEL_REGISTRY`` (e.g. 'resnet56', 'vit').
        num_classes: Number of output logits (e.g. 10 for CIFAR-10,
            100 for CIFAR-100).

    Returns:
        An ``nn.Module`` mapping ``(B, 3, 32, 32)`` inputs to
        ``(B, num_classes)`` logits.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model name '{name}'. Available: {available}."
        )
    return MODEL_REGISTRY[name](num_classes=num_classes)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "list_models",
    "ResNetCIFAR",
    "VisionTransformer",
    "resnet56",
    "resnet44",
    "resnet32",
    "resnet20",
    "resnet110",
    "vit_small_cifar",
]
