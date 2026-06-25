"""CNN model registry: ResNet family for CIFAR."""

from __future__ import annotations

from .resnet import (
    BasicBlock,
    ResNetCIFAR,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
)

__all__ = [
    "BasicBlock",
    "ResNetCIFAR",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
]
