"""ResNet for CIFAR (He et al. 2016, CIFAR variant).

Implements the "ResNet for CIFAR" family (3x3 stem, no maxpool, no 7x7 conv)
with the standard 6n+2 depth convention: depth = 6n + 2, n = (depth - 2) / 6.
For ResNet-56, n = 9.

Each of the 3 stages contains ``n`` BasicBlocks. Stage widths are 16, 32, 64.
Downsampling (stride-2) is performed by the first block of stages 2 and 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding=1, bias=False."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock (two 3x3 convs).

    Expansion factor = 1, so inplanes == planes for the residual path.
    The shortcut uses a 1x1 conv with stride when spatial dims change.
    """

    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet for CIFAR 32x32 inputs (He et al. 2016 CIFAR variant)."""

    def __init__(self, num_blocks: int, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes

        # Stem: 3x3 conv, BN, ReLU (no maxpool, no 7x7).
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 3 stages with widths 16, 32, 64.
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)

        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """He initialization for conv layers; linear -> trunc_normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def _resnet(depth: int, num_classes: int = 10) -> ResNetCIFAR:
    if (depth - 2) % 6 != 0:
        raise ValueError(f"ResNet depth must satisfy 6n+2; got {depth}.")
    n = (depth - 2) // 6
    return ResNetCIFAR(num_blocks=n, num_classes=num_classes)


def resnet56(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-56 for CIFAR (primary model)."""
    return _resnet(56, num_classes)


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 for CIFAR (ablation)."""
    return _resnet(20, num_classes)


def resnet110(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-110 for CIFAR (ablation)."""
    return _resnet(110, num_classes)


def resnet44(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-44 for CIFAR (ablation)."""
    return _resnet(44, num_classes)


def resnet32(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-32 for CIFAR (ablation)."""
    return _resnet(32, num_classes)
