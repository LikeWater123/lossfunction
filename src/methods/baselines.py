"""Baseline loss functions for CIFAR-10/100 image classification experiments.

Provides three standard losses used as reference points against the proposed
MBA (Monotone Bounded Amplification) loss family:

- ``CrossEntropy`` (registry name ``ce``): standard PyTorch cross-entropy with
  optional label smoothing.
- ``FocalLoss`` (registry name ``focal``): Lin et al., ICCV 2017 focal loss
  with per-class weighting ``alpha`` and focusing parameter ``gamma``.
- ``PolyLoss`` (registry name ``poly``): Leng et al., CVPR 2022 Poly-1 loss,
  ``-log(p_t) + epsilon * (1 - p_t)`` with default ``epsilon=2.0``.

All losses accept ``(logits, targets)`` where ``logits`` is ``(B, C)`` and
``targets`` is a ``(B,)`` long tensor, and return a scalar tensor (mean over
the batch). Computations use ``torch.nn.functional`` (``F.log_softmax`` etc.)
for numerical stability. No GPU-specific calls are made.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    """Standard cross-entropy loss with optional label smoothing.

    Thin wrapper around ``F.cross_entropy`` with ``reduction='mean'``.
    No learnable parameters.

    Args:
        num_classes: Number of target classes ``C``.
        label_smoothing: Label smoothing factor in ``[0, 1)`` (default ``0.0``).
    """

    def __init__(self, num_classes: int, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., ICCV 2017).

    ``L = -alpha_t * (1 - p_t)^gamma * log(p_t)`` where ``p_t`` is the softmax
    probability of the target class. Implementations use ``F.log_softmax`` for
    numerical stability.

    Args:
        num_classes: Number of target classes ``C``.
        gamma: Focusing parameter down-weights easy examples (default ``2.0``).
        alpha: Either ``None`` for uniform per-class weight ``1.0``, a scalar
            (Python float or 0-D / 1-element tensor) broadcast to a uniform
            per-class weight of that value, or a 1-D tensor of length
            ``num_classes`` giving per-class weights (default ``None``).

    Note: the trainer's shared ``loss_kwargs.alpha=0.5`` (intended for f-Multi's
    f-divergence parameter) is accepted by FocalLoss and broadcast to a uniform
    per-class weight of ``0.5``. The resulting loss is scaled by ``0.5`` vs the
    standard Focal loss, which is consistent across all focal runs in this sweep.
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, list, tuple, float]] = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = num_classes
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", torch.ones(num_classes))
        else:
            if not torch.is_tensor(alpha):
                alpha = torch.as_tensor(alpha, dtype=torch.float)
            alpha = alpha.to(dtype=torch.float)
            if alpha.numel() == 1:
                alpha = torch.full((num_classes,), float(alpha.flatten()[0].item()))
            elif alpha.numel() != num_classes:
                raise ValueError(
                    f"alpha must be scalar, length {num_classes}, or None; "
                    f"got numel={alpha.numel()}"
                )
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)  # (B, C)
        log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        p_t = log_p_t.exp()
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha.gather(0, targets)  # (B,)
        loss = -alpha_t * focal_weight * log_p_t
        return loss.mean()


class PolyLoss(nn.Module):
    """Poly-1 loss (Leng et al., CVPR 2022).

    Poly-1 approximation of the cross-entropy:

        L_Poly = -log(p_t) + epsilon * (1 - p_t)

    where ``p_t`` is the softmax probability of the target class and
    ``epsilon`` is the leading coefficient of the Taylor expansion of
    ``-log(1 - x)``. The paper's default ``epsilon=2.0`` recovers the
    first-order Taylor term.

    Args:
        num_classes: Number of target classes ``C``.
        epsilon: Poly-1 hyperparameter (default ``2.0``).
    """

    def __init__(self, num_classes: int, epsilon: float = 2.0) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)  # (B, C)
        log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        p_t = log_p_t.exp()
        loss = -log_p_t + self.epsilon * (1.0 - p_t)
        return loss.mean()


__all__ = ["CrossEntropy", "FocalLoss", "PolyLoss"]
