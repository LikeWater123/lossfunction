"""Loss function registry for the MBA loss-family experiments.

Provides a unified ``build_loss(name, num_classes, **kwargs)`` factory that
instantiates any registered loss module by name, plus ``get_loss_names()``
listing the supported loss names.

This module is the SINGLE source of truth for loss construction: training and
evaluation scripts should never import loss classes directly but instead call
``build_loss`` with a registry name string.

Registered losses (8 total):
- ``ce``         : standard cross-entropy with optional label smoothing.
- ``focal``      : Lin et al., ICCV 2017 focal loss.
- ``poly``       : Leng et al., CVPR 2022 Poly-1 loss.
- ``lace_multi`` : multiplicative-modulation cross-entropy (Chapter 1).
- ``f_multi``    : multiplicative-modulation alpha-divergence loss (Chapter 2).
- ``mba_ce``     : MBA-CE loss (Chapter 6.5).
- ``mba_f``      : MBA-f loss (Chapter 6.6).
- ``mba_ps``     : MBA-PS loss (Chapter 6.7).
"""

import inspect
from typing import Callable, List

import torch.nn as nn

from .baselines import CrossEntropy, FocalLoss, PolyLoss
from .lace_variants import LACEMulti, FMulti
from .mba import MBACE, MBAF, MBAPS

# Type alias for a loss constructor: (num_classes, **kwargs) -> nn.Module.
LossConstructor = Callable[..., nn.Module]

# Dict-based dispatch table so the registry is trivial to extend.
# To add a new loss, register it here and append its name to the list
# returned by ``get_loss_names()``.
_LOSS_REGISTRY: dict[str, LossConstructor] = {
    "ce": CrossEntropy,
    "focal": FocalLoss,
    "poly": PolyLoss,
    "lace_multi": LACEMulti,
    "f_multi": FMulti,
    "mba_ce": MBACE,
    "mba_f": MBAF,
    "mba_ps": MBAPS,
}


def build_loss(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Build a loss function by name.

    Only kwargs that the loss constructor accepts are forwarded; unknown
    keys are silently dropped so a config block containing all loss kwargs
    (gamma, delta, alpha, epsilon, ...) can be passed to every loss without
    raising ``TypeError``.

    Args:
        name: Registry key, e.g. ``"ce"``, ``"focal"``, ``"mba_ce"``.
            Case-insensitive (the lookup key is lower-cased).
        num_classes: Number of target classes ``C``.
        **kwargs: Filtered to the loss constructor's accepted parameters.

    Returns:
        An instantiated ``nn.Module`` loss.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    name = name.lower()
    if name not in _LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Supported: {sorted(get_loss_names())}"
        )
    cls = _LOSS_REGISTRY[name]
    # Introspect __init__ to filter kwargs that aren't accepted.
    sig = inspect.signature(cls.__init__)
    accepted = {
        p_name
        for p_name, p in sig.parameters.items()
        if p_name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(num_classes=num_classes, **filtered)


def get_loss_names() -> List[str]:
    """Return the list of supported loss names (sorted, lowercase)."""
    return sorted(_LOSS_REGISTRY.keys())


def get_registry() -> dict[str, LossConstructor]:
    """Return a copy of the internal loss registry dict (for extension)."""
    return dict(_LOSS_REGISTRY)


def register_loss(name: str, constructor: LossConstructor) -> None:
    """Register an additional loss constructor under ``name``.

    Useful for extending the registry without mutating the dict directly.

    Args:
        name: Registry key (case-insensitive on lookup; stored lower-cased).
        constructor: Callable ``(num_classes, **kwargs) -> nn.Module``.
    """
    _LOSS_REGISTRY[name.lower()] = constructor


__all__ = [
    "CrossEntropy",
    "FocalLoss",
    "PolyLoss",
    "LACEMulti",
    "FMulti",
    "MBACE",
    "MBAF",
    "MBAPS",
    "build_loss",
    "get_loss_names",
    "get_registry",
    "register_loss",
]
