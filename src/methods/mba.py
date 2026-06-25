"""MBA (Monotone Bounded Amplification) loss family.

Implements the three MBA loss functions defined in Chapter 6 of the LACE
improvement analysis document (``documents/LACE改进方案深度理论分析.md``,
lines 789-1020). These are the core contribution of the project: a unified
multiplicative amplification framework that fixes the gradient non-monotonicity,
loss divergence under noise, and f-Multi gradient-misalignment issues of the
original LACE-Multi / f-Multi / training-state-coupling proposals.

The MBA framework has two shared components (Section 6.4):

1. **Rational gate** (replaces the ``(1 - P_t)`` gate of LACE-Multi)::

       phi_gamma(P_t) = (1 - P_t) / (1 + gamma * P_t),   gamma >= 0

   Properties: ``phi(0) = 1``, ``phi(1) = 0``, strictly decreasing, bounded in
   ``[0, 1]``. ``gamma = 0`` recovers ``(1 - P_t)`` (the LACE-Multi gate).

2. **Tempered (truncated) loss** (replaces the unbounded ``-ln P_t``)::

       tau_delta(P_t) = -ln max(P_t, delta),   delta in (0, 1) small.

   Properties: bounded above by ``-ln delta``; gradient truncated to zero for
   ``P_t < delta`` (noise robustness); ``delta -> 0`` recovers ``-ln P_t``.

**Unified form**::

    L_MBA = [1 + sigma(eps_y) * Lambda(P_t, s(t)) * phi_gamma(P_t)] * tau_delta(P_t)

Three concrete members (Sections 6.5-6.7):

- :class:`MBACE`  (registry: ``mba_ce``):  ``Lambda = 1``.
- :class:`MBAF`   (registry: ``mba_f``):   ``Lambda = 1`` but ``tau_delta`` is
  replaced by the alpha-divergence loss ``L_alpha`` and ``P_t`` by the
  f-softargmax output ``P_t^alpha``.
- :class:`MBAPS`  (registry: ``mba_ps``):  ``Lambda = lambda_y(s(t))`` is a
  per-class proactive/reactive schedule.

All losses accept ``(logits, targets)`` where ``logits`` is ``(B, C)`` and
``targets`` is a ``(B,)`` long tensor, and return a scalar tensor (mean over
the batch). Computations use ``torch.nn.functional`` for numerical stability.
No GPU-specific calls are made.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gamma_value(gamma: torch.Tensor) -> torch.Tensor:
    """Return a non-negative ``gamma``.

    If ``gamma`` is a learnable ``nn.Parameter`` we apply ``softplus`` to
    enforce the constraint ``gamma >= 0`` (per spec). If it is a registered
    buffer the user has already validated non-negativity at construction
    time, so the raw tensor is returned.
    """
    if isinstance(gamma, nn.Parameter):
        return F.softplus(gamma)
    return gamma


def _rational_gate(pt: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Rational gate ``phi_gamma(P_t) = (1 - P_t) / (1 + gamma * P_t)``.

    Properties: ``phi(0) = 1``, ``phi(1) = 0``, strictly decreasing, bounded
    in ``[0, 1]`` for ``gamma >= 0`` and ``P_t in [0, 1]``. ``gamma = 0``
    recovers the LACE-Multi gate ``(1 - P_t)``.
    """
    return (1.0 - pt) / (1.0 + gamma * pt)


def _tempered_loss(pt: torch.Tensor, delta: float) -> torch.Tensor:
    """Tempered (truncated) NLL ``tau_delta(P_t) = -ln max(P_t, delta)``.

    Properties: bounded above by ``-ln delta``; gradient is zero for
    ``P_t < delta`` (noise robustness, Theorem 6.6 (iii)); ``delta -> 0``
    recovers ``-ln P_t``.
    """
    pt_clamped = pt.clamp(min=delta)
    return -torch.log(pt_clamped)


# ---------------------------------------------------------------------------
# MBACE
# ---------------------------------------------------------------------------

class MBACE(nn.Module):
    """MBA-CE loss: rational gate + tempered cross-entropy (Section 6.5).

    Definition::

        L_MBACE = [1 + sigma(eps_y) * phi_gamma(P_t)] * tau_delta(P_t)

    where ``P_t = softmax(logits)[target]``.

    Degenerate relationships:
        - ``gamma = 0`` and ``delta -> 0`` recovers LACE-Multi.
        - ``sigma(eps_y) -> 0`` recovers ``tau_delta`` (bounded CE).

    Args:
        num_classes: Number of target classes ``C``.
        gamma: Rational-gate steepness (default ``1.0``). Must be ``>= 0``.
        delta: Tempering floor in ``(0, 1)`` (default ``1e-3``).
        gamma_learnable: If ``True``, ``gamma`` becomes an ``nn.Parameter``
            with ``softplus`` enforcing non-negativity (default ``False``).
        eps_init: Initial value of the per-class ``eps_y`` parameter
            (default ``0.0`` so ``sigma(eps_y) = 0.5`` initially).
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 1.0,
        delta: float = 1e-3,
        gamma_learnable: bool = False,
        eps_init: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.num_classes = num_classes
        self.delta = float(delta)
        if gamma_learnable:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.eps = nn.Parameter(torch.full((num_classes,), float(eps_init)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)                   # (B, C)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        pt = log_pt.exp()                                            # (B,)
        gamma = _gamma_value(self.gamma)
        phi = _rational_gate(pt, gamma)
        tau = _tempered_loss(pt, self.delta)
        sigma_eps = torch.sigmoid(self.eps[targets])                 # (B,)
        loss = (1.0 + sigma_eps * phi) * tau
        return loss.mean()


# ---------------------------------------------------------------------------
# MBAF
# ---------------------------------------------------------------------------

class MBAF(nn.Module):
    """MBA-f loss: rational gate + alpha-divergence (Section 6.6).

    Definition::

        L_MBAF = [1 + sigma(eps_y) * phi_gamma(P_t^alpha)] * L_alpha(z, y)

    where ``P_t^alpha`` is the f-softargmax output at the target and
    ``L_alpha`` is the alpha-divergence loss of Roulet et al. (ICML 2025).

    Degenerate relationship: ``alpha = 0`` gives ``P_t^alpha -> P_t`` and
    ``L_alpha -> -ln P_t`` (CE), so MBAF reduces to MBACE (without the
    ``delta`` clamp on the inner loss, since ``tau_delta`` is replaced by
    ``L_alpha``).

    Args:
        num_classes: Number of target classes ``C``.
        alpha: Alpha-divergence parameter (default ``0.5``). Typical
            values ``{0.0, 0.5, 1.5}``; ``alpha = 0`` recovers MBACE.
        gamma: Rational-gate steepness (default ``1.0``). Must be ``>= 0``.
        gamma_learnable: If ``True``, ``gamma`` becomes an ``nn.Parameter``
            (default ``False``).
        eps_init: Initial value of the per-class ``eps_y`` parameter
            (default ``0.0``).

    Note:
        The closed-form f-softargmax used here, ``p* = softmax((1 - alpha) * z)``,
        is a simplification of the general Roulet et al. f-softargmax (which for
        ``alpha > 1`` produces sparse distributions via a bisection algorithm and
        is not expressible as a tempered softmax). The loss ``L_alpha`` is taken
        as the negative log-likelihood ``-log p*_y``, which is the Fenchel-Young
        loss associated with the f-softargmax output and which degenerates
        exactly to CE at ``alpha = 0``. The alpha-divergence formula written in
        section 2 of the analysis doc, ``L_alpha = (1/(alpha*(1-alpha))) *
        [1 - sum_j pi_j * (p_j/pi_j)^(1-alpha)]``, is the divergence between
        the model distribution and the uniform reference and does not depend on
        the target ``y``; it cannot therefore be used directly as a
        classification loss. We use the NLL form, which is the proper
        target-dependent Fenchel-Young loss and satisfies the spec's stated
        degenerate relationship ``alpha = 0 -> CE``.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        gamma: float = 1.0,
        gamma_learnable: bool = False,
        eps_init: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.num_classes = num_classes
        self.alpha = float(alpha)
        if gamma_learnable:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.eps = nn.Parameter(torch.full((num_classes,), float(eps_init)))

    @staticmethod
    def f_softmax(logits: torch.Tensor, alpha: float) -> torch.Tensor:
        """f-softargmax output for the alpha-divergence with uniform reference.

        Returns ``softmax((1 - alpha) * logits)``. For ``alpha = 0`` this is
        the standard ``softmax(logits)``; for ``alpha = 1`` it is the uniform
        distribution ``softmax(0) = 1/C``.
        """
        scale = 1.0 - float(alpha)
        return F.softmax(scale * logits, dim=-1)

    @staticmethod
    def alpha_divergence_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        num_classes: int,
    ) -> torch.Tensor:
        """Alpha-divergence loss (Roulet et al., ICML 2025).

        Computed as the negative log-likelihood of the f-softargmax output::

            L_alpha = -log softmax((1 - alpha) * z)_y

        which degenerates to ``-ln softmax(z)_y = CE`` at ``alpha = 0``.
        See the class docstring for the rationale and for the relationship to
        the divergence formula in the analysis document.
        """
        scale = 1.0 - float(alpha)
        log_pt_alpha = F.log_softmax(scale * logits, dim=-1).gather(
            1, targets.view(-1, 1)
        ).squeeze(1)
        return -log_pt_alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scale = 1.0 - self.alpha
        log_pt_alpha = F.log_softmax(scale * logits, dim=-1).gather(
            1, targets.view(-1, 1)
        ).squeeze(1)                                                 # (B,)
        pt_alpha = log_pt_alpha.exp()                               # (B,)
        L_alpha = -log_pt_alpha                                     # (B,)
        gamma = _gamma_value(self.gamma)
        phi = _rational_gate(pt_alpha, gamma)
        sigma_eps = torch.sigmoid(self.eps[targets])               # (B,)
        loss = (1.0 + sigma_eps * phi) * L_alpha
        return loss.mean()


# ---------------------------------------------------------------------------
# MBAPS
# ---------------------------------------------------------------------------

class MBAPS(nn.Module):
    """MBA-PS loss: rational gate + tempered CE + proactive/reactive schedule.

    Definition (Section 6.7)::

        L_MBAPS = [1 + sigma(eps_y) * lambda_y(s(t)) * phi_gamma(P_t)] * tau_delta(P_t)

    where the amplification modulator is::

        lambda_y(s(t)) = sigma(a_y * rho(t) + b_y * s_react(t) + c_y)
        rho(t)         = 0.5 * (1 + cos(pi * step_ratio))    # active cosine
        s_react(t)     = Var_i(P_{t,i}) = mean_i((P_t - mean)^2)   # batch conf

    The trainer provides ``step_ratio = t / T in [0, 1]`` (current epoch over
    total epochs). If ``step_ratio`` is ``None``, defaults to
    ``current_epoch / num_epochs`` (which is ``0.0`` at the start of training).

    The proactive ``rho(t)`` is a cosine schedule that does not depend on the
    model's confidence (unlike the simplification in Section 6.3.1), so it
    genuinely oscillates with training progress and avoids the
    "false-rebound" issue. The reactive ``s_react(t)`` provides per-batch
    confidence feedback. Per-class parameters ``(a_y, b_y, c_y)`` restore
    class-awareness (Section 6.3.3).

    Degenerate relationship (Theorem 6.11): ``a_y = b_y = 0`` gives
    ``lambda_y = sigma(c_y)`` (constant), absorbed into ``sigma(eps_y)``,
    recovering MBACE up to the constant scaling. To obtain numerical
    equivalence one must rescale ``eps_y`` so that
    ``sigma(eps_y_MBA_CE) = sigma(eps_y_MBAPS) * sigma(c_y)``.

    Args:
        num_classes: Number of target classes ``C``.
        gamma: Rational-gate steepness (default ``1.0``). Must be ``>= 0``.
        delta: Tempering floor in ``(0, 1)`` (default ``1e-3``).
        gamma_learnable: If ``True``, ``gamma`` becomes an ``nn.Parameter``
            (default ``False``).
        num_epochs: Total number of epochs ``T`` (default ``200``). Used for
            fallback ``step_ratio`` computation when ``step_ratio`` is
            ``None``.
        eps_init: Initial value of per-class ``eps_y`` (default ``0.0``).
        a_init: Initial value of per-class ``a_y`` (default ``0.0``).
        b_init: Initial value of per-class ``b_y`` (default ``0.0``).
        c_init: Initial value of per-class ``c_y`` (default ``0.0``).
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 1.0,
        delta: float = 1e-3,
        gamma_learnable: bool = False,
        num_epochs: int = 200,
        eps_init: float = 0.0,
        a_init: float = 0.0,
        b_init: float = 0.0,
        c_init: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
        self.num_classes = num_classes
        self.num_epochs = int(num_epochs)
        self.delta = float(delta)
        if gamma_learnable:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        # ``current_epoch`` is a buffer so that it moves with the module's
        # device and is part of the state_dict (resumable training).
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
        self.eps = nn.Parameter(torch.full((num_classes,), float(eps_init)))
        self.a = nn.Parameter(torch.full((num_classes,), float(a_init)))
        self.b = nn.Parameter(torch.full((num_classes,), float(b_init)))
        self.c = nn.Parameter(torch.full((num_classes,), float(c_init)))

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch (used when ``step_ratio`` is ``None``).

        The trainer can call this at the start of each epoch so that
        ``forward(logits, targets)`` (without an explicit ``step_ratio``)
        uses the correct progress fraction ``current_epoch / num_epochs``.
        """
        self.current_epoch.fill_(int(epoch))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        step_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        # Resolve step_ratio.
        if step_ratio is None:
            step_ratio = float(self.current_epoch.item()) / max(self.num_epochs, 1)
        step_ratio = float(step_ratio)
        # Clamp to [0, 1] for safety.
        if step_ratio < 0.0:
            step_ratio = 0.0
        elif step_ratio > 1.0:
            step_ratio = 1.0

        log_probs = F.log_softmax(logits, dim=-1)                   # (B, C)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        pt = log_pt.exp()                                            # (B,)
        gamma = _gamma_value(self.gamma)
        phi = _rational_gate(pt, gamma)
        tau = _tempered_loss(pt, self.delta)

        # Proactive schedule: rho(t) = 0.5 * (1 + cos(pi * step_ratio)).
        # step_ratio=0 -> rho=1 (start); step_ratio=1 -> rho=0 (end);
        # step_ratio=0.5 -> rho=0.5.
        rho = 0.5 * (1.0 + math.cos(math.pi * step_ratio))          # python float

        # Reactive signal: batch confidence variance s_react(t).
        s_react = ((pt - pt.mean()) ** 2).mean()                     # scalar tensor

        # Per-class lambda_y gathered at the target indices.
        a_y = self.a[targets]                                        # (B,)
        b_y = self.b[targets]                                        # (B,)
        c_y = self.c[targets]                                        # (B,)
        lambda_y = torch.sigmoid(a_y * rho + b_y * s_react + c_y)    # (B,)

        sigma_eps = torch.sigmoid(self.eps[targets])                 # (B,)
        loss = (1.0 + sigma_eps * lambda_y * phi) * tau
        return loss.mean()


__all__ = ["MBACE", "MBAF", "MBAPS"]
