"""
Automatic loss balancing for DeluluNet's multi-term objective.

Motivation
----------
SHOT optimizes four terms (latent, prefusion, distill, ce) whose raw scales
differ and, more importantly, whose DECAY RATES differ: over a 128-epoch run
prefusion falls ~15x while ce falls only ~3x, so a fixed lambda that balances
the terms at step 0 no longer balances them at step 50k. Hand-tuning three
lambdas is also most of the sweep's dimensionality, which is expensive to
search and easy to miss.

Modes
-----
none
    Use the fixed lambdas as-is (current behaviour). Baseline.

running_mean
    Divide each term by an EMA of its own recent magnitude, so every term
    enters the sum at ~1.0 regardless of scale or decay rate. The fixed lambda
    is then a genuine *preference* rather than a scale correction. No learned
    parameters, no extra backward pass.

uncertainty
    Kendall, Gal & Cipolla (CVPR 2018), "Multi-Task Learning Using Uncertainty
    to Weigh Losses". Learns one log-variance scalar s_i per term:

        total = sum_i  exp(-s_i) * L_i  +  s_i

    exp(-s_i) is the learned precision; the +s_i term prevents the degenerate
    solution of driving all precisions to zero. The s_i are ordinary parameters
    trained by the same optimizer. This REPLACES lambda_latent /
    lambda_prefusion / lambda_distill, removing three dimensions from the sweep.

Caveat worth keeping in mind: both automatic modes push terms toward EQUAL
contribution. If a task genuinely wants one term to dominate -- e.g. distill
carrying most of the signal for modality transfer -- an automatic balancer will
actively suppress that. Run against `none` rather than assuming balance is
optimal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Terms the balancer may learn weights for.
#
# CE is deliberately EXCLUDED. SHOT alternates batch types: a labeled step
# produces only ce, an unlabeled step produces only latent/prefusion/distill
# (see _labeled_batch_step / _unlabeled_batch_step in shot.py). Uncertainty
# weighting assumes every task contributes to every step, so that the learned
# precisions are tied together by a shared gradient. Here s_ce would be
# estimated from a disjoint ~23% of steps with no competing pressure from the
# other three, and the balancer could quietly down-weight the only
# ground-truth-supervised term.
#
# So CE keeps its fixed --lambda_ce (the anchor), and the three unlabeled terms
# -- which DO co-occur in every unlabeled step, making their relative balance
# well-posed -- are the ones balanced. The labeled/unlabeled trade-off is
# instead controlled by --labeled_frequency, which the sweep already searches.
BALANCED_NAMES = ('latent', 'prefusion', 'distill')
ANCHOR_NAMES = ('ce',)
LOSS_NAMES = BALANCED_NAMES + ANCHOR_NAMES


class LossBalancer(nn.Module):
    def __init__(self, mode: str = 'none', base_weights: dict | None = None,
                 names=BALANCED_NAMES, ema: float = 0.99, eps: float = 1e-8,
                 device=None, anchors=ANCHOR_NAMES):
        super().__init__()
        if mode not in ('none', 'running_mean', 'uncertainty'):
            raise ValueError(f'unknown loss_balance mode: {mode!r}')
        self.mode = mode
        self.names = tuple(names)          # balanced terms
        self.anchors = tuple(anchors)      # fixed-weight terms (ce)
        self.ema = ema
        self.eps = eps
        all_names = self.names + self.anchors
        self.base = {n: float((base_weights or {}).get(n, 1.0)) for n in all_names}

        if mode == 'uncertainty':
            # s_i = log sigma_i^2, initialised at 0 -> precision exp(0) = 1
            self.log_var = nn.ParameterDict(
                {n: nn.Parameter(torch.zeros((), device=device)) for n in self.names})
        else:
            self.log_var = None

        if mode == 'running_mean':
            # buffers so they survive checkpointing
            for n in self.names:
                self.register_buffer(f'_rm_{n}', torch.ones((), device=device))
                self.register_buffer(f'_seen_{n}', torch.zeros((), device=device))

    # -- running_mean bookkeeping -------------------------------------------
    @torch.no_grad()
    def observe(self, values: dict):
        """Update EMA magnitudes from the detached scalar value of each loss."""
        if self.mode != 'running_mean':
            return
        for n in self.names:
            v = values.get(n)
            if v is None or v == 0.0:
                continue
            v = abs(float(v))
            buf = getattr(self, f'_rm_{n}')
            seen = getattr(self, f'_seen_{n}')
            if seen.item() == 0:
                buf.fill_(v)
                seen.fill_(1.0)
            else:
                buf.mul_(self.ema).add_((1 - self.ema) * v)

    def weights(self) -> dict:
        """Current multiplier per loss term, for the fixed-lambda code path."""
        if self.mode == 'none':
            return dict(self.base)
        # anchors always pass through at their fixed lambda
        out = {n: self.base[n] for n in self.anchors}
        if self.mode == 'running_mean':
            out.update({n: self.base[n] / (float(getattr(self, f'_rm_{n}').item()) + self.eps)
                        for n in self.names})
        else:  # uncertainty: exp(-s_i); the +s_i term is added by regularizer()
            out.update({n: torch.exp(-self.log_var[n]) * self.base[n]
                        for n in self.names})
        return out

    def regularizer(self, active: set | None = None):
        """The sum_i s_i term. Zero unless mode == 'uncertainty'."""
        if self.mode != 'uncertainty':
            return 0.0
        names = [n for n in self.names if active is None or n in active]
        if not names:
            return 0.0
        return sum(self.log_var[n] for n in names)

    def report(self) -> dict:
        """Loggable scalars describing the current balance."""
        if self.mode == 'none':
            return {}
        if self.mode == 'running_mean':
            return {f'balance/scale_{n}': float(getattr(self, f'_rm_{n}').item())
                    for n in self.names}
        return {f'balance/precision_{n}': float(torch.exp(-self.log_var[n]).item())
                for n in self.names}
