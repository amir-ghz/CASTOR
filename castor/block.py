"""The CASTOR propagation block.

One step of the block is

    dZ    = (I - W_hat) Z                        # smoothing residual (high-pass)
    a     = softmax( rho([dZ || Z0]) + b_k )     # per-node weights on the 3-simplex
    T_k Z = a0 * (W_hat Z) + a1 * dZ + a2 * Z    # routed step
    Z'    = T_k Z + beta_k * (Z - Z_prev)        # momentum coupling

and the block iterates it ``K`` times.  This module is the single source of
truth for that computation: every variant that appears in the paper (C-dim vs
wide state, shared vs per-hop beta, sign-flexible vs sigmoid-constrained beta,
heavy-ball vs Nesterov, and all nine ablation switches) is a flag here rather
than a separate copy of the code.

Every number in the paper was produced by this computation; the released
configurations under ``configs/table1/`` name the exact flags for each
benchmark.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (BatchNorm1d, Identity, LayerNorm, Linear, Module,
                      Parameter, ReLU, Sequential)

#: Index of each primitive in the router's 3-dim output.
SMOOTH, RESIDUAL, IDENTITY = 0, 1, 2

_PRIMITIVE_INDEX = {"smooth": SMOOTH, "resid": RESIDUAL, "identity": IDENTITY}


def _make_norm(kind: str, dim: int) -> Module:
    if kind == "layer":
        return LayerNorm(dim)
    if kind == "batch":
        return BatchNorm1d(dim)
    if kind in ("none", None):
        return Identity()
    raise ValueError(f"unknown norm {kind!r}")


def _init_hop_bias(bias: Tensor, kind: str, K: int) -> None:
    """In-place initialisation of the per-step router bias ``b_k``.

    ``'default'`` is the smoothing-then-preserve prior of Eq. (7):
    ``b_k = 1/2 * ((K-1-k)/(K-1), 0, k/(K-1))``.
    """
    if kind == "zero":
        return
    if kind == "preserve":
        bias[:, IDENTITY] = 1.0
    elif kind == "contrast":
        bias[:, RESIDUAL] = 1.0
    elif kind == "smooth":
        bias[:, SMOOTH] = 1.0
    elif kind == "default":
        for k in range(K):
            t = k / max(K - 1, 1)
            bias[k, SMOOTH] = 0.5 * (1.0 - t)
            bias[k, IDENTITY] = 0.5 * t
    else:
        raise ValueError(f"unknown hop_bias_init {kind!r}")


class CastorBlock(Module):
    """``K`` iterations of the routed step with momentum coupling.

    Parameters
    ----------
    state_dim
        Width of the propagation state.  Set to the number of classes ``C`` for
        the class-logit variant described in Section 4.3 of the paper, or to a
        wider ``D`` for the variant used on the Platonov benchmarks (which then
        needs a ``Linear(D, C)`` head -- see :class:`castor.model.Castor`).
    K
        Number of propagation steps.
    router_hidden
        Hidden width of the router MLP.  ``None`` makes the router a single
        linear map ``Linear(2 * state_dim, 3)``.
    momentum_mode
        ``'off'`` (beta == 0, first-order), ``'heavyball'`` (Polyak: momentum
        added after the routed step) or ``'nesterov'`` (lookahead: momentum
        applied to the state *before* the routed step).
    constrained_beta
        If True, beta is reparameterised as ``sigmoid(raw)`` and is therefore
        confined to (0, 1) -- this is ablation **A6 PosBeta**.  The paper's
        default is False, i.e. beta is a free real number.
    per_hop_beta
        One beta per step (paper default) vs a single shared scalar (**A7**).
    use_h0
        Feed the initial state ``Z0`` to the router alongside the residual.
        Disabling it is ablation **A2 NoAnchor**.
    no_router
        Replace the per-node router by one shared 3-vector per step
        (ablation **A1 NoRouter**).
    drop_primitive
        Mask one primitive's logit before the softmax: ``'smooth'`` (**A3a**),
        ``'resid'`` (**A3b**) or ``'identity'`` (**A3c**).
    routing_groups
        How many channel groups share one routing decision.  ``1`` (the paper's
        default) emits three weights per node that are broadcast across every
        channel of the state -- the rank-one routing of Section 4.1.  ``G > 1``
        splits the ``state_dim`` channels into ``G`` contiguous blocks with one
        3-vector each, and ``'channel'`` (equivalently ``G == state_dim``) gives
        every channel its own decision.  Together with ``no_router`` this is a
        node-uniform per-channel filter bank: one learnable 3-vector per
        (step, group) and no dependence on the node at all.  The ``G == 1`` code
        path is untouched by this option.
    """

    def __init__(
        self,
        state_dim: int,
        K: int,
        *,
        norm: str = "layer",
        router_hidden: Optional[int] = 64,
        hop_bias_init: str = "default",
        momentum_mode: str = "heavyball",
        momentum_init: float = 0.0,
        per_hop_beta: bool = True,
        constrained_beta: bool = False,
        use_h0: bool = True,
        no_router: bool = False,
        drop_primitive: Optional[str] = None,
        routing_groups: Union[int, str] = 1,
    ) -> None:
        super().__init__()
        if momentum_mode not in ("off", "heavyball", "nesterov"):
            raise ValueError(f"unknown momentum_mode {momentum_mode!r}")
        if drop_primitive not in (None, "smooth", "resid", "identity"):
            raise ValueError(f"unknown drop_primitive {drop_primitive!r}")
        if routing_groups in ("channel", "per_channel"):
            routing_groups = state_dim
        routing_groups = int(routing_groups)
        if routing_groups < 1 or state_dim % routing_groups != 0:
            raise ValueError(
                f"routing_groups={routing_groups} must be a positive divisor of "
                f"state_dim={state_dim}")

        self.K = K
        self.state_dim = state_dim
        self.groups = routing_groups
        self.momentum_mode = momentum_mode
        self.per_hop_beta = per_hop_beta
        self.constrained_beta = constrained_beta
        self.use_h0 = use_h0
        self.no_router = no_router
        self.drop_index = _PRIMITIVE_INDEX.get(drop_primitive) if drop_primitive else None

        # --- router -------------------------------------------------------
        router_in = (2 if use_h0 else 1) * state_dim
        router_out = 3 * self.groups
        if no_router:
            self.router = None
            shape = (K, 3) if self.groups == 1 else (K, self.groups, 3)
            self.global_logits = Parameter(torch.zeros(*shape))
        elif router_hidden is None:
            self.router = Linear(router_in, router_out)
        else:
            self.router = Sequential(
                Linear(router_in, router_hidden), ReLU(), Linear(router_hidden, router_out)
            )

        # --- per-step bias b_k -------------------------------------------
        self.hop_bias = Parameter(torch.zeros(K, 3))
        with torch.no_grad():
            _init_hop_bias(self.hop_bias, hop_bias_init, K)

        # --- per-step momentum beta_k ------------------------------------
        if momentum_mode != "off":
            if constrained_beta:
                p = min(max(float(momentum_init), 1e-6), 1.0 - 1e-6)
                raw = math.log(p / (1.0 - p))
            else:
                raw = float(momentum_init)
            shape = (K,) if per_hop_beta else ()
            self.beta_raw = Parameter(torch.full(shape, raw))
        else:
            self.beta_raw = None

        self.norm = _make_norm(norm, state_dim)

    # -- introspection ----------------------------------------------------

    def betas(self) -> Optional[Tensor]:
        """The effective ``beta_k`` values, after any sigmoid reparameterisation."""
        if self.beta_raw is None:
            return None
        raw = self.beta_raw if self.per_hop_beta else self.beta_raw.expand(self.K)
        return torch.sigmoid(raw) if self.constrained_beta else raw

    def router_parameters(self):
        """Parameters that belong in the separate ``router_lr`` optimiser group."""
        params = []
        if self.router is not None:
            params += list(self.router.parameters())
        if self.no_router:
            params.append(self.global_logits)
        params.append(self.hop_bias)
        if self.beta_raw is not None:
            params.append(self.beta_raw)
        return params

    # -- forward ----------------------------------------------------------

    def _beta_at(self, k: int) -> Optional[Tensor]:
        """Effective ``beta_k``.  ``None`` at k == 0, where the velocity vanishes."""
        if self.momentum_mode == "off" or k == 0:
            return None
        raw = self.beta_raw[k] if self.per_hop_beta else self.beta_raw
        return torch.sigmoid(raw) if self.constrained_beta else raw

    def _mixing_weights(self, residual: Tensor, h0: Tensor, k: int) -> Tensor:
        """Weights on the 3-simplex: ``[N, 3]`` per node, or ``[N, G, 3]`` per
        (node, channel group) when ``routing_groups > 1``."""
        n = residual.size(0)
        if self.no_router:
            per_step = self.global_logits[k]                      # [3] or [G, 3]
            logits = per_step.unsqueeze(0).expand(n, *per_step.shape)
        else:
            router_in = torch.cat([residual, h0], dim=-1) if self.use_h0 else residual
            logits = self.router(router_in)
            if self.groups > 1:
                logits = logits.view(n, self.groups, 3)
        logits = logits + self.hop_bias[k]                        # [3] broadcasts
        if self.drop_index is not None:
            logits = logits.clone()
            logits[..., self.drop_index] = -1e9
        return F.softmax(logits, dim=-1)

    def _combine_grouped(self, a: Tensor, smoothed: Tensor, residual: Tensor,
                         state: Tensor) -> Tensor:
        """Routed step for ``routing_groups > 1``.

        ``a`` is ``[N, G, 3]`` and each group's 3-vector is applied to its
        contiguous block of ``state_dim // G`` channels.  Viewing the state as
        ``[N, G, state_dim // G]`` applies the weights without materialising a
        per-channel copy of them, so a grouped step holds no more activation
        memory than the scalar step; only the per-channel router output itself
        (``3 * state_dim`` logits per node) grows with the granularity.
        """
        n = state.size(0)
        shape = (n, self.groups, self.state_dim // self.groups)
        routed = (
            a[:, :, SMOOTH:SMOOTH + 1] * smoothed.reshape(shape)
            + a[:, :, RESIDUAL:RESIDUAL + 1] * residual.reshape(shape)
            + a[:, :, IDENTITY:IDENTITY + 1] * state.reshape(shape)
        )
        return routed.reshape(n, self.state_dim)

    def forward(self, z0: Tensor, w_hat: Tensor, return_trace: bool = False):
        """Run the block.  ``z0`` is ``Z^{(0)}``; ``w_hat`` the sparse operator.

        With ``return_trace=True`` also returns the per-step mixing weights,
        which is what the per-node routing figure (Appendix A.2) reads off.
        """
        h = z0
        h_anchor = z0
        h_prev = z0.clone()
        trace = [] if return_trace else None

        for k in range(self.K):
            beta = self._beta_at(k)

            # Nesterov looks ahead before propagating; heavy-ball does not.
            if self.momentum_mode == "nesterov" and beta is not None:
                state = h + beta * (h - h_prev)
            else:
                state = h

            smoothed = torch.sparse.mm(w_hat, state)
            residual = state - smoothed

            a = self._mixing_weights(residual, h_anchor, k)
            if self.groups == 1:
                routed = (
                    a[:, SMOOTH:SMOOTH + 1] * smoothed
                    + a[:, RESIDUAL:RESIDUAL + 1] * residual
                    + a[:, IDENTITY:IDENTITY + 1] * state
                )
            else:
                routed = self._combine_grouped(a, smoothed, residual, state)

            if self.momentum_mode == "heavyball" and beta is not None:
                h_new = routed + beta * (h - h_prev)
            else:
                h_new = routed

            if return_trace:
                trace.append(a.detach())

            h_prev = h
            h = h_new

        out = self.norm(h)
        return (out, trace) if return_trace else out
