"""The full CASTOR pipeline: encoder -> CASTOR block -> (optional head) -> softmax.

    X  ->  Z^(0)  ->  Z^[K]  ->  Y

The encoder is the two-layer feedforward map of Eq. (12).  The block is
:class:`castor.block.CastorBlock`.  Whether a head is needed depends on the
propagation state width:

* ``state_dim == num_classes`` reproduces Section 4.3 exactly -- the block's
  output is already class-logit-shaped and the softmax sits directly on it.
* ``state_dim = D > num_classes`` is the wider variant used on the Platonov
  benchmarks, where a ``Linear(D, C)`` head maps the propagation state to
  logits.  Appendix D.3 documents this variant for the filtered benchmarks.

Both are the same code path; ``state_dim`` is the only switch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity, Linear, Module, Parameter

from .block import CastorBlock
from .graph import apply_curvature_gate


class Castor(Module):
    """Encoder + :class:`CastorBlock` + optional linear head.

    Parameters
    ----------
    num_features, num_classes
        Input feature width and number of target classes.
    hidden
        Encoder hidden width ``D`` in Eq. (12).
    state_dim
        Propagation state width.  ``None`` (default) means ``num_classes``,
        i.e. the class-logit state of Section 4.3 with no head.
    pre_dropout
        Feature dropout applied before each encoder linear map (``Drop_p``).
    hid_dropout
        Dropout applied once to ``Z^(0)`` before the recurrence (``Drop_h``).
    use_bfc
        Gate ``W_hat``'s edges by learnable Balanced Forman Curvature weights
        ``1 + gamma * tanh(curvature / tau)``.  Off by default; ``gamma`` is
        initialised to 0 so the gate starts as the identity.
    **block_kwargs
        Forwarded to :class:`CastorBlock` (``K``, ``momentum_mode``, ...).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        hidden: int = 64,
        state_dim: Optional[int] = None,
        K: int = 5,
        pre_dropout: float = 0.3,
        hid_dropout: float = 0.5,
        use_bfc: bool = False,
        **block_kwargs,
    ) -> None:
        super().__init__()
        state_dim = num_classes if state_dim is None else int(state_dim)

        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, state_dim)
        self.block = CastorBlock(state_dim, K, **block_kwargs)
        self.head = Identity() if state_dim == num_classes else Linear(state_dim, num_classes)

        self.pre_dropout = pre_dropout
        self.hid_dropout = hid_dropout

        self.use_bfc = use_bfc
        if use_bfc:
            self.curv_gamma = Parameter(torch.tensor([0.0]))
            self.curv_tau = Parameter(torch.tensor([1.0]))

    # -- optimiser groups -------------------------------------------------

    def router_parameters(self):
        """Router, per-step bias and momentum -- trained at ``router_lr``, no weight decay."""
        return self.block.router_parameters()

    def main_parameters(self):
        """Everything else -- encoder, head, curvature gate."""
        router_ids = {id(p) for p in self.router_parameters()}
        return [p for p in self.parameters() if id(p) not in router_ids]

    # -- forward ----------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Raw features -> initial propagation state ``Z^(0)``."""
        x = F.dropout(x, p=self.pre_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.pre_dropout, training=self.training)
        x = self.lin2(x)
        if self.hid_dropout > 0:
            x = F.dropout(x, p=self.hid_dropout, training=self.training)
        return x

    def operator(self, w_hat: Tensor, curvature: Optional[Tensor]) -> Tensor:
        """``W_hat``, optionally reweighted by the learnable curvature gate."""
        if not self.use_bfc:
            return w_hat
        if curvature is None:
            raise ValueError("use_bfc=True but no curvature vector was supplied")
        return apply_curvature_gate(w_hat, curvature, self.curv_gamma, self.curv_tau)

    def forward(
        self,
        x: Tensor,
        w_hat: Tensor,
        curvature: Optional[Tensor] = None,
        return_trace: bool = False,
    ):
        """Raw features to per-node log-probabilities.

        ``curvature`` is required only when ``use_bfc`` is set.  With
        ``return_trace=True`` the per-step routing weights are returned
        alongside the logits, for the routing figure of Appendix A.2.
        """
        w_eff = self.operator(w_hat, curvature)
        z0 = self.encode(x)
        out = self.block(z0, w_eff, return_trace=return_trace)
        if return_trace:
            z_k, trace = out
            return F.log_softmax(self.head(z_k), dim=-1), trace
        return F.log_softmax(self.head(out), dim=-1)
