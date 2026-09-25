"""Graph operators used by the CASTOR block.

Two objects live here:

* :func:`build_w_hat` -- the symmetrically normalised adjacency with self-loops
  ``W_hat = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}`` returned as a *coalesced* sparse
  COO tensor.  Everything downstream multiplies by this one matrix.
* :func:`balanced_forman_curvature` -- a per-edge Balanced Forman Curvature
  vector aligned with ``W_hat._values()`` so it can gate the operator
  element-wise.  This is the optional ``use_bfc`` component.

Both are the implementations used to produce the paper's numbers.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


def build_w_hat(edge_index: Tensor, num_nodes: int, device) -> Tensor:
    """Symmetric normalised adjacency with self-loops, as a coalesced sparse COO tensor.

    The degree is taken from the *self-loop-augmented* edge list, so every node
    has degree >= 1 and the inverse square root is always finite.
    """
    edge_index = edge_index.to(device)
    loop = torch.arange(num_nodes, device=device)
    row = torch.cat([edge_index[0], loop])
    col = torch.cat([edge_index[1], loop])

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0), values, (num_nodes, num_nodes)
    ).coalesce()


def balanced_forman_curvature(w_hat: Tensor, num_nodes: int, device) -> Tensor:
    """Per-edge Balanced Forman Curvature, ordered to match ``w_hat._values()``.

    Triangle counts come from a scipy sparse-sparse product (memory-safe up to
    N ~ 50k).  Degrees are computed from the *symmetrised simple-graph*
    adjacency: mixing a directed out-degree with a symmetrised triangle count
    produces meaningless curvature on the Platonov graphs, which ship directed
    edges.

    Self-loop entries and degree-1 endpoints are zeroed, so the returned vector
    only gates genuine edges between nodes that can participate in a triangle.
    """
    indices = w_hat._indices()
    rows, cols = indices[0], indices[1]
    nonloop = rows != cols

    rows_np = rows[nonloop].cpu().numpy()
    cols_np = cols[nonloop].cpu().numpy()
    adj = sp.csr_matrix(
        (np.ones(rows_np.size, dtype=np.float32), (rows_np, cols_np)),
        shape=(num_nodes, num_nodes),
    )
    adj = ((adj + adj.T) > 0).astype(np.float32)

    deg_np = np.asarray(adj.sum(axis=1)).flatten()
    deg = torch.from_numpy(deg_np.astype(np.float32)).to(device)

    adj2 = adj @ adj
    rows_all = np.array(rows.cpu().numpy(), copy=True)
    cols_all = np.array(cols.cpu().numpy(), copy=True)
    tri_np = np.asarray(adj2[rows_all, cols_all]).flatten()
    tri = torch.from_numpy(tri_np.astype(np.float32)).to(device)

    d_i = deg[rows].clamp_min(1e-12)
    d_j = deg[cols].clamp_min(1e-12)
    d_max = torch.maximum(d_i, d_j)
    d_min = torch.minimum(d_i, d_j)

    ric = (2.0 / d_i) + (2.0 / d_j) - 2.0 + 2.0 * tri / d_max + tri / d_min
    ric[~nonloop] = 0.0
    ric[d_min < 1.5] = 0.0
    return ric


def apply_curvature_gate(w_hat: Tensor, curvature: Tensor, gamma: Tensor, tau: Tensor) -> Tensor:
    """Reweight ``w_hat``'s values by ``1 + gamma * tanh(curvature / tau)``."""
    tau = tau.clamp_min(1e-2)
    gate = 1.0 + gamma * torch.tanh(curvature / tau)
    return torch.sparse_coo_tensor(
        w_hat._indices(), w_hat._values() * gate, w_hat.shape
    ).coalesce()
