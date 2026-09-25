"""Training loop, per-split runner, and the config that drives them.

The loop is the one used for every number in the paper:

* full-batch training with NLL on the labelled training nodes;
* two optimiser groups -- the router, per-step bias and momentum coefficients
  at ``router_lr`` with no weight decay, everything else at ``lr``;
* BernNet-style early stopping: stop once the current validation loss exceeds
  the mean of the previous ``patience`` validation losses;
* the reported score is the test metric at the epoch with the best *validation
  metric*, never the best test metric.

``torch.manual_seed(split_index)`` before each split, so a given (config,
split) pair is deterministic and independent of the order splits are run in.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset
from .graph import balanced_forman_curvature, build_w_hat
from .metrics import evaluate, summarise
from .model import Castor


@dataclass
class TrainConfig:
    """Everything that defines one CASTOR run, in one serialisable object."""

    # --- architecture ---------------------------------------------------
    hidden: int = 64
    state_dim: Optional[int] = None      # None -> num_classes (paper's C-dim state)
    K: int = 5
    router_hidden: Optional[int] = 64    # None -> linear router
    norm: str = "layer"                  # 'layer' | 'batch' | 'none'
    hop_bias_init: str = "default"       # smoothing-then-preserve prior
    use_h0: bool = True
    pre_dropout: float = 0.3
    hid_dropout: float = 0.5
    use_bfc: bool = False
    routing_groups: Any = 1              # 1 = scalar (paper) | G groups | 'channel'

    # --- momentum -------------------------------------------------------
    momentum_mode: str = "heavyball"     # 'off' | 'heavyball' | 'nesterov'
    momentum_init: float = 0.0
    per_hop_beta: bool = True
    constrained_beta: bool = False       # False = sign-flexible (paper default)

    # --- ablation switches ----------------------------------------------
    no_router: bool = False
    no_anchor: bool = False
    drop_primitive: Optional[str] = None

    # --- optimisation ---------------------------------------------------
    lr: float = 0.05
    router_lr: Optional[float] = 0.01    # None -> one param group, router shares `lr` and `weight_decay`
    weight_decay: float = 0.0
    optimizer: str = "adam"              # 'adam' | 'adamw'
    epochs: int = 1000
    patience: int = 200                  # 0 disables early stopping
    grad_clip: Optional[float] = None

    # --- model selection --------------------------------------------------
    # Which quantity picks the reported epoch.  'val_metric' takes the epoch
    # with the highest validation accuracy/AUC; 'val_loss' takes the epoch with
    # the lowest validation loss.  Both appear in the original scripts, so both
    # are reproducible here; 'val_metric' is the protocol stated in Appendix E.4.
    selection: str = "val_metric"        # 'val_metric' | 'val_loss'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def build(self, dataset: Dataset) -> Castor:
        return Castor(
            dataset.num_features,
            dataset.num_classes,
            hidden=self.hidden,
            state_dim=self.state_dim,
            K=self.K,
            pre_dropout=self.pre_dropout,
            hid_dropout=self.hid_dropout,
            use_bfc=self.use_bfc,
            norm=self.norm,
            router_hidden=self.router_hidden,
            hop_bias_init=self.hop_bias_init,
            momentum_mode=self.momentum_mode,
            momentum_init=self.momentum_init,
            per_hop_beta=self.per_hop_beta,
            constrained_beta=self.constrained_beta,
            use_h0=self.use_h0 and not self.no_anchor,
            no_router=self.no_router,
            drop_primitive=self.drop_primitive,
            routing_groups=self.routing_groups,
        )


class GraphCache:
    """Device-resident features, labels and operator for one dataset.

    Built once and reused across every split and config on a worker, so the
    sparse normalisation and the (expensive) curvature computation are not
    repeated per run.
    """

    def __init__(self, dataset: Dataset, device, need_curvature: bool = False):
        from torch_geometric.utils import to_undirected

        self.dataset = dataset
        self.device = device
        self.x = dataset.x.to(device)
        self.y = dataset.y.to(device)
        edge_index = to_undirected(dataset.edge_index, num_nodes=dataset.num_nodes)
        self.w_hat = build_w_hat(edge_index, dataset.num_nodes, device)
        self.curvature = (
            balanced_forman_curvature(self.w_hat, dataset.num_nodes, device)
            if need_curvature else None
        )

    def ensure_curvature(self):
        if self.curvature is None:
            self.curvature = balanced_forman_curvature(
                self.w_hat, self.dataset.num_nodes, self.device
            )
        return self.curvature


def build_optimizer(model: Castor, cfg: TrainConfig):
    """Two param groups when ``router_lr`` is set, one when it is ``None``.

    Appendix E.4 describes the two-group form: the router, per-step bias and
    momentum coefficients get their own learning rate and no weight decay.  A
    few of the small-graph recipes were tuned with a single group instead, so
    ``router_lr=None`` reproduces those exactly.
    """
    cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    if cfg.router_lr is None:
        return cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return cls([
        {"params": model.main_parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": model.router_parameters(), "lr": cfg.router_lr, "weight_decay": 0.0},
    ])


def train_split(cache: GraphCache, cfg: TrainConfig, split_index: int,
                return_model: bool = False) -> Dict[str, Any]:
    """Train on one split; return the test metric at the best-validation epoch.

    With ``return_model=True`` the returned model carries the weights of the
    *selected* epoch (the one the reported test metric comes from), not the
    last epoch trained, so anything read off it describes the reported number.
    """
    ds = cache.dataset
    device = cache.device
    curvature = cache.ensure_curvature() if cfg.use_bfc else None

    torch.manual_seed(split_index)
    train_mask, val_mask, test_mask = ds.split(split_index)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = cfg.build(ds).to(device)
    opt = build_optimizer(model, cfg)

    if cfg.selection not in ("val_metric", "val_loss"):
        raise ValueError(f"unknown selection rule {cfg.selection!r}")
    best_val, best_test, best_epoch = -1.0, 0.0, -1
    best_val_loss = float("inf")
    best_state = None
    val_loss_hist = []

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(cache.x, cache.w_hat, curvature)
        loss = F.nll_loss(out[train_mask], cache.y[train_mask])
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(cache.x, cache.w_hat, curvature)
            val_loss = F.nll_loss(logits[val_mask], cache.y[val_mask]).item()
            val_metric = evaluate(logits, cache.y, val_mask, ds.is_binary)
            test_metric = evaluate(logits, cache.y, test_mask, ds.is_binary)

        if cfg.selection == "val_metric":
            improved = val_metric > best_val
        else:
            improved = val_loss < best_val_loss
        if improved:
            best_val, best_val_loss = val_metric, min(val_loss, best_val_loss)
            best_test, best_epoch = test_metric, epoch
            if return_model:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        val_loss_hist.append(val_loss)
        if cfg.patience > 0 and epoch > cfg.patience:
            tail = val_loss_hist[-(cfg.patience + 1):-1]
            if val_loss > float(np.mean(tail)):
                break

    result = {
        "split": split_index,
        "val": best_val,
        "test": best_test,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
    }
    betas = model.block.betas()
    if betas is not None:
        result["beta"] = [float(b) for b in betas.detach().cpu()]
    if return_model:
        if best_state is not None:
            model.load_state_dict(best_state)
        result["model"] = model
    return result


def run_config(dataset: Dataset, cfg: TrainConfig, splits: Optional[Sequence[int]] = None,
               device=None, cache: Optional[GraphCache] = None,
               name: str = "", save_dir: Optional[str] = None,
               save_tag: str = "") -> Dict[str, Any]:
    """Train ``cfg`` on each requested split and aggregate into mean / CI.

    With ``save_dir`` set, the selected-epoch weights of every split are written
    to ``<save_dir>/<dataset>__<save_tag>__split<i>.pt`` together with the config
    and the split's scores, so analyses can be run on exactly the models that
    produced the reported numbers.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = cache or GraphCache(dataset, device, need_curvature=cfg.use_bfc)
    splits = list(range(dataset.num_splits)) if splits is None else list(splits)

    t0 = time.time()
    per_split = []
    for s in splits:
        r = train_split(cache, cfg, s, return_model=save_dir is not None)
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            model = r.pop("model")
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": cfg.to_dict(), "dataset": dataset.name, "split": s,
                "name": name, "val": r["val"], "test": r["test"],
                "best_epoch": r["best_epoch"],
            }, os.path.join(save_dir, f"{dataset.name}__{save_tag}__split{s:02d}.pt"))
        per_split.append(r)
    test_scores = [r["test"] for r in per_split]
    val_scores = [r["val"] for r in per_split]

    summary = summarise(test_scores)
    return {
        "name": name,
        "dataset": dataset.name,
        "metric": dataset.metric_name,
        "config": cfg.to_dict(),
        "splits": splits,
        "test_mean": summary["mean"],
        "test_std": summary["std"],
        "test_ci95": summary["ci95"],
        "test_per_split": summary["per_split"],
        "val_mean": float(np.mean(val_scores) * 100),
        "beta_final": per_split[-1].get("beta"),
        "seconds": time.time() - t0,
    }
