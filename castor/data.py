"""Dataset and split loading for all sixteen benchmarks used in the paper.

Three families of graphs, three split protocols:

===========================  =========================  ==========================
family                       graphs                     split protocol
===========================  =========================  ==========================
Planetoid (homophilic)       cora, citeseer, pubmed     10 x 48/32/20 (BernNet)
WebKB / Wikipedia / Actor    texas, wisconsin, cornell  10 x 48/32/20 (BernNet)
                             actor, squirrel, chameleon
Platonov mid-scale           roman_empire,              10 x 50/25/25 (official)
                             amazon_ratings,
                             minesweeper, tolokers,
                             questions
Platonov de-duplicated       chameleon_filtered,        10 x 50/25/25 (official)
                             squirrel_filtered
===========================  =========================  ==========================

The 48/32/20 splits are the ones distributed with the paper under
``splits/bernet_48_32_20/``; the 60/20/20 GeomGCN splits are also shipped
(``splits/geom_gcn_splits/``) because several baselines were originally tuned
under that protocol.  Which one is used is an explicit argument, never a
default that silently changes between experiments.

Node features are row-normalised (``NormalizeFeatures``) on the PyG graphs, and
all edge sets are symmetrised, matching every heterophily paper we compare to.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

#: Package root, so the default data/ and splits/ directories resolve without
#: the caller having to know where the repository lives.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_SPLITS_DIR = os.path.join(REPO_ROOT, "splits")

PLATONOV_DATASETS = [
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions",
]
FILTERED_DATASETS = ["chameleon_filtered", "squirrel_filtered"]
GEOM_GCN_DATASETS = [
    "cora", "citeseer", "pubmed",
    "cornell", "texas", "wisconsin",
    "chameleon", "squirrel", "actor",
]
ALL_DATASETS = GEOM_GCN_DATASETS + PLATONOV_DATASETS + FILTERED_DATASETS

#: Datasets scored with ROC-AUC rather than accuracy (binary targets).
BINARY_DATASETS = {"minesweeper", "tolokers", "questions"}

#: The fourteen datasets of Table 1, in table order.
TABLE1_DATASETS = [
    "cora", "citeseer", "pubmed",
    "texas", "wisconsin", "actor", "squirrel", "chameleon", "cornell",
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions",
]

#: Display names used in generated tables.
DISPLAY_NAME = {
    "cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed",
    "texas": "Texas", "wisconsin": "Wisconsin", "actor": "Actor",
    "squirrel": "Squirrel", "chameleon": "Chameleon", "cornell": "Cornell",
    "roman_empire": "roman-empire", "amazon_ratings": "amazon-ratings",
    "minesweeper": "minesweeper", "tolokers": "tolokers", "questions": "questions",
    "chameleon_filtered": "Chameleon-filtered",
    "squirrel_filtered": "Squirrel-filtered",
}


@dataclass
class Dataset:
    """One benchmark: features, labels, a symmetrised edge set, and its splits.

    ``train_masks`` / ``val_masks`` / ``test_masks`` are ``[S, N]`` boolean, one
    row per official split.  ``split_protocol`` records which partitioning the
    masks came from (``'48/32/20'``, ``'50/25/25'`` or ``'60/20/20'``).
    """

    name: str
    x: Tensor           # [N, F] float32
    y: Tensor           # [N]    int64
    edge_index: Tensor  # [2, E] int64, symmetrised
    train_masks: Tensor  # [S, N] bool
    val_masks: Tensor    # [S, N] bool
    test_masks: Tensor   # [S, N] bool
    split_protocol: str  # '48/32/20' | '50/25/25' | '60/20/20'

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def num_splits(self) -> int:
        return self.train_masks.size(0)

    @property
    def is_binary(self) -> bool:
        return self.name in BINARY_DATASETS

    @property
    def metric_name(self) -> str:
        return "auc" if self.is_binary else "acc"

    def split(self, i: int):
        return self.train_masks[i], self.val_masks[i], self.test_masks[i]

    def __repr__(self) -> str:
        return (f"Dataset({self.name}, N={self.num_nodes}, F={self.num_features}, "
                f"C={self.num_classes}, E={self.edge_index.size(1)}, "
                f"splits={self.num_splits}x{self.split_protocol})")


# ---------------------------------------------------------------------------
# Platonov `.npz` graphs (mid-scale + de-duplicated wiki)
# ---------------------------------------------------------------------------

def _load_platonov(name: str, data_dir: str) -> Dataset:
    path = os.path.join(data_dir, "platonov", f"{name}.npz")
    if not os.path.exists(path):  # tolerate a flat data/ layout too
        path = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Platonov archive not found for {name!r}: {path}")

    z = np.load(path)
    return Dataset(
        name=name,
        x=torch.tensor(z["node_features"], dtype=torch.float32),
        y=torch.tensor(z["node_labels"], dtype=torch.long),
        edge_index=torch.tensor(z["edges"], dtype=torch.long).T.contiguous(),
        train_masks=torch.tensor(z["train_masks"], dtype=torch.bool),
        val_masks=torch.tensor(z["val_masks"], dtype=torch.bool),
        test_masks=torch.tensor(z["test_masks"], dtype=torch.bool),
        split_protocol="50/25/25",
    )


# ---------------------------------------------------------------------------
# PyG graphs (Planetoid / WebKB / WikipediaNetwork / Actor)
# ---------------------------------------------------------------------------

def _pyg_dataset(name: str, cache_dir: str):
    from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork
    from torch_geometric.transforms import NormalizeFeatures

    n = name.lower()
    tx = NormalizeFeatures()
    if n in ("cora", "citeseer", "pubmed"):
        return Planetoid(root=os.path.join(cache_dir, n), name=n.capitalize(), transform=tx)
    if n in ("cornell", "texas", "wisconsin"):
        return WebKB(root=os.path.join(cache_dir, n), name=n.capitalize(), transform=tx)
    if n in ("chameleon", "squirrel"):
        return WikipediaNetwork(root=os.path.join(cache_dir, n), name=n, transform=tx)
    if n == "actor":
        return Actor(root=os.path.join(cache_dir, "actor"), transform=tx)
    raise ValueError(f"unknown PyG dataset {name!r}")


def _stack_split_files(pattern: str, num_splits: int, num_nodes: int, name: str):
    train, val, test = [], [], []
    for i in range(num_splits):
        path = pattern.format(i=i)
        if not os.path.exists(path):
            raise FileNotFoundError(f"split file not found: {path}")
        z = np.load(path)
        train.append(torch.tensor(z["train_mask"].flatten(), dtype=torch.bool))
        val.append(torch.tensor(z["val_mask"].flatten(), dtype=torch.bool))
        test.append(torch.tensor(z["test_mask"].flatten(), dtype=torch.bool))
    train, val, test = torch.stack(train), torch.stack(val), torch.stack(test)
    if train.size(1) != num_nodes:
        raise ValueError(
            f"split width {train.size(1)} != {num_nodes} nodes for {name!r}; "
            "this usually means a chameleon/squirrel version mismatch."
        )
    return train, val, test


def _load_geom_gcn(name: str, data_dir: str, splits_dir: str, protocol: str) -> Dataset:
    from torch_geometric.utils import to_undirected

    pyg = _pyg_dataset(name, os.path.join(data_dir, "pyg"))
    data = pyg[0]
    num_nodes = data.x.size(0)
    edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)

    if protocol == "48/32/20":
        pattern = os.path.join(splits_dir, "bernet_48_32_20", f"{name}_split_0.48_0.32_{{i}}.npz")
    elif protocol == "60/20/20":
        pattern = os.path.join(splits_dir, "geom_gcn_splits", f"{name}_split_0.6_0.2_{{i}}.npz")
    else:
        raise ValueError(f"unknown split protocol {protocol!r} for a GeomGCN dataset")

    train, val, test = _stack_split_files(pattern, 10, num_nodes, name)
    return Dataset(name, data.x, data.y, edge_index, train, val, test, protocol)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(
    name: str,
    data_dir: Optional[str] = None,
    splits_dir: Optional[str] = None,
    protocol: Optional[str] = None,
) -> Dataset:
    """Load a benchmark by name.

    ``protocol`` selects the split family for the nine PyG graphs and defaults
    to ``'48/32/20'`` -- the protocol used for Table 1.  Platonov graphs carry
    their official 50/25/25 masks inside the archive and ignore this argument.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    splits_dir = splits_dir or DEFAULT_SPLITS_DIR

    if name in PLATONOV_DATASETS or name in FILTERED_DATASETS:
        return _load_platonov(name, data_dir)
    if name in GEOM_GCN_DATASETS:
        return _load_geom_gcn(name, data_dir, splits_dir, protocol or "48/32/20")
    raise ValueError(f"unknown dataset {name!r}; available: {ALL_DATASETS}")
