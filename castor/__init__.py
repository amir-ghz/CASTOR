"""CASTOR -- a self-tuning graph filter via state-dependent operator composition.

Typical use::

    from castor import load, TrainConfig, run_config

    ds  = load("cornell")
    cfg = TrainConfig(K=10, momentum_mode="heavyball", momentum_init=0.7)
    print(run_config(ds, cfg)["test_mean"])

The package is deliberately small: :mod:`castor.block` is the propagation
block, :mod:`castor.model` wraps it in the encoder/head pipeline,
:mod:`castor.data` loads all sixteen benchmarks, and :mod:`castor.train` runs
one config over the ten official splits.
"""

from .block import CastorBlock
from .data import (ALL_DATASETS, BINARY_DATASETS, DISPLAY_NAME,
                   FILTERED_DATASETS, GEOM_GCN_DATASETS, PLATONOV_DATASETS,
                   TABLE1_DATASETS, Dataset, load)
from .graph import balanced_forman_curvature, build_w_hat
from .metrics import bootstrap_ci, evaluate, summarise
from .model import Castor
from .train import GraphCache, TrainConfig, run_config, train_split

__version__ = "1.0.0"

__all__ = [
    "Castor", "CastorBlock", "Dataset", "GraphCache", "TrainConfig",
    "load", "run_config", "train_split",
    "build_w_hat", "balanced_forman_curvature",
    "evaluate", "bootstrap_ci", "summarise",
    "ALL_DATASETS", "TABLE1_DATASETS", "GEOM_GCN_DATASETS",
    "PLATONOV_DATASETS", "FILTERED_DATASETS", "BINARY_DATASETS", "DISPLAY_NAME",
]
