"""Evaluation metrics and the bootstrap confidence interval used for reporting.

Accuracy on the multi-class datasets, ROC-AUC on the binary Platonov targets
(minesweeper, tolokers, questions), as in Platonov et al. (2023).

The reported spread is a non-parametric 95% bootstrap CI over the ten splits:
1000 resamples of the ten per-split scores, the 2.5/97.5 percentiles of the
resampled means, and the larger of the two deviations from the observed mean as
the half-width.  This is a dependency-free reimplementation of the
``seaborn.algorithms.bootstrap`` + ``seaborn.utils.ci`` call used in the
original experiments; the two agree to within 1e-3.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def evaluate(logits: Tensor, y: Tensor, mask: Tensor, is_binary: bool) -> float:
    """ROC-AUC on binary targets, accuracy otherwise."""
    if is_binary:
        from sklearn.metrics import roc_auc_score

        probs = torch.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
        targets = y[mask].detach().cpu().numpy()
        if len(set(targets.tolist())) < 2:  # degenerate split, AUC undefined
            return 0.5
        return float(roc_auc_score(targets, probs))
    return float((logits[mask].argmax(1) == y[mask]).float().mean().item())


def bootstrap_ci(scores: Sequence[float], n_boot: int = 1000, level: float = 95.0,
                 seed: int = 0) -> float:
    """Half-width of the ``level``% bootstrap CI of the mean, in the same units as ``scores``."""
    scores = np.asarray(list(scores), dtype=np.float64)
    if scores.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, scores.size, size=(n_boot, scores.size))
    means = scores[idx].mean(axis=1)
    tail = (100.0 - level) / 2.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return float(np.max(np.abs(np.array([lo, hi]) - scores.mean())))


def summarise(scores: Sequence[float], scale: float = 100.0) -> dict:
    """``{'mean', 'std', 'ci95'}`` for a list of per-split scores, scaled to percent."""
    arr = np.asarray(list(scores), dtype=np.float64)
    return {
        "mean": float(arr.mean() * scale),
        "std": float(arr.std() * scale),
        "ci95": bootstrap_ci(arr) * scale,
        "per_split": [float(v * scale) for v in arr],
    }
