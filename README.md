# Self-Tuning Graph Filters via State-Dependent Operator Composition

Code for the NeurIPS 2026 paper *Self-Tuning Graph Filters via State-Dependent
Operator Composition*. The model in the paper is called CASTOR.

CASTOR replaces the fixed propagation operator of a graph neural network with
one that is chosen per node and per propagation step. At each step a router
reads a node's current state and emits weights over three primitives, low-pass
smoothing, high-pass residual and identity, and a learnable momentum
coefficient couples consecutive steps through a second-order recurrence. The
same block is used on every benchmark in the paper, from Cora to Roman-empire,
with no per-dataset architectural switch.

## Overview

A standard GNN propagates through one linear filter, fixed at construction
time and applied to every node, every layer and every graph. The paper argues
that the appropriate filter varies along two axes and builds a block that
follows both:

- **Across nodes within a hop.** Different nodes in the same graph need
  different mixtures of smoothing, contrast and pass-through, and the mixture
  changes from one hop to the next. The router decides this per node at every
  step.
- **Across hops within a trajectory.** Some graphs benefit from carrying
  propagation forward across many hops and others need it damped. The momentum
  coefficient `β_k` is learned per step and can take either sign; `β_k = 0`
  recovers ordinary first-order propagation.

With node-uniform routing the block realises a polynomial in the normalised
adjacency, and SGC, APPNP, GPR-GNN, BernNet and rescaled-spectrum ChebNet are
special cases of its schedules. Per-node routing leaves that polynomial family.
The propagation block holds no per-hop weight matrices, so its parameter count
and its memory do not grow with the number of steps.

## Installation

```bash
git clone https://github.com/amir-ghz/CASTOR.git
cd CASTOR
pip install -r requirements.txt     # install torch first, matched to your CUDA
python scripts/fetch_data.py        # downloads all datasets into data/
```

Tested with Python 3.10, torch 2.7.0 (CUDA 12.8) and torch-geometric 2.6.1 on
NVIDIA A100 GPUs. `torch-sparse` and `torch-scatter` are not needed.

Datasets are downloaded, not committed: the nine Planetoid, WebKB, Wikipedia
and Actor graphs through PyTorch Geometric, the five Platonov graphs from
their authors' repository. The split files are committed under `splits/`
because they define the evaluation protocol.

## Repository Structure

```
CASTOR/
├── train.py                  # train one dataset under its released configuration
├── castor/
│   ├── block.py              # the CASTOR block: routing kernel and momentum coupling
│   ├── model.py              # encoder, block, optional linear head, softmax
│   ├── train.py              # TrainConfig, the training loop, the per-split runner
│   ├── data.py               # the 16 benchmarks and their split protocols
│   ├── graph.py              # normalised adjacency and the optional curvature gate
│   ├── metrics.py            # accuracy, ROC-AUC and the bootstrap confidence interval
│   └── runner.py             # multi-GPU job pool used by the reproduction script
├── configs/table1/           # one YAML per dataset: the configuration behind its Table 1 entry
├── scripts/
│   ├── fetch_data.py         # download every dataset
│   └── reproduce_table1.py   # retrain all 14 entries and compare with the paper
├── splits/                   # the official 48/32/20 and 60/20/20 split files
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quick Start

Train the released Cornell configuration on the ten official splits:

```bash
python train.py --dataset cornell
```

```
  split  0:  63.89
  split  1:  80.56
  ...
  split  9:  77.78

  acc = 80.00 +/- 4.72 (95% bootstrap CI over 10 splits)
  reported in Table 1: 81.67
```

Any field of the configuration can be overridden on the command line:

```bash
python train.py --dataset roman_empire --gpu 1
python train.py --dataset squirrel --splits 0,1,2
python train.py --dataset texas --set K=5 momentum_mode=off
python train.py --dataset roman_empire --set routing_groups=channel
python train.py --dataset cornell --save-dir runs/cornell
```

Retrain all fourteen Table 1 entries and print each deviation from the paper:

```bash
python scripts/reproduce_table1.py --gpus 0,1,2,3,4,5,6,7
```

Or build a model directly:

```python
from castor import Castor, TrainConfig, load, run_config

ds = load("squirrel")
cfg = TrainConfig(K=10, state_dim=64,            # None: propagate in class-logit space
                  momentum_mode="heavyball",     # 'off' | 'heavyball' | 'nesterov'
                  momentum_init=-0.4,            # the sign is free and learned
                  per_hop_beta=True, constrained_beta=False)
model = cfg.build(ds)                            # nn.Module: (x, W_hat) -> log-probabilities
print(run_config(ds, cfg)["test_mean"])          # mean test accuracy over the ten splits
```

## Model Architecture

One step of the block, for a propagation state `Z` and the symmetric normalised
adjacency with self-loops `W_hat` (Section 4 of the paper):

```
ΔZ     = (I − W_hat) Z                            # smoothing residual (high-pass)
a      = softmax( ρ([ΔZ ‖ Z⁽⁰⁾]) + b_k )          # per-node weights on the 3-simplex
T_k Z  = a₀ · (W_hat Z) + a₁ · ΔZ + a₂ · Z         # routed step
Z⁽ᵏ⁾   = T_k Z⁽ᵏ⁻¹⁾ + β_k (Z⁽ᵏ⁻¹⁾ − Z⁽ᵏ⁻²⁾)        # momentum coupling
```

- **Router `ρ`.** A small MLP shared across nodes and steps, linear or with one
  hidden layer depending on the configuration. It reads the smoothing residual
  together with the initial state `Z⁽⁰⁾`, and a per-step bias `b_k` gives each
  step its own default.
- **Momentum `β_k`.** A learnable scalar, either one per step or one shared
  across steps, either free in sign or sigmoid-constrained, as set in each
  released configuration. Heavy-ball and Nesterov forms are both implemented.
- **Propagation state.** With `state_dim=None` the block propagates in
  class-logit space and its output is the logits. With `state_dim=64` it
  propagates in a 64-channel latent state followed by a linear read-out. The
  released configurations use the class-logit state on Cora, CiteSeer, Texas,
  Wisconsin, Squirrel, Chameleon and Cornell, and the 64-channel state on
  PubMed, Actor and the five Platonov datasets.
- **Routing granularity.** `routing_groups=1` is the router in the paper: three
  weights per node, broadcast across every channel of the state.
  `routing_groups=G` gives each of `G` channel groups its own decision and
  `routing_groups='channel'` gives every channel its own. The paper's results
  use `1`.
- **Curvature gate.** The wide-state configurations enable an optional
  learnable per-edge reweighting of `W_hat` by Balanced Forman curvature
  (`use_bfc`). Its measured contribution is within split-to-split noise; set
  `use_bfc: false` to disable it.

## Results

Node classification on the ten official splits: accuracy, or ROC-AUC on the
binary Platonov targets (Minesweeper, Tolokers, Questions). The first column
is Table 1 of the paper; the second is `scripts/reproduce_table1.py` run from
this repository with the released configurations.

| Dataset | Table 1 | Retrained |  | Dataset | Table 1 | Retrained |
|---|---:|---:|---|---|---:|---:|
| Cora | 88.24 | 88.16 |  | Cornell | 81.67 | 80.00 |
| CiteSeer | 77.70 | 77.88 |  | Roman-empire | 78.39 | 78.39 |
| PubMed | 89.85 | 89.57 |  | Amazon-ratings | 47.41 | 47.30 |
| Texas | 80.28 | 80.28 |  | Minesweeper | 91.70 | 91.70 |
| Wisconsin | 86.67 | 86.54 |  | Tolokers | 80.08 | 80.06 |
| Actor | 36.49 | 36.23 |  | Questions | 76.13 | 76.05 |
| Squirrel | 49.20 | 50.20 |  |  |  |  |
| Chameleon | 65.60 | 64.84 |  |  |  |  |

The retrained numbers fall inside the paper's 95% bootstrap interval on
thirteen datasets and 0.03 outside it on PubMed; the mean absolute deviation
over the fourteen is 0.33 points. Each `configs/table1/*.yaml` records the
reported number and the ten per-split scores it was computed from. Every
split is seeded with its index; the remaining run-to-run variation comes from
non-deterministic CUDA sparse kernels and is well inside the reported intervals.

## Citation

```bibtex
@inproceedings{ghazizadeh2026selftuning,
  title     = {Self-Tuning Graph Filters via State-Dependent Operator Composition},
  author    = {Ghazizadeh, Amir and Alinejad, Mahyar and Atia, George K. and Ewetz, Rickard and Zheng, Hao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgments

The 48/32/20 split protocol follows [BernNet](https://github.com/ivam-he/BernNet);
the mid-scale heterophilous benchmarks and their official splits are from
[Platonov et al. (2023)](https://github.com/yandex-research/heterophilous-graphs).

## License

MIT. See [LICENSE](LICENSE).

## Contact

Amir Ghazizadeh, amir.ghz1995@gmail.com
