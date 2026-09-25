#!/usr/bin/env python
"""Fetch every dataset the paper uses into ``data/``.

Two sources:

* the nine GeomGCN-protocol graphs (Cora, CiteSeer, PubMed, Cornell, Texas,
  Wisconsin, Chameleon, Squirrel, Actor) are downloaded by PyTorch Geometric
  into ``data/pyg/``;
* the five Platonov mid-scale graphs and the two de-duplicated wiki graphs are
  ``.npz`` archives from the authors' repository, written to ``data/platonov/``.

The split files themselves are committed under ``splits/`` -- they are part of
the evaluation protocol rather than raw data, so they are not re-downloaded.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --only platonov
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PLATONOV_BASE = ("https://raw.githubusercontent.com/yandex-research/"
                 "heterophilous-graphs/main/data")
PLATONOV_FILES = [
    "roman_empire.npz", "amazon_ratings.npz", "minesweeper.npz",
    "tolokers.npz", "questions.npz",
    "chameleon_filtered.npz", "squirrel_filtered.npz",
]
PYG_DATASETS = ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin",
                "chameleon", "squirrel", "actor"]


def fetch_platonov(target: str) -> None:
    os.makedirs(target, exist_ok=True)
    for name in PLATONOV_FILES:
        path = os.path.join(target, name)
        if os.path.exists(path):
            print(f"  have {name} ({os.path.getsize(path) / 1e6:.1f} MB)")
            continue
        url = f"{PLATONOV_BASE}/{name}"
        print(f"  downloading {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"{os.path.getsize(path) / 1e6:.1f} MB")
        except Exception as exc:
            if os.path.exists(path):
                os.remove(path)
            print(f"FAILED ({exc})\n    fetch manually from {url}")


def fetch_pyg() -> None:
    from castor.data import load
    for name in PYG_DATASETS:
        try:
            ds = load(name)
            print(f"  {name:<11s} N={ds.num_nodes:<7d} F={ds.num_features:<6d} "
                  f"C={ds.num_classes:<3d} E={ds.edge_index.size(1)}")
        except Exception as exc:
            print(f"  {name:<11s} FAILED: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["platonov", "pyg"], default=None)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    args = ap.parse_args()

    if args.only != "pyg":
        print("Platonov archives -> data/platonov/")
        fetch_platonov(os.path.join(args.data_dir, "platonov"))
    if args.only != "platonov":
        print("\nPyG graphs -> data/pyg/  (downloaded on first load)")
        fetch_pyg()
    print("\ndone")


if __name__ == "__main__":
    main()
