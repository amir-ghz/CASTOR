#!/usr/bin/env python
"""Train CASTOR on one benchmark under its released configuration.

By default this loads ``configs/table1/<dataset>.yaml`` -- the exact recipe
behind that dataset's entry in Table 1 of the paper -- trains it on the ten
official splits, and prints the mean test metric with its 95% bootstrap
confidence interval next to the number reported in the paper.

Examples::

    python train.py --dataset cornell
    python train.py --dataset roman_empire --gpu 1
    python train.py --dataset squirrel --splits 0,1,2
    python train.py --dataset texas --set K=5 momentum_mode=off
    python train.py --dataset roman_empire --set routing_groups=channel
    python train.py --dataset cornell --save-dir runs/cornell   # keep the selected-epoch weights

Any field of :class:`castor.train.TrainConfig` can be overridden with
``--set key=value``; values are parsed as YAML, so ``--set state_dim=null``
switches to the class-logit state and ``--set use_bfc=false`` disables the
curvature gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from castor import DISPLAY_NAME, TABLE1_DATASETS, TrainConfig, load, run_config  # noqa: E402


def parse_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=TABLE1_DATASETS)
    ap.add_argument("--config", default=None,
                    help="YAML recipe; default configs/table1/<dataset>.yaml")
    ap.add_argument("--splits", default=None,
                    help="comma-separated split indices (default: all ten)")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config fields")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save-dir", default=None,
                    help="write the selected-epoch weights of every split here")
    ap.add_argument("--out", default=None, help="write the run summary to this JSON file")
    args = ap.parse_args()

    path = args.config or os.path.join(REPO, "configs", "table1", f"{args.dataset}.yaml")
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    config = dict(doc["config"], **parse_overrides(args.set))
    cfg = TrainConfig.from_dict(config)
    reported = doc.get("provenance", {}).get("reported")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = load(args.dataset)
    splits = [int(s) for s in args.splits.split(",")] if args.splits else None

    print(f"{DISPLAY_NAME.get(args.dataset, args.dataset)}: {dataset}")
    print("config: " + ", ".join(f"{k}={v}" for k, v in config.items()))
    print(f"device: {device}\n")

    t0 = time.time()
    result = run_config(dataset, cfg, splits=splits, device=device,
                        save_dir=args.save_dir, save_tag="castor")
    for i, score in zip(result["splits"], result["test_per_split"]):
        print(f"  split {i:2d}: {score:6.2f}")
    print(f"\n  {result['metric']} = {result['test_mean']:.2f} +/- {result['test_ci95']:.2f} "
          f"(95% bootstrap CI over {len(result['splits'])} splits, {time.time() - t0:.0f}s)")
    if reported is not None and not args.set and splits is None:
        print(f"  reported in Table 1: {reported:.2f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
