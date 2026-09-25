#!/usr/bin/env python
"""Retrain every Table-1 CASTOR entry from ``configs/table1/`` and compare.

For each of the fourteen datasets this trains the selected configuration on all
ten official splits with the packaged implementation and prints the reproduced
mean next to the value reported in the paper, together with the deviation and
whether it falls inside the reported bootstrap CI.

Results are written to ``runs/table1_reproduction.json``.

Usage::

    python scripts/reproduce_table1.py --gpus 0,1,2,3,4,5,6,7
    python scripts/reproduce_table1.py --datasets cornell,chameleon --gpus 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from castor.data import DISPLAY_NAME, TABLE1_DATASETS          # noqa: E402
from castor.runner import Job, format_result, run_jobs         # noqa: E402

CONFIG_DIR = os.path.join(REPO, "configs", "table1")
OUT_DIR = os.path.join(REPO, "runs")

#: Rough relative cost, used only to schedule the long jobs first.
COST = {
    "questions": 100, "roman_empire": 60, "amazon_ratings": 55, "tolokers": 40,
    "minesweeper": 30, "pubmed": 25, "squirrel": 20, "actor": 15,
    "chameleon": 10, "citeseer": 6, "cora": 6,
    "wisconsin": 2, "texas": 2, "cornell": 2,
}


def load_configs(datasets):
    docs = {}
    for name in datasets:
        path = os.path.join(CONFIG_DIR, f"{name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found")
        with open(path) as fh:
            docs[name] = yaml.safe_load(fh)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=4,
                    help="concurrent training jobs per GPU; these graphs are small, "
                         "so packing several per device raises throughput several-fold")
    ap.add_argument("--datasets", default=",".join(TABLE1_DATASETS))
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent retrainings of each config (>1 estimates run-to-run spread)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "table1_reproduction.json"))
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    docs = load_configs(datasets)

    jobs = []
    for rep in range(args.repeats):
        for name, doc in docs.items():
            jobs.append(Job(
                dataset=name,
                config=doc["config"],
                name=f"table1{'' if args.repeats == 1 else f'/rep{rep}'}",
                tags={"cost": COST.get(name, 10), "repeat": rep,
                      "reported": doc["provenance"]["reported"]},
            ))

    print(f"Reproducing Table 1: {len(jobs)} jobs on GPUs {gpus}\n")
    started = time.time()

    def progress(result, done, total):
        print(f"[{done:3d}/{total}] {format_result(result)}", flush=True)

    results = run_jobs(jobs, gpus, on_result=progress,
                      workers_per_gpu=args.workers_per_gpu)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"results": results, "seconds": time.time() - started}, fh, indent=1)

    # ---- comparison table ------------------------------------------------
    by_dataset = {}
    for r in results:
        if r.get("status") == "ok":
            by_dataset.setdefault(r["dataset"], []).append(r)

    print(f"\n{'dataset':<18s} {'paper':>7s} {'repro':>7s} {'+/-CI':>7s} "
          f"{'delta':>7s}  {'within CI':>9s}")
    print("-" * 62)
    deltas = []
    for name in datasets:
        runs = by_dataset.get(name, [])
        if not runs:
            print(f"{DISPLAY_NAME.get(name, name):<18s} {'':>7s} {'FAILED':>7s}")
            continue
        reported = docs[name]["provenance"]["reported"]
        mean = sum(r["test_mean"] for r in runs) / len(runs)
        ci = sum(r["test_ci95"] for r in runs) / len(runs)
        delta = mean - reported
        deltas.append(delta)
        inside = "yes" if abs(delta) <= ci else "no"
        print(f"{DISPLAY_NAME.get(name, name):<18s} {reported:7.2f} {mean:7.2f} "
              f"{ci:7.2f} {delta:+7.2f}  {inside:>9s}")

    if deltas:
        mad = sum(abs(d) for d in deltas) / len(deltas)
        print("-" * 62)
        print(f"mean |delta| over {len(deltas)} datasets: {mad:.2f} points"
              f"   |   total wall clock {(time.time() - started) / 60:.1f} min")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
