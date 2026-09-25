"""Multi-GPU job runner shared by every experiment script.

A *job* is one ``(dataset, TrainConfig, splits)`` triple.  Jobs are handed to a
pool of worker processes, one per GPU; each worker pins itself to its GPU via
``CUDA_VISIBLE_DEVICES`` before importing torch, loads each dataset once, and
caches the normalised operator (and curvature, when needed) across all jobs that
share a dataset.

Everything is plain ``multiprocessing`` with the ``spawn`` start method, so a
crashed job takes down one worker's task rather than the run, and results stream
back as they finish.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

_WORKER_GPU: Optional[int] = None
_CACHES: Dict[tuple, Any] = {}


@dataclass
class Job:
    """One config to train on one dataset."""

    dataset: str
    config: Dict[str, Any]           # a TrainConfig as a plain dict, so it pickles
    name: str = ""
    splits: Optional[Sequence[int]] = None
    protocol: Optional[str] = None   # split protocol for the PyG graphs
    tags: Dict[str, Any] = field(default_factory=dict)   # copied into the result
    save_dir: Optional[str] = None   # write selected-epoch weights per split here


def _init_worker(gpu_queue) -> None:
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    torch.set_num_threads(2)
    global _WORKER_GPU
    _WORKER_GPU = gpu


def _get_cache(dataset: str, protocol: Optional[str], need_curvature: bool):
    """Per-worker dataset cache, keyed by (dataset, protocol)."""
    import torch

    from .data import load
    from .train import GraphCache

    key = (dataset, protocol)
    cache = _CACHES.get(key)
    if cache is None:
        ds = load(dataset, protocol=protocol)
        cache = GraphCache(ds, torch.device("cuda:0"), need_curvature=need_curvature)
        _CACHES[key] = cache
    if need_curvature:
        cache.ensure_curvature()
    return cache


def _run_job(job: Job) -> Dict[str, Any]:
    from .train import TrainConfig, run_config

    started = time.time()
    try:
        cfg = TrainConfig.from_dict(job.config)
        cache = _get_cache(job.dataset, job.protocol, cfg.use_bfc)
        tag = job.name.replace("/", "-").replace(" ", "_")
        result = run_config(cache.dataset, cfg, splits=job.splits,
                            device=cache.device, cache=cache, name=job.name,
                            save_dir=job.save_dir, save_tag=tag)
        result["gpu"] = _WORKER_GPU
        result["status"] = "ok"
        result.update(job.tags)
        return result
    except Exception as exc:                                   # keep the pool alive
        return {
            "name": job.name,
            "dataset": job.dataset,
            "config": job.config,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gpu": _WORKER_GPU,
            "seconds": time.time() - started,
            **job.tags,
        }


def run_jobs(jobs: Sequence[Job], gpus: Sequence[int],
             on_result: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
             workers_per_gpu: int = 1) -> List[Dict[str, Any]]:
    """Run every job across ``gpus``; call ``on_result`` as each finishes.

    Jobs are sorted longest-first (by ``tags['cost']`` when present) so the tail
    of the run is not one slow job finishing alone.
    """
    jobs = sorted(jobs, key=lambda j: -float(j.tags.get("cost", 0)))
    slots = [g for g in gpus for _ in range(workers_per_gpu)]

    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue()
    for slot in slots:
        queue.put(slot)

    results: List[Dict[str, Any]] = []
    total = len(jobs)
    with ctx.Pool(processes=len(slots), initializer=_init_worker, initargs=(queue,)) as pool:
        for result in pool.imap_unordered(_run_job, jobs):
            results.append(result)
            if on_result is not None:
                on_result(result, len(results), total)
    return results


def format_result(result: Dict[str, Any]) -> str:
    """One console line per finished job."""
    if result.get("status") == "error":
        return (f"  [FAIL] {result['dataset']:<20s} {result.get('name', ''):<34s} "
                f"{result['error']}")
    return (f"  {result['dataset']:<20s} {result.get('name', ''):<34s} "
            f"{result['test_mean']:6.2f} +/-{result['test_ci95']:5.2f} "
            f"({result['metric']}, {result['seconds']:.0f}s, gpu{result.get('gpu')})")
