from __future__ import annotations
import itertools, time, statistics, math, random, os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict
import csv
from .tsp import TSPInstance
from .aco_base import ACOConfig
from .ant_system import AntSystem
from .acs import AntColonySystem
from .mmas import MaxMinAntSystem

def _algo_constructor(name: str):
    if name.lower() == "as":
        return AntSystem
    if name.lower() == "acs":
        return AntColonySystem
    if name.lower() == "mmas":
        return MaxMinAntSystem
    raise ValueError(f"Unknown algorithm {name}")

def run_repeated_trials(instance: TSPInstance, algo: str, cfg: ACOConfig, n_runs: int = 10, base_seed: int = 42):
    D = instance.distance_matrix()
    lengths = []
    times = []
    best_tours = []
    for r in range(n_runs):
        cfg_r = ACOConfig(**{**asdict(cfg), "seed": base_seed + r})
        algocls = _algo_constructor(algo)
        solver = algocls(D, cfg_r)
        res = solver.run()
        lengths.append(res.best_length)
        times.append(res.elapsed_sec)
        best_tours.append(res.best_tour)
    stats = {
        "mean_length": statistics.mean(lengths),
        "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": statistics.median(lengths),
        "mean_time": statistics.mean(times),
        "algo": algo,
        "n_runs": n_runs,
    }
    return stats, list(zip(lengths, times, best_tours))

def run_parameter_sweep(instance: TSPInstance, algo: str, param_grid: Dict[str, List[Any]], 
                        base_cfg: Optional[ACOConfig] = None, n_runs: int = 5, base_seed: int = 100,
                        csv_path: Optional[str] = None):
    base_cfg = base_cfg or ACOConfig()
    keys = sorted(param_grid.keys())
    rows = []
    for values in itertools.product(*[param_grid[k] for k in keys]):
        cfg_dict = {**{k: getattr(base_cfg, k) for k in asdict(base_cfg).keys()}, **dict(zip(keys, values))}
        cfg = ACOConfig(**cfg_dict)
        stats, _ = run_repeated_trials(instance, algo, cfg, n_runs=n_runs, base_seed=base_seed)
        row = {**{k: getattr(cfg, k) for k in keys}, **stats}
        rows.append(row)
        if csv_path is not None:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    w.writeheader()
                w.writerow(row)
    return rows