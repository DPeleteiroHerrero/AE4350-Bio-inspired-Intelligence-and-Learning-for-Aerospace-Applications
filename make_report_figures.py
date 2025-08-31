#!/usr/bin/env python3
"""
make_report_figures.py

Generates baseline comparison figures/tables and (optionally) parameter grids/sweeps
for AS, ACS, and MMAS on Euclidean TSP instances.

Fixes:
- distance_matrix may be a list -> convert to NumPy in nearest_neighbor_length
- MMAS bounds rebuilt per-rho in sweep_rho
- Fair baseline: same rho across algos, adaptive MMAS bounds from NN length

Update:
- Added consistent, larger font sizes for figures expected to be placed
  side-by-side in LaTeX (alpha-beta grids, rho sweeps, ACS q0/phi grid,
  MMAS bounds sweep, ants-vs-iters tradeoff). Baseline plots keep
  normal font sizes by default, but can be upscaled with a flag.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aco import TSPInstance, ACOConfig, AntSystem, AntColonySystem, MaxMinAntSystem

ALGO_MAP = {"AS": AntSystem, "ACS": AntColonySystem, "MMAS": MaxMinAntSystem}

# ------------------------------- Font helpers -------------------------------- #

# Font sizes for figures you’ll place 3-up in LaTeX
SIDE_TITLE_FS  = 22
SIDE_LABEL_FS  = 20
SIDE_TICK_FS   = 18
SIDE_LEGEND_FS = 18
SIDE_CBAR_FS   = 18

# Font sizes for single, standalone figures (baseline).
SINGLE_TITLE_FS  = 16
SINGLE_LABEL_FS  = 14
SINGLE_TICK_FS   = 12
SINGLE_LEGEND_FS = 12
SINGLE_CBAR_FS   = 12

def apply_fonts(ax, *, side_by_side: bool, legend=None, cbar=None):
    """Apply consistent font sizes to an Axes (and optional legend/colorbar)."""
    if side_by_side:
        ax.set_title(ax.get_title(), fontsize=SIDE_TITLE_FS, pad=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=SIDE_LABEL_FS, labelpad=6)
        ax.set_ylabel(ax.get_ylabel(), fontsize=SIDE_LABEL_FS, labelpad=6)
        ax.tick_params(axis="both", which="both", labelsize=SIDE_TICK_FS, length=4, width=1)
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontsize(SIDE_LEGEND_FS)
        if cbar is not None:
            cbar.ax.tick_params(labelsize=SIDE_CBAR_FS)
            if cbar.ax.get_ylabel():
                cbar.set_label(cbar.ax.get_ylabel(), fontsize=SIDE_CBAR_FS)
    else:
        ax.set_title(ax.get_title(), fontsize=SINGLE_TITLE_FS, pad=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=SINGLE_LABEL_FS, labelpad=4)
        ax.set_ylabel(ax.get_ylabel(), fontsize=SINGLE_LABEL_FS, labelpad=4)
        ax.tick_params(axis="both", which="both", labelsize=SINGLE_TICK_FS, length=3, width=1)
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontsize(SINGLE_LEGEND_FS)
        if cbar is not None:
            cbar.ax.tick_params(labelsize=SINGLE_CBAR_FS)
            if cbar.ax.get_ylabel():
                cbar.set_label(cbar.ax.get_ylabel(), fontsize=SINGLE_CBAR_FS)

# ------------------------------- Utilities ---------------------------------- #

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_list_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def build_cfg(name, ants, iters, alpha=1.0, beta=3.0, rho=0.2, Q=1.0, tau0=None,
              q0=None, phi=None, tau_min=None, tau_max=None, seed=None):
    if name == "AS":
        return ACOConfig(alpha=alpha, beta=beta, rho=rho, Q=Q, tau0=tau0,
                         n_ants=ants, n_iterations=iters, seed=seed)
    if name == "ACS":
        if q0 is None: q0 = 0.9
        if phi is None: phi = 0.1
        return ACOConfig(alpha=alpha, beta=beta, rho=rho, Q=Q, tau0=tau0,
                         q0=q0, phi=phi, n_ants=ants, n_iterations=iters, seed=seed)
    if name == "MMAS":
        if tau_min is None: tau_min = 1e-4
        if tau_max is None: tau_max = 1.0
        return ACOConfig(alpha=alpha, beta=beta, rho=rho, Q=Q, tau0=tau0,
                         n_ants=ants, n_iterations=iters, tau_min=tau_min, tau_max=tau_max, seed=seed)
    raise ValueError(name)

def run_once(instance, algo_name, cfg):
    algo_cls = ALGO_MAP[algo_name]
    solver = algo_cls(instance.distance_matrix(), cfg)
    res = solver.run()
    return {
        "best_length": res.best_length,
        "elapsed_sec": res.elapsed_sec,
        "history": list(solver.history_best_lengths),
    }

def repeated_runs(instance, algo_name, base_cfg, runs=10, base_seed=10_000):
    out = []
    for r in range(runs):
        cfg_r = build_cfg(
            algo_name, base_cfg.n_ants, base_cfg.n_iterations,
            alpha=base_cfg.alpha, beta=base_cfg.beta, rho=base_cfg.rho, Q=base_cfg.Q, tau0=base_cfg.tau0,
            q0=getattr(base_cfg, "q0", None), phi=getattr(base_cfg, "phi", None),
            tau_min=getattr(base_cfg, "tau_min", None), tau_max=getattr(base_cfg, "tau_max", None),
            seed=(base_seed + r)
        )
        out.append(run_once(instance, algo_name, cfg_r))
    return out

def summarize_results(results):
    lengths = [r["best_length"] for r in results]
    times = [r["elapsed_sec"] for r in results]
    return {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths, ddof=1) if len(lengths) > 1 else 0.0),
        "median_length": float(np.median(lengths)),
        "min_length": float(np.min(lengths)),
        "max_length": float(np.max(lengths)),
        "mean_time": float(np.mean(times)),
        "n_runs": len(results),
    }

def nearest_neighbor_length(instance, start=0):
    """Greedy NN tour length for scale-aware MMAS bounds. Robust to list/array input."""
    D = np.asarray(instance.distance_matrix(), dtype=float)
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dist_row = D[cur].copy()
        dist_row[visited] = np.inf
        nxt = int(np.argmin(dist_row))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    L = 0.0
    for i in range(n):
        L += D[tour[i], tour[(i + 1) % n]]
    return L

def scaled_mmas_bounds(instance, rho, a_ratio=20, start=0):
    """Compute tau_max=1/(rho*L_NN), tau_min=tau_max/a."""
    L_nn = nearest_neighbor_length(instance, start=start)
    tau_max = 1.0 / (rho * L_nn)
    tau_min = tau_max / float(a_ratio)
    return tau_min, tau_max

# --------------------------------- Plots ------------------------------------ #

def plot_distribution(results_by_algo, out_png, side_by_side=False):
    fig, ax = plt.subplots()
    algos = list(results_by_algo.keys())
    for i, algo in enumerate(algos, start=1):
        lengths = [r["best_length"] for r in results_by_algo[algo]]
        x = np.random.normal(loc=i, scale=0.03, size=len(lengths))
        ax.plot(x, lengths, "o")
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos)
    ax.set_ylabel("Best tour length")
    ax.set_title("Best lengths across runs")
    apply_fonts(ax, side_by_side=side_by_side)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_boxplot(results_by_algo, out_png, side_by_side=False):
    fig, ax = plt.subplots()
    data = [[r["best_length"] for r in results_by_algo[algo]] for algo in results_by_algo]
    ax.boxplot(data, tick_labels=list(results_by_algo.keys()))
    ax.set_ylabel("Best tour length")
    ax.set_title("Distribution of best tour lengths")
    apply_fonts(ax, side_by_side=side_by_side)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_convergence_mean(results_by_algo, out_png, side_by_side=False):
    fig, ax = plt.subplots()
    for algo, runs in results_by_algo.items():
        H = np.array([r["history"] for r in runs], dtype=float)
        mean = np.mean(H, axis=0)
        iters = np.arange(1, len(mean) + 1)
        ax.plot(iters, mean, label=algo)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far tour length (mean over runs)")
    ax.set_title("Convergence (mean trajectories)")
    leg = ax.legend()
    apply_fonts(ax, side_by_side=side_by_side, legend=leg)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------------- Grids / Sweeps (opt) --------------------------- #

def heatmap_alpha_beta(instance, algo_name, ants, iters, alphas, betas, rho, out_png,
                       runs=3, base_seed=500, tau_min=None, tau_max=None):
    """
    Produce a large, publication-style (alpha, beta) heatmap with bigger fonts
    (intended for 3-up side-by-side with AS/ACS/MMAS).
    """
    # Compute grid of mean best lengths
    M = np.zeros((len(alphas), len(betas)), dtype=float)
    total = len(alphas) * len(betas)
    c = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            c += 1
            print(f"[alphabeta:{algo_name}] {c}/{total} -> alpha={a}, beta={b} "
                  f"(runs={runs}, iters={iters})", flush=True)
            if algo_name == "MMAS":
                cfg = build_cfg(algo_name, ants, iters, alpha=a, beta=b, rho=rho,
                                tau_min=tau_min, tau_max=tau_max)
            elif algo_name == "ACS":
                cfg = build_cfg(algo_name, ants, iters, alpha=a, beta=b, rho=rho,
                                q0=0.9, phi=0.1)
            else:
                cfg = build_cfg(algo_name, ants, iters, alpha=a, beta=b, rho=rho)
            results = repeated_runs(instance, algo_name, cfg, runs=runs, base_seed=base_seed)
            s = summarize_results(results)
            M[i, j] = s["mean_length"]

    # Figure with bigger fonts for side-by-side readability
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(M, origin="lower", aspect="auto")

    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([str(a) for a in alphas])

    ax.set_xlabel("β (heuristic exponent)")
    ax.set_ylabel("α (pheromone exponent)")
    ax.set_title(f"{algo_name}: mean best tour length vs (α, β)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean best length")

    # Apply big side-by-side fonts
    apply_fonts(ax, side_by_side=True, cbar=cbar)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return M

def sweep_rho(instance, algo_name, ants, iters, rhos, alpha, beta, out_png, runs=3, base_seed=700,
              tau_min=None, tau_max=None, a_ratio_default=20):
    # Evaluate mean best length vs rho
    means = []
    print(f"[rho:{algo_name}] rhos={rhos}, runs={runs}, iters={iters}", flush=True)
    for rho in rhos:
        if algo_name == "MMAS":
            # If bounds not passed, rebuild scale-aware bounds for this rho
            if tau_min is None or tau_max is None:
                tmin, tmax = scaled_mmas_bounds(instance, rho, a_ratio=a_ratio_default, start=0)
            else:
                tmin, tmax = tau_min, tau_max
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho,
                            tau_min=tmin, tau_max=tmax)
        elif algo_name == "ACS":
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho, q0=0.9, phi=0.1)
        else:
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho)
        results = repeated_runs(instance, algo_name, cfg, runs=runs, base_seed=base_seed)
        s = summarize_results(results)
        means.append(s["mean_length"])

    fig, ax = plt.subplots()
    ax.plot(rhos, means, "o-")
    ax.set_xlabel("ρ (evaporation)")
    ax.set_ylabel("Mean best length")
    ax.set_title(f"{algo_name}: effect of evaporation")

    # Big fonts for side-by-side
    apply_fonts(ax, side_by_side=True)

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return np.array(means)

def heatmap_acs_q0_phi(instance, ants, iters, q0_list, phi_list, alpha, beta, rho, out_png, runs=3, base_seed=900):
    M = np.zeros((len(q0_list), len(phi_list)), dtype=float)
    total = len(q0_list) * len(phi_list)
    c = 0
    for i, q0 in enumerate(q0_list):
        for j, phi in enumerate(phi_list):
            c += 1
            print(f"[acs q0-phi] {c}/{total} -> q0={q0}, phi={phi} (runs={runs}, iters={iters})", flush=True)
            cfg = build_cfg("ACS", ants, iters, alpha=alpha, beta=beta, rho=rho, q0=q0, phi=phi)
            results = repeated_runs(instance, "ACS", cfg, runs=runs, base_seed=base_seed)
            s = summarize_results(results)
            M[i, j] = s["mean_length"]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(M, origin="lower", aspect="auto")
    ax.set_xticks(range(len(phi_list)))
    ax.set_xticklabels([str(x) for x in phi_list])
    ax.set_yticks(range(len(q0_list)))
    ax.set_yticklabels([str(x) for x in q0_list])
    ax.set_xlabel("φ (local update)")
    ax.set_ylabel("q₀ (exploitation prob.)")
    ax.set_title("ACS: mean best length vs (q₀, φ)")
    cbar = fig.colorbar(im, label="Mean best length")
    # Big fonts for side-by-side
    apply_fonts(ax, side_by_side=True, cbar=cbar)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return M

def sweep_mmas_bounds(instance, ants, iters, alpha, beta, rho, a_list, out_png, runs=3, base_seed=1100):
    """Vary a = tau_max / tau_min with tau_max = 1/(rho * L_NN)."""
    tmin_base, tmax_base = scaled_mmas_bounds(instance, rho, a_ratio=20, start=0)
    L_means = []
    print(f"[mmas bounds] a_list={a_list}, runs={runs}, iters={iters} (tau_max from NN length)", flush=True)
    for a in a_list:
        tau_max = tmax_base
        tau_min = tau_max / float(a)
        cfg = build_cfg("MMAS", ants, iters, alpha=alpha, beta=beta, rho=rho, tau_min=tau_min, tau_max=tau_max)
        results = repeated_runs(instance, "MMAS", cfg, runs=runs, base_seed=base_seed)
        s = summarize_results(results)
        L_means.append(s["mean_length"])

    fig, ax = plt.subplots()
    ax.plot(a_list, L_means, "o-")
    ax.set_xlabel("a (τ_max / τ_min)")
    ax.set_ylabel("Mean best length")
    ax.set_title("MMAS: effect of pheromone bounds")
    # Big fonts for side-by-side
    apply_fonts(ax, side_by_side=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return np.array(L_means)

def ants_iters_tradeoff(instance, algo_name, budget, ants_list, alpha, beta, rho, out_png, runs=3, base_seed=1300,
                        tau_min=None, tau_max=None):
    means = []
    print(f"[tradeoff:{algo_name}] ants_list={ants_list}, budget={budget}, runs={runs}", flush=True)
    for ants in ants_list:
        iters = max(1, budget // ants)
        if algo_name == "MMAS":
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho,
                            tau_min=tau_min, tau_max=tau_max)
        elif algo_name == "ACS":
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho, q0=0.9, phi=0.1)
        else:
            cfg = build_cfg(algo_name, ants, iters, alpha=alpha, beta=beta, rho=rho)
        results = repeated_runs(instance, algo_name, cfg, runs=runs, base_seed=base_seed)
        s = summarize_results(results)
        means.append(s["mean_length"])

    fig, ax = plt.subplots()
    ax.plot(ants_list, means, "o-")
    ax.set_xlabel("Number of ants (iterations adjusted to keep budget)")
    ax.set_ylabel("Mean best length")
    ax.set_title(f"{algo_name}: ants vs iterations (fixed budget={budget})")
    # Big fonts for side-by-side
    apply_fonts(ax, side_by_side=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return np.array(means)

# ------------------------------ Table export -------------------------------- #

def make_baseline_table(results_by_algo, out_csv, out_tex):
    rows = []
    for algo, runs in results_by_algo.items():
        s = summarize_results(runs)
        s["algo"] = algo
        rows.append(s)
    df = pd.DataFrame(rows).set_index("algo")
    df.to_csv(out_csv)
    try:
        df.to_latex(out_tex, float_format="%.3f")
    except Exception:
        pass
    return df

# ----------------------------------- Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--square", type=float, default=100.0)
    ap.add_argument("--runs", type=int, default=10, help="seeds for baseline")
    ap.add_argument("--ants", type=int, default=25)
    ap.add_argument("--iters", type=int, default=150)

    # Grid/sweep controls
    ap.add_argument("--grid-runs", type=int, default=3, help="seeds per cell for grids/sweeps")
    ap.add_argument("--grid-iters", type=int, default=0, help="iterations for grids (0 => iters//2)")
    ap.add_argument("--alphas", default="0.5,1.0,1.5")
    ap.add_argument("--betas",  default="2,3,4")
    ap.add_argument("--rhos",   default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--acs-q0", default="0.85,0.9,0.95")
    ap.add_argument("--acs-phi", default="0.05,0.1,0.2")
    ap.add_argument("--mmas-a", default="20,50,100",
                    help="ratios for MMAS sweeps (a = tau_max / tau_min)")
    ap.add_argument("--mmas-a-baseline", type=int, default=20,
                    help="MMAS ratio used in baseline (tau_min = tau_max / a)")

    # Skips
    ap.add_argument("--skip-alphabeta", action="store_true")
    ap.add_argument("--skip-rho", action="store_true")
    ap.add_argument("--skip-acs", action="store_true")
    ap.add_argument("--skip-mmas", action="store_true")
    ap.add_argument("--skip-tradeoff", action="store_true")

    # Fast profile
    ap.add_argument("--fast", action="store_true",
                    help="shortcut: grid-runs=2, grid-iters=min(60, iters)")

    # If you also want baseline plots larger for a single wide figure, set this flag.
    ap.add_argument("--big-baseline", action="store_true",
                    help="also use big fonts for baseline plots (normally single-figure sized)")

    ap.add_argument("--outdir", default="report_figures")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    grid_iters = args.grid_iters if args.grid_iters > 0 else max(1, args.iters // 2)
    if args.fast:
        args.grid_runs = 2
        grid_iters = min(60, args.iters)

    alphas = parse_list_floats(args.alphas)
    betas  = parse_list_floats(args.betas)
    rhos   = parse_list_floats(args.rhos)
    q0_list  = parse_list_floats(args.acs_q0)
    phi_list = parse_list_floats(args.acs_phi)
    a_list   = parse_list_ints(args.mmas_a)

    # Instance
    inst = TSPInstance.random_euclidean(n=args.n, seed=123, square_size=args.square, name=f"report{args.n}")

    # Fair evaporation for baseline
    rho_common = 0.2

    # Adaptive MMAS bounds from NN tour length
    tau_min_base, tau_max_base = scaled_mmas_bounds(inst, rho_common, a_ratio=args.mmas_a_baseline, start=0)

    # Baseline configs (same ants/iters across algos)
    cfg_AS   = build_cfg("AS",   args.ants, args.iters, alpha=1.0, beta=3.0, rho=rho_common)
    cfg_ACS  = build_cfg("ACS",  args.ants, args.iters, alpha=1.0, beta=3.0, rho=rho_common, q0=0.9, phi=0.1)
    cfg_MMAS = build_cfg("MMAS", args.ants, args.iters, alpha=1.0, beta=3.0, rho=rho_common,
                         tau_min=tau_min_base, tau_max=tau_max_base)

    print("Running baseline experiments...", flush=True)
    res_AS   = repeated_runs(inst, "AS",   cfg_AS, runs=args.runs, base_seed=1000)
    res_ACS  = repeated_runs(inst, "ACS",  cfg_ACS, runs=args.runs, base_seed=2000)
    res_MMAS = repeated_runs(inst, "MMAS", cfg_MMAS, runs=args.runs, base_seed=3000)
    results_by_algo = {"AS": res_AS, "ACS": res_ACS, "MMAS": res_MMAS}

    # Baseline plots/tables (normal fonts unless --big-baseline is set)
    plot_distribution(results_by_algo, os.path.join(args.outdir, "baseline_distribution.png"),
                      side_by_side=args.big_baseline)
    plot_boxplot(results_by_algo, os.path.join(args.outdir, "baseline_boxplot.png"),
                 side_by_side=args.big_baseline)
    plot_convergence_mean(results_by_algo, os.path.join(args.outdir, "baseline_convergence_mean.png"),
                          side_by_side=args.big_baseline)
    df_base = make_baseline_table(results_by_algo,
                                  os.path.join(args.outdir, "baseline_table.csv"),
                                  os.path.join(args.outdir, "baseline_table.tex"))
    print("Baseline summary:")
    print(df_base)

    # Alpha-Beta grids (3-up AS/ACS/MMAS) -> big fonts
    if not args.skip_alphabeta:
        print(f"Alpha-Beta grids with grid_runs={args.grid_runs}, grid_iters={grid_iters}", flush=True)
        heatmap_alpha_beta(inst, "AS",   args.ants, grid_iters, alphas, betas, rho=rho_common,
                           out_png=os.path.join(args.outdir, "alphabeta_AS.png"), runs=args.grid_runs)
        heatmap_alpha_beta(inst, "ACS",  args.ants, grid_iters, alphas, betas, rho=rho_common,
                           out_png=os.path.join(args.outdir, "alphabeta_ACS.png"), runs=args.grid_runs)
        heatmap_alpha_beta(inst, "MMAS", args.ants, grid_iters, alphas, betas, rho=rho_common,
                           out_png=os.path.join(args.outdir, "alphabeta_MMAS.png"), runs=args.grid_runs,
                           tau_min=tau_min_base, tau_max=tau_max_base)
    else:
        print("Skipping alpha-beta grids.", flush=True)

    # Rho sweeps (3-up AS/ACS/MMAS) -> big fonts
    if not args.skip_rho:
        print(f"Rho sweeps with grid_runs={args.grid_runs}, grid_iters={grid_iters}", flush=True)
        sweep_rho(inst, "AS",   args.ants, grid_iters, rhos, alpha=1.0, beta=3.0,
                  out_png=os.path.join(args.outdir, "rho_AS.png"), runs=args.grid_runs)
        sweep_rho(inst, "ACS",  args.ants, grid_iters, rhos, alpha=1.0, beta=3.0,
                  out_png=os.path.join(args.outdir, "rho_ACS.png"), runs=args.grid_runs)
        # MMAS: rebuild bounds per rho from L_NN
        sweep_rho(inst, "MMAS", args.ants, grid_iters, rhos, alpha=1.0, beta=3.0,
                  out_png=os.path.join(args.outdir, "rho_MMAS.png"), runs=args.grid_runs,
                  tau_min=None, tau_max=None)
    else:
        print("Skipping rho sweeps.", flush=True)

    # ACS q0/phi (single, but likely placed in triptych with other algos/results) -> big fonts
    if not args.skip_acs:
        print(f"ACS q0/phi grid with grid_runs={args.grid_runs}, grid_iters={grid_iters}", flush=True)
        heatmap_acs_q0_phi(inst, args.ants, grid_iters, q0_list, phi_list,
                           alpha=1.0, beta=3.0, rho=rho_common,
                           out_png=os.path.join(args.outdir, "acs_q0_phi.png"),
                           runs=args.grid_runs)
    else:
        print("Skipping ACS q0/phi.", flush=True)

    # MMAS bounds sweep (single, often shown with others) -> big fonts
    if not args.skip_mmas:
        print(f"MMAS bounds sweep with grid_runs={args.grid_runs}, grid_iters={grid_iters}", flush=True)
        sweep_mmas_bounds(inst, args.ants, grid_iters, alpha=1.0, beta=3.0, rho=rho_common,
                          a_list=a_list, out_png=os.path.join(args.outdir, "mmas_bounds.png"),
                          runs=args.grid_runs)
    else:
        print("Skipping MMAS bounds.", flush=True)

    # Tradeoff (3-up AS/ACS/MMAS) -> big fonts
    if not args.skip_tradeoff:
        budget = args.ants * args.iters
        ants_list = [10, 15, 20, 25, 30, 40]
        print(f"Ants vs iterations tradeoff with grid_runs={args.grid_runs} (budget={budget})", flush=True)
        ants_iters_tradeoff(inst, "AS",   budget, ants_list, alpha=1.0, beta=3.0, rho=rho_common,
                            out_png=os.path.join(args.outdir, "tradeoff_AS.png"),
                            runs=args.grid_runs)
        ants_iters_tradeoff(inst, "ACS",  budget, ants_list, alpha=1.0, beta=3.0, rho=rho_common,
                            out_png=os.path.join(args.outdir, "tradeoff_ACS.png"),
                            runs=args.grid_runs)
        ants_iters_tradeoff(inst, "MMAS", budget, ants_list, alpha=1.0, beta=3.0, rho=rho_common,
                            out_png=os.path.join(args.outdir, "tradeoff_MMAS.png"),
                            runs=args.grid_runs,
                            tau_min=tau_min_base, tau_max=tau_max_base)
    else:
        print("Skipping ants-iterations tradeoff.", flush=True)

    # Index file
    index = {
        "baseline": {
            "distribution_png": str(Path(args.outdir, "baseline_distribution.png")),
            "boxplot_png": str(Path(args.outdir, "baseline_boxplot.png")),
            "convergence_mean_png": str(Path(args.outdir, "baseline_convergence_mean.png")),
            "table_csv": str(Path(args.outdir, "baseline_table.csv")),
            "table_tex": str(Path(args.outdir, "baseline_table.tex")),
        },
        "alphabeta": {
            "AS": str(Path(args.outdir, "alphabeta_AS.png")),
            "ACS": str(Path(args.outdir, "alphabeta_ACS.png")),
            "MMAS": str(Path(args.outdir, "alphabeta_MMAS.png")),
        },
        "rho_sweeps": {
            "AS": str(Path(args.outdir, "rho_AS.png")),
            "ACS": str(Path(args.outdir, "rho_ACS.png")),
            "MMAS": str(Path(args.outdir, "rho_MMAS.png")),
        },
        "acs": {
            "q0_phi_png": str(Path(args.outdir, "acs_q0_phi.png")),
        },
        "mmas": {
            "bounds_png": str(Path(args.outdir, "mmas_bounds.png")),
        },
        "tradeoff": {
            "AS": str(Path(args.outdir, "tradeoff_AS.png")),
            "ACS": str(Path(args.outdir, "tradeoff_ACS.png")),
            "MMAS": str(Path(args.outdir, "tradeoff_MMAS.png")),
        }
    }
    with open(Path(args.outdir, "figure_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print("All figures written to:", args.outdir)

if __name__ == "__main__":
    main()
    