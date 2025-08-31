# run_experiments.py
import os, json, argparse, tempfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from aco import TSPInstance, ACOConfig, AntSystem, AntColonySystem, MaxMinAntSystem
from aco.experiments import run_repeated_trials, run_parameter_sweep

OUTDIR = os.path.dirname(__file__)
ALGO_MAP = {"AS": AntSystem, "ACS": AntColonySystem, "MMAS": MaxMinAntSystem}


def ensure(path: str) -> str:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def build_config(name, n_ants=25, n_iterations=150):
    if name == "AS":
        return ACOConfig(alpha=1.0, beta=3.0, rho=0.5, Q=1.0,
                         n_ants=n_ants, n_iterations=n_iterations)
    if name == "ACS":
        return ACOConfig(alpha=1.0, beta=2.0, rho=0.1, Q=1.0, q0=0.9, phi=0.1,
                         n_ants=n_ants, n_iterations=n_iterations)
    if name == "MMAS":
        return ACOConfig(alpha=1.0, beta=3.0, rho=0.2, Q=1.0,
                         n_ants=n_ants, n_iterations=n_iterations,
                         tau_min=1e-4, tau_max=1.0)
    raise ValueError(name)


def plot_scatter(details_by_algo, save_path):
    plt.figure()
    algos = list(details_by_algo.keys())
    for i, algo in enumerate(algos, start=1):
        lengths = [L for (L, t, tour) in details_by_algo[algo]]
        x = np.random.normal(loc=i, scale=0.03, size=len(lengths))
        plt.plot(x, lengths, "o")
    plt.xticks(range(1, len(algos) + 1), algos)
    plt.ylabel("Best tour length")
    plt.title("Best lengths across runs")
    ensure(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_convergence(inst, algo_name, cfg, save_path):
    algo_cls = ALGO_MAP[algo_name]
    solver = algo_cls(inst.distance_matrix(), cfg)
    _ = solver.run()
    plt.figure()
    plt.plot(solver.history_best_lengths)
    plt.xlabel("Iteration")
    plt.ylabel("Best-so-far tour length")
    plt.title(f"{algo_name} convergence")
    ensure(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_gif(inst, algo_name, cfg, save_gif, step=5, frames_dir=None, keep_frames=False):
    """Render a convergence GIF. By default, writes frames to a temp folder and deletes them."""
    algo_cls = ALGO_MAP[algo_name]
    solver = algo_cls(inst.distance_matrix(), cfg)
    _ = solver.run()
    coords = inst.coords

    # choose where to put frames
    tmpdir_was_auto = False
    if frames_dir is None:
        frames_dir = tempfile.mkdtemp(prefix=f"{algo_name}_frames_")
        tmpdir_was_auto = True
    else:
        os.makedirs(frames_dir, exist_ok=True)

    frames = []
    iters = list(range(0, len(solver.history_best_tours), step))
    for it in iters:
        tour = solver.history_best_tours[it]
        L = solver.history_best_lengths[it]
        xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
        ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
        cx = [c[0] for c in coords]
        cy = [c[1] for c in coords]

        plt.figure(figsize=(5, 5))
        plt.plot(cx, cy, "o")
        plt.plot(xs, ys, "-")
        plt.title(f"{algo_name} best-so-far\niter={it+1} length={L:.2f}")
        plt.axis("equal")
        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"{algo_name}_{it:03d}.png")
        plt.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close()
        frames.append(frame_path)

    ensure(save_gif)
    with imageio.get_writer(save_gif, mode="I", duration=0.6) as w:
        for fp in frames:
            w.append_data(imageio.v2.imread(fp))

    # clean frames
    if not keep_frames and tmpdir_was_auto:
        shutil.rmtree(frames_dir, ignore_errors=True)
    elif keep_frames:
        print("Frames saved in:", frames_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--square", type=float, default=100.0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--ants", type=int, default=25)
    ap.add_argument("--visualize", action="store_true", help="save a convergence GIF per algorithm")
    ap.add_argument("--keep-frames", action="store_true", help="keep the PNG frames used for the GIF(s)")
    ap.add_argument("--frames-dir", default=None, help="where to store frames (if keeping them)")
    args = ap.parse_args()

    inst = TSPInstance.random_euclidean(n=args.n, seed=123, square_size=args.square, name=f"demo{args.n}")
    algos = ["AS", "ACS", "MMAS"]
    configs = [(name, build_config(name, n_ants=args.ants, n_iterations=args.iters)) for name in algos]

    # repeated trials
    records = []
    details_by_algo = {}
    for name, cfg in configs:
        stats, details = run_repeated_trials(inst, name, cfg, n_runs=args.runs)
        print(name, json.dumps(stats, indent=2))
        records.append({"algo": name, **stats})
        details_by_algo[name] = details

    # summary CSV + scatter plot
    df_summary = pd.DataFrame.from_records(records)
    summary_csv = os.path.join(OUTDIR, "results_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    scatter_png = os.path.join(OUTDIR, "results_distribution.png")
    plot_scatter(details_by_algo, scatter_png)

    # convergence plots (per algorithm)
    for name, cfg in configs:
        conv_png = os.path.join(OUTDIR, f"convergence_{name}.png")
        plot_convergence(inst, name, cfg, conv_png)

    # parameter sweep (MMAS example)
    grid = {"alpha": [0.5, 1.0, 1.5], "beta": [2.0, 3.0, 4.0], "rho": [0.1, 0.3]}
    rows = run_parameter_sweep(
        inst, "MMAS", grid, base_cfg=configs[-1][1],
        n_runs=3, base_seed=500, csv_path=os.path.join(OUTDIR, "mmas_grid.csv")
    )
    print("Grid search evaluated:", len(rows))

    # optional GIFs per algorithm (frames auto-cleaned unless --keep-frames)
    if args.visualize:
        for name, cfg in configs:
            gif_path = os.path.join(OUTDIR, f"{name}_convergence.gif")
            make_gif(inst, name, cfg, gif_path, step=5,
                     frames_dir=args.frames_dir, keep_frames=args.keep_frames)
            print("Saved GIF:", gif_path)


if __name__ == "__main__":
    main()
