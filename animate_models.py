# animate_models.py
# Visual animations of ACO variants (AS, ACS, MMAS) with few ants,
# now with explicit seed control, multi-run selection, fixed top margins,
# and final snapshots (PNG) of the best tours.
#
# Usage (run from your project root, where `aco/` is importable):
#   python animate_models.py --side-by-side --separate --inst-seed 2025
#   python animate_models.py --side-by-side --repeats 3 --pick worst-start
#   python animate_models.py --side-by-side --inst-seed 123 --algo-seed-base 1000
#
# Tips:
#   --step 2   # fewer frames (lighter GIF)
#   --n 40 --iters 80  # larger instance/longer animation
#
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio

from aco import TSPInstance, ACOConfig, AntSystem, AntColonySystem, MaxMinAntSystem

ALGO_MAP = {"AS": AntSystem, "ACS": AntColonySystem, "MMAS": MaxMinAntSystem}


def build_cfg(
    name,
    ants,
    iters,
    alpha=1.0,
    beta=3.0,
    rho=0.2,
    Q=1.0,
    tau0=None,
    q0=None,
    phi=None,
    tau_min=None,
    tau_max=None,
    seed=None,
):
    if name == "AS":
        return ACOConfig(
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            tau0=tau0,
            n_ants=ants,
            n_iterations=iters,
            seed=seed,
        )
    if name == "ACS":
        if q0 is None:
            q0 = 0.9
        if phi is None:
            phi = 0.1
        return ACOConfig(
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            tau0=tau0,
            q0=q0,
            phi=phi,
            n_ants=ants,
            n_iterations=iters,
            seed=seed,
        )
    if name == "MMAS":
        if tau_min is None:
            tau_min = 1e-4
        if tau_max is None:
            tau_max = 1.0
        return ACOConfig(
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            tau0=tau0,
            n_ants=ants,
            n_iterations=iters,
            tau_min=tau_min,
            tau_max=tau_max,
            seed=seed,
        )
    raise ValueError(name)


def run_solver(instance, algo_name, cfg):
    algo_cls = ALGO_MAP[algo_name]
    solver = algo_cls(instance.distance_matrix(), cfg)
    res = solver.run()
    return solver, res


def tour_to_xy(coords, tour):
    xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
    return xs, ys


def draw_panel(ax, coords, tour, title, subtitle):
    cx = [c[0] for c in coords]
    cy = [c[1] for c in coords]
    ax.plot(cx, cy, "o")  # cities
    if tour is not None and len(tour) > 1:
        xs, ys = tour_to_xy(coords, tour)
        ax.plot(xs, ys, "-")  # current best tour
    ax.set_title(title + "\n" + subtitle, pad=10)  # add padding to avoid clipping
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def make_side_by_side_gif(instance, histories, out_gif, step=1):
    coords = instance.coords
    names = list(histories.keys())
    iters = min(len(histories[n]["lengths"]) for n in names)

    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_gif, mode="I", duration=0.6) as writer:
        for it in range(0, iters, step):
            # Slightly taller and reserve top margin so titles are not cut
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
            for idx, name in enumerate(names):
                L = histories[name]["lengths"][it]
                tour = histories[name]["tours"][it]
                title = f"{name} — iter {it+1}/{iters}"
                subtitle = f"best length = {L:.2f}"
                draw_panel(axes[idx], coords, tour, title, subtitle)
            fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave ~6% top margin
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                h, w, 3
            )
            writer.append_data(frame)
            plt.close(fig)


def make_single_gif(instance, name, history, out_gif, step=1):
    coords = instance.coords
    iters = len(history["lengths"])
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_gif, mode="I", duration=0.6) as writer:
        for it in range(0, iters, step):
            fig = plt.figure(figsize=(5.8, 5.8))
            L = history["lengths"][it]
            tour = history["tours"][it]
            title = f"{name} — iter {it+1}/{iters}"
            subtitle = f"best length = {L:.2f}"
            ax = plt.gca()
            draw_panel(ax, coords, tour, title, subtitle)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                h, w, 3
            )
            writer.append_data(frame)
            plt.close(fig)


def save_side_by_side_final(instance, histories, out_png):
    """Save a single PNG showing the final best tours for AS, ACS, MMAS."""
    coords = instance.coords
    names = list(histories.keys())
    iters = min(len(histories[n]["lengths"]) for n in names)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for idx, name in enumerate(names):
        L = histories[name]["lengths"][iters - 1]
        tour = histories[name]["tours"][iters - 1]
        title = f"{name} — final"
        subtitle = f"best length = {L:.2f}"
        draw_panel(axes[idx], coords, tour, title, subtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_single_final(instance, name, history, out_png):
    """Save a single PNG showing the final best tour for one algorithm."""
    coords = instance.coords
    iters = len(history["lengths"])
    fig = plt.figure(figsize=(5.8, 5.8))
    L = history["lengths"][iters - 1]
    tour = history["tours"][iters - 1]
    title = f"{name} — final"
    subtitle = f"best length = {L:.2f}"
    ax = plt.gca()
    draw_panel(ax, coords, tour, title, subtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def score_run_start(histories):
    """
    Score a run by how 'good' the very first best-so-far looks.
    We take the min across algorithms of their iteration-1 best length.
    A *smaller* min => someone started very well (lucky). For 'worst-start', we want this min to be large.
    """
    first_lengths = [histories[name]["lengths"][0] for name in histories]
    return float(np.min(first_lengths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of cities")
    ap.add_argument("--square", type=float, default=100.0, help="square size for coords")
    ap.add_argument("--ants", type=int, default=6, help="few ants for clarity")
    ap.add_argument("--iters", type=int, default=60, help="iterations to animate")
    ap.add_argument("--step", type=int, default=1, help="frame step (e.g., 2 halves frames)")
    ap.add_argument("--outdir", default="animations")
    ap.add_argument("--inst-seed", type=int, default=None, help="seed for coordinates")
    ap.add_argument(
        "--algo-seed-base",
        type=int,
        default=None,
        help="base seed for algorithm RNGs; offsets per algo",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how many different instances to try; pick one by --pick",
    )
    ap.add_argument(
        "--pick",
        choices=["first", "worst-start", "best-start"],
        default="first",
        help="which candidate to keep after repeats",
    )
    ap.add_argument("--side-by-side", action="store_true", help="3-panel GIF")
    ap.add_argument("--separate", action="store_true", help="one GIF per algorithm")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build instance seeds
    if args.inst_seed is None:
        rng = np.random.default_rng()
        seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(max(1, args.repeats))]
    else:
        seeds = [int(args.inst_seed + r) for r in range(max(1, args.repeats))]

    candidates = []
    for idx, s in enumerate(seeds, start=1):
        print(f"[run {idx}/{len(seeds)}] inst_seed={s}")
        inst = TSPInstance.random_euclidean(
            n=args.n, seed=s, square_size=args.square, name=f"anim{args.n}"
        )

        # Per-algorithm seeds (optional)
        if args.algo_seed_base is not None:
            seeds_algo = {
                "AS": args.algo_seed_base + 10,
                "ACS": args.algo_seed_base + 20,
                "MMAS": args.algo_seed_base + 30,
            }
        else:
            seeds_algo = {"AS": None, "ACS": None, "MMAS": None}

        cfgs = {
            "AS": build_cfg(
                "AS",
                args.ants,
                args.iters,
                alpha=1.0,
                beta=3.0,
                rho=0.5,  # slightly higher rho to keep movement visible
                seed=seeds_algo["AS"],
            ),
            "ACS": build_cfg(
                "ACS",
                args.ants,
                args.iters,
                alpha=1.0,
                beta=3.0,
                rho=0.1,
                q0=0.9,
                phi=0.1,
                seed=seeds_algo["ACS"],
            ),
            "MMAS": build_cfg(
                "MMAS",
                args.ants,
                args.iters,
                alpha=1.0,
                beta=3.0,
                rho=0.2,
                tau_min=1e-4,
                tau_max=1.0,
                seed=seeds_algo["MMAS"],
            ),
        }

        histories = {}
        for name, cfg in cfgs.items():
            solver, _ = run_solver(inst, name, cfg)
            lengths = list(getattr(solver, "history_best_lengths", []))
            tours = list(getattr(solver, "history_best_tours", []))
            if not tours or len(tours) != len(lengths):
                tours = [None] * len(lengths)
            histories[name] = {"lengths": lengths, "tours": tours}

        score = score_run_start(histories)
        candidates.append(
            {"seed": s, "histories": histories, "instance": inst, "score": score}
        )
        print(f"  score_start(min first lengths) = {score:.2f}")

    # Select candidate
    if args.pick == "first":
        chosen = candidates[0]
        tag = f"seed{chosen['seed']}"
    elif args.pick == "worst-start":
        chosen = max(candidates, key=lambda x: x["score"])
        tag = f"seed{chosen['seed']}_worststart"
    else:  # best-start
        chosen = min(candidates, key=lambda x: x["score"])
        tag = f"seed{chosen['seed']}_beststart"

    histories = chosen["histories"]
    inst = chosen["instance"]

    if args.side_by_side:
        out_gif = str(outdir / f"ACO_side_by_side_{tag}.gif")
        make_side_by_side_gif(inst, histories, out_gif, step=args.step)
        print("Saved:", out_gif)
        # Save final snapshot PNG (side-by-side)
        out_png = str(outdir / f"ACO_side_by_side_{tag}_final.png")
        save_side_by_side_final(inst, histories, out_png)
        print("Saved:", out_png)

    if args.separate:
        for name in ["AS", "ACS", "MMAS"]:
            gif_path = str(outdir / f"{name}_small_{tag}.gif")
            make_single_gif(inst, name, histories[name], gif_path, step=args.step)
            print("Saved:", gif_path)
            # Save final snapshot PNG (single)
            png_path = str(outdir / f"{name}_final_{tag}.png")
            save_single_final(inst, name, histories[name], png_path)
            print("Saved:", png_path)


if __name__ == "__main__":
    main()
