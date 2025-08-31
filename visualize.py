import os, argparse
import matplotlib.pyplot as plt
import imageio

from aco import TSPInstance, ACOConfig, AntSystem, AntColonySystem, MaxMinAntSystem

ALGO_MAP = {"AS": AntSystem, "ACS": AntColonySystem, "MMAS": MaxMinAntSystem}

def build_config(algo, n_ants=25, n_iterations=120):
    if algo == "AS":
        return ACOConfig(alpha=1.0, beta=3.0, rho=0.5, Q=1.0, n_ants=n_ants, n_iterations=n_iterations)
    if algo == "ACS":
        return ACOConfig(alpha=1.0, beta=2.0, rho=0.1, Q=1.0, q0=0.9, phi=0.1,
                         n_ants=n_ants, n_iterations=n_iterations)
    if algo == "MMAS":
        return ACOConfig(alpha=1.0, beta=3.0, rho=0.2, Q=1.0, n_ants=n_ants, n_iterations=n_iterations,
                         tau_min=1e-4, tau_max=1.0)
    raise ValueError("Unsupported algo")

def visualize(inst, algo_name, cfg, outdir, step=5):
    os.makedirs(outdir, exist_ok=True)
    solver = ALGO_MAP[algo_name](inst.distance_matrix(), cfg)
    _ = solver.run()

    coords = inst.coords
    frames = []
    iters = list(range(0, len(solver.history_best_tours), step))
    for it in iters:
        tour = solver.history_best_tours[it]
        L = solver.history_best_lengths[it]
        xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
        ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
        cx = [c[0] for c in coords]
        cy = [c[1] for c in coords]

        plt.figure(figsize=(5,5))
        plt.plot(cx, cy, "o")
        plt.plot(xs, ys, "-")
        plt.title(f"{algo_name} best-so-far\niter={it+1}  length={L:.2f}")
        plt.axis("equal")
        plt.tight_layout()
        frame_path = os.path.join(outdir, f"{algo_name}_frame_{it:03d}.png")
        plt.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close()
        frames.append(frame_path)

    gif_path = os.path.join(outdir, f"{algo_name}_convergence.gif")
    with imageio.get_writer(gif_path, mode="I", duration=0.6) as writer:
        for fp in frames:
            writer.append_data(imageio.v2.imread(fp))

    print("Saved:", gif_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["AS","ACS","MMAS"], default="MMAS")
    p.add_argument("--n", type=int, default=50, help="number of cities")
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--ants", type=int, default=25)
    p.add_argument("--square", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--outdir", default="viz")
    p.add_argument("--step", type=int, default=5, help="frame every k iterations")
    args = p.parse_args()

    inst = TSPInstance.random_euclidean(n=args.n, seed=args.seed, square_size=args.square, name=f"viz{args.n}")
    cfg = build_config(args.algo, n_ants=args.ants, n_iterations=args.iters)
    visualize(inst, args.algo, cfg, args.outdir, step=args.step)

if __name__ == "__main__":
    main()