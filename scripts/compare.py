"""Comparison plots: learning curves, region of attraction, time traces, swing-up trace.

Writes PNGs to results/plots/ and skips any plot whose models are missing.
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.env import DoublePendulumCartEnv
from src.utils import load_controller, time_to_upright

PLOTS_DIR = pathlib.Path("results/plots")


def load_eval_runs(task: str) -> list[dict]:
    runs = []
    for seed_dir in sorted(pathlib.Path("results/sac").glob(f"{task}_seed[0-9]")):
        npz_path = seed_dir / "eval" / "evaluations.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            runs.append({"seed": seed_dir.name, "timesteps": data["timesteps"], "results": data["results"].mean(axis=1)})
    return runs


def plot_learning_curves(task: str, out: pathlib.Path) -> bool:
    runs = load_eval_runs(task)
    if len(runs) < 1:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ts = runs[0]["timesteps"]
    all_curves = []
    for run in runs:
        curve = np.interp(ts, run["timesteps"], run["results"])
        all_curves.append(curve)
        ax.plot(ts, curve, color="tab:blue", alpha=0.35, lw=1)
    curves = np.asarray(all_curves)
    ax.plot(ts, curves.mean(axis=0), color="tab:blue", lw=2, label=f"SAC mean ± std ({len(runs)} seeds)")
    ax.fill_between(ts, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0), alpha=0.2, color="tab:blue")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title(f"SAC learning curves ({task})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def roa_success(controller, th1: float, th2: float, task: str) -> bool:
    env = DoublePendulumCartEnv(task=task)
    init = np.array([0.0, th1, th2, 0.0, 0.0, 0.0])
    obs, _ = env.reset(options={"init_state": init})
    done = False
    terminated = False
    states = []
    while not done:
        obs, _, term, trunc, info = env.step(controller.act(obs))
        terminated = terminated or term
        states.append(info["state"])
        done = term or trunc
    states = np.asarray(states)
    final = states[-100:].max(axis=0)
    return not terminated and abs(final[1]) < 0.05 and abs(final[2]) < 0.05


def plot_roa(controllers: dict, out: pathlib.Path, grid: int = 21, span: float = 0.6) -> bool:
    if not controllers:
        return False
    ths = np.linspace(-span, span, grid)
    fig, axes = plt.subplots(1, len(controllers), figsize=(5 * len(controllers), 4.6), squeeze=False)
    for ax, (name, controller) in zip(axes[0], controllers.items()):
        success = np.zeros((grid, grid))
        for i, th1 in enumerate(ths):
            for j, th2 in enumerate(ths):
                success[i, j] = roa_success(controller, th1, th2, "balance")
        ax.imshow(success, origin="lower", extent=(-span, span, -span, span), cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"{name}: {success.mean() * 100:.0f}% success")
        ax.set_xlabel(r"$\theta_1(0)$ [rad]")
        ax.set_ylabel(r"$\theta_2(0)$ [rad]")
    fig.suptitle(f"Region of attraction, balance task ({grid}×{grid} grid)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def collect_trace(controller, task: str, init: np.ndarray):
    env = DoublePendulumCartEnv(task=task)
    obs, _ = env.reset(options={"init_state": init})
    done = False
    rows = []
    forces = []
    t = 0.0
    while not done:
        action = controller.act(obs)
        obs, _, term, trunc, info = env.step(action)
        rows.append(info["state"].copy())
        forces.append(info["force"])
        t = info["time"]
        done = term or trunc
    states = np.asarray(rows)
    times = np.arange(1, len(rows) + 1) * env.params.control_dt
    return times, states, np.asarray(forces)


def plot_traces(controllers: dict, out: pathlib.Path, init: np.ndarray, task: str = "balance") -> bool:
    if not controllers:
        return False
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    colors = {"lqr": "tab:red", "sac": "tab:blue"}
    for name, controller in controllers.items():
        times, states, forces = collect_trace(controller, task, init)
        color = colors.get(name, None)
        axes[0].plot(times, states[:, 0], label=name, color=color)
        axes[1].plot(times, states[:, 1], color=color)
        axes[1].plot(times, states[:, 2], color=color, ls="--")
        axes[2].plot(times, states[:, 3], color=color)
        axes[3].plot(times, forces, color=color)
    axes[0].set_ylabel("x [m]")
    axes[0].legend()
    axes[1].set_ylabel(r"$\theta_1$, $\theta_2$ [rad]")
    axes[2].set_ylabel(r"$\dot{x}$ [m/s]")
    axes[3].set_ylabel("F [N]")
    axes[3].set_xlabel("t [s]")
    axes[3].set_ylim(-22, 22)
    fig.suptitle(f"Time traces ({task}), init " + ", ".join(f"{v:.2f}" for v in init))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sac", default="results/sac/balance_seed0", help="SAC checkpoint dir for ROA/traces")
    parser.add_argument("--sac-swingup", default=None, help="SAC swingup checkpoint dir (default: auto-detect)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    controllers = {"lqr": load_controller("lqr")}
    try:
        controllers["sac"] = load_controller(args.sac)
    except Exception as exc:
        print(f"skipping SAC comparison plots: {exc}")

    made = plot_learning_curves("balance", PLOTS_DIR / "learning_curves_balance.png")
    print(f"learning curves (balance): {'written' if made else 'skipped'}")
    made = plot_roa(controllers, PLOTS_DIR / "roa_balance.png")
    print(f"region of attraction: {'written' if made else 'skipped'}")
    made = plot_traces(controllers, PLOTS_DIR / "traces_balance.png", np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0]))
    print(f"time traces (balance): {'written' if made else 'skipped'}")

    made = plot_learning_curves("swingup", PLOTS_DIR / "learning_curves_swingup.png")
    print(f"learning curves (swingup): {'written' if made else 'skipped'}")
    swingup_dir = args.sac_swingup
    if swingup_dir is None:
        candidates = sorted(pathlib.Path("results/sac").glob("swingup_seed[0-9]"))
        swingup_dir = str(candidates[-1]) if candidates else None
    if swingup_dir is not None:
        try:
            sac_sw = load_controller(swingup_dir)
            made = plot_traces({"sac": sac_sw}, PLOTS_DIR / "traces_swingup.png", np.array([0.0, np.pi, np.pi, 0, 0, 0.0]), task="swingup")
            print(f"swing-up traces: {'written' if made else 'skipped'}")
        except Exception as exc:
            print(f"swing-up traces skipped: {exc}")
    else:
        print("swing-up traces skipped: no swingup checkpoint found")


if __name__ == "__main__":
    main()
