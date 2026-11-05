"""Robustness sweeps: m2 / l2 scale 0.5x-1.5x and observation noise, LQR vs SAC vs SAC (randomised).

Plots success rate vs parameter for each controller on the balance task.
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dynamics import PendulumParams
from src.env import DoublePendulumCartEnv
from src.utils import load_controller, time_to_upright

PLOTS_DIR = pathlib.Path("results/plots")


def success_rate(controller, params: PendulumParams, episodes: int, obs_noise: float, seed: int) -> float:
    env = DoublePendulumCartEnv(task="balance", params=params, obs_noise_std=obs_noise)
    env.reset(seed=seed)
    wins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        terminated = False
        states = []
        while not done:
            obs, _, term, trunc, info = env.step(controller.act(obs))
            terminated = terminated or term
            states.append(info["state"])
            done = term or trunc
        upright = time_to_upright(np.asarray(states), env.params.control_dt) is not None
        wins += int(upright and not terminated)
    return wins / episodes


def sweep(controller_specs: dict, vary: str, values: np.ndarray, episodes: int, noise: float) -> dict:
    results = {name: np.zeros(len(values)) for name in controller_specs}
    controllers = {name: load_controller(spec) for name, spec in controller_specs.items()}
    nominal = PendulumParams()
    for k, val in enumerate(values):
        params = replace(nominal, **{vary: getattr(nominal, vary) * val})
        for name, controller in controllers.items():
            results[name][k] = success_rate(controller, params, episodes, noise, seed=1000 + k)
    return results


def plot_sweep(values: np.ndarray, results: dict, xlabel: str, title: str, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, rates in results.items():
        ax.plot(values, rates, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sac", default="results/sac/balance_seed0")
    parser.add_argument("--sac-randomized", default="results/sac/balance_seed0_rand")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    controller_specs = {"lqr": "lqr"}
    for label, spec in [("sac", args.sac), ("sac_rand", args.sac_randomized)]:
        if pathlib.Path(spec).exists():
            controller_specs[label] = spec
        else:
            print(f"skipping {label} ({spec} not found)")

    scales = np.linspace(0.5, 1.5, 7)
    for vary, symbol, fname in [("m2", r"$m_2$ scale", "robustness_m2.png"), ("l2", r"$l_2$ scale", "robustness_l2.png")]:
        results = sweep(controller_specs, vary, scales, args.episodes, noise=0.0)
        plot_sweep(scales, results, symbol, f"Success rate vs {symbol} (nominal = 1.0)", PLOTS_DIR / fname)
        print(f"wrote {PLOTS_DIR / fname}: " + ", ".join(f"{k}={v.tolist()}" for k, v in results.items()))

    noises = np.array([0.0, 0.01, 0.05])
    results = {}
    nominal = PendulumParams()
    controllers = {name: load_controller(spec) for name, spec in controller_specs.items()}
    for noise in noises:
        for name, controller in controllers.items():
            results.setdefault(name, []).append(
                success_rate(controller, nominal, args.episodes, float(noise), seed=2000)
            )
    results = {k: np.asarray(v) for k, v in results.items()}
    plot_sweep(noises, results, "observation noise std", "Success rate vs observation noise", PLOTS_DIR / "robustness_noise.png")
    print(f"wrote {PLOTS_DIR / 'robustness_noise.png'}: " + ", ".join(f"{k}={v.tolist()}" for k, v in results.items()))


if __name__ == "__main__":
    main()
