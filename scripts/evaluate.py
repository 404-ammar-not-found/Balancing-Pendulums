"""Run any controller on the pendulum environment for N episodes and record metrics.

Metrics: success rate, mean return, time to upright (both |theta| < 0.1 sustained
2 s), max |x|, control effort sum(F^2 dt). Saves a GIF of the first episode and
a JSON summary. Success = sustained upright reached and no early termination.
"""

import argparse
import json
import pathlib
import sys
from dataclasses import replace

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dynamics import PendulumParams
from src.env import DoublePendulumCartEnv
from src.utils import load_controller, time_to_upright


def run_episode(controller, env, record_frames: bool = False):
    obs, _ = env.reset()
    frames: list[np.ndarray] = [] if record_frames else None  # type: ignore[assignment]
    done = False
    reward_sum = 0.0
    effort = 0.0
    max_abs_x = abs(float(env.state[0]))
    states = [env.state]
    forces = []
    while not done:
        if record_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        action = controller.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        reward_sum += reward
        effort += float(info["force"]) ** 2 * env.params.control_dt
        max_abs_x = max(max_abs_x, abs(float(info["state"][0])))
        states.append(info["state"])
        forces.append(float(info["force"]))
        done = term or trunc
    states = np.asarray(states)
    return {
        "return": reward_sum,
        "effort": effort,
        "max_abs_x": max_abs_x,
        "terminated": bool(term),
        "time_to_upright": time_to_upright(states[:-1], env.params.control_dt),
        "success": bool(time_to_upright(states[:-1], env.params.control_dt) is not None and not term),
        "frames": frames,
        "states": states,
        "forces": np.asarray(forces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True, help='"lqr" or a checkpoint dir/model.zip path')
    parser.add_argument("--task", choices=["balance", "swingup"], default="balance")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-noise", type=float, default=0.0)
    parser.add_argument("--m2-scale", type=float, default=1.0)
    parser.add_argument("--l2-scale", type=float, default=1.0)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--video", action="store_true", help="save a GIF of the first episode")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    nominal = PendulumParams()
    params = replace(nominal, m2=nominal.m2 * args.m2_scale, l2=nominal.l2 * args.l2_scale)
    env = DoublePendulumCartEnv(
        task=args.task,
        params=params,
        obs_noise_std=args.obs_noise,
        render_mode="rgb_array" if args.video else None,
        max_episode_steps=args.episode_steps,
    )
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    controller = load_controller(args.controller, params)

    episodes = []
    for ep in range(args.episodes):
        result = run_episode(controller, env, record_frames=args.video and ep == 0)
        episodes.append(result)

    if args.video and episodes[0]["frames"]:
        out_dir = pathlib.Path("results/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_dir / f"{controller.name}_{args.task}.gif", episodes[0]["frames"], duration=20)

    summary = {
        "controller": args.controller,
        "name": controller.name,
        "task": args.task,
        "episodes": args.episodes,
        "obs_noise_std": args.obs_noise,
        "m2_scale": args.m2_scale,
        "l2_scale": args.l2_scale,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "std_return": float(np.std([e["return"] for e in episodes])),
        "mean_time_to_upright": float(
            np.mean([e["time_to_upright"] for e in episodes if e["time_to_upright"] is not None])
        ) if any(e["time_to_upright"] is not None for e in episodes) else None,
        "max_abs_x_worst": float(np.max([e["max_abs_x"] for e in episodes])),
        "mean_effort": float(np.mean([e["effort"] for e in episodes])),
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
