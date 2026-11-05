"""Train SAC on the double pendulum cart environment.

Runs on CPU with 8 environments in subprocesses and VecNormalize observation
scaling. macOS uses the spawn start method, so the entry point is guarded.
Run long trainings under caffeinate to keep the Mac awake, e.g.

    caffeinate -i uv run python scripts/train_sac.py --task balance --seed 0
"""

import argparse
import pathlib
import sys

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.env import DoublePendulumCartEnv

DEFAULT_TIMESTEPS = {"balance": 200_000, "swingup": 2_000_000}
N_ENVS = 8
EVAL_FREQ = 20_000
NET_ARCH = [256, 256]


def make_env_fn(task: str, seed: int, rank: int, randomize: bool, velocity_penalty: float | None = None,
                episode_steps: int | None = None):
    def _init():
        reward_weights = None if velocity_penalty is None else {"velocity": velocity_penalty}
        env = DoublePendulumCartEnv(
            task=task,
            randomize_params=randomize,
            reward_weights=reward_weights,
            max_episode_steps=episode_steps,
        )
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["balance", "swingup"], default="balance")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--velocity-penalty", type=float, default=None)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--init-from", type=str, default=None, help="checkpoint dir to fine-tune from (model.zip + vecnormalize.pkl)")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(2)

    timesteps = args.timesteps if args.timesteps is not None else DEFAULT_TIMESTEPS[args.task]
    out_dir = args.out if args.out is not None else pathlib.Path(f"results/sac/{args.task}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env_fn(args.task, args.seed, rank, args.randomize, args.velocity_penalty, args.episode_steps)
        for rank in range(N_ENVS)
    ]
    if args.init_from is not None:
        init_dir = pathlib.Path(args.init_from)
        vec_env = VecNormalize.load(
            str(init_dir / "vecnormalize.pkl"), SubprocVecEnv(env_fns, start_method="spawn")
        )
        vec_env.training = True
        vec_env.norm_reward = True
        model = SAC.load(
            str(init_dir / "best_model.zip"),
            env=vec_env,
            device="cpu",
            tensorboard_log=str(pathlib.Path("results/tb")),
            verbose=1,
        )
    else:
        vec_env = VecNormalize(SubprocVecEnv(env_fns, start_method="spawn"))
        model = SAC(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            device="cpu",
            policy_kwargs=dict(net_arch=NET_ARCH),
            tensorboard_log=str(pathlib.Path("results/tb")),
            verbose=1,
        )

    eval_env = VecNormalize(
        DummyVecEnv([make_env_fn(args.task, args.seed + 1000, 0, False, args.velocity_penalty, args.episode_steps)]),
        training=False,
        norm_reward=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=str(out_dir / "eval"),
        n_eval_episodes=10,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=f"{args.task}_seed{args.seed}")
    model.save(out_dir / "model")
    vec_env.save(out_dir / "vecnormalize.pkl")
    vec_env.close()
    eval_env.close()
    print(f"Saved final model and normalisation stats to {out_dir}")


if __name__ == "__main__":
    main()
