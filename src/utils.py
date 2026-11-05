"""Shared helpers: controller wrappers with a common act(obs) interface and run metrics."""

from __future__ import annotations

import pathlib

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.dynamics import PendulumParams
from src.env import DoublePendulumCartEnv
from src.lqr import LQRController


class SB3Controller:
    """Wraps a trained SB3 model plus its VecNormalize stats behind act(obs)."""

    def __init__(self, model_path: str | pathlib.Path, vecnormalize_path: str | pathlib.Path, name: str | None = None):
        self.model = SAC.load(model_path, device="cpu")
        env = DoublePendulumCartEnv(task="balance")
        self.normalizer = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: env]))
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        self.name = name or pathlib.Path(model_path).parent.name

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_n = self.normalizer.normalize_obs(np.asarray(obs))
        action, _ = self.model.predict(obs_n, deterministic=True)
        return action


class LQRWrapper:
    """LQRController behind the same act(obs) interface and naming."""

    def __init__(self, params: PendulumParams | None = None, name: str = "lqr"):
        self.controller = LQRController(params)
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.controller.act(obs)


def load_controller(spec: str, params: PendulumParams | None = None) -> SB3Controller | LQRWrapper:
    """Load "lqr" or a checkpoint directory containing model.zip/vecnormalize.pkl (or a model.zip path)."""
    if spec == "lqr":
        return LQRWrapper(params)
    path = pathlib.Path(spec)
    if path.is_dir():
        model_path = path / "best_model.zip" if (path / "best_model.zip").exists() else path / "model.zip"
        vec_path = path / "vecnormalize.pkl"
    else:
        model_path = path
        vec_path = path.parent / "vecnormalize.pkl"
    return SB3Controller(model_path, vec_path)


def time_to_upright(
    states: np.ndarray, control_dt: float, thresh: float = 0.1, sustain_s: float = 2.0
) -> float | None:
    """Seconds until both angles stay within ``thresh`` of upright for ``sustain_s`` continuously.

    Returns None if the condition is never met. ``states`` is (T, 6) raw states.
    """
    need = int(round(sustain_s / control_dt))
    inside = (np.abs(states[:, 1]) < thresh) & (np.abs(states[:, 2]) < thresh)
    count = 0
    for t, ok in enumerate(inside):
        count = count + 1 if ok else 0
        if count >= need:
            return (t - need + 1) * control_dt
    return None
