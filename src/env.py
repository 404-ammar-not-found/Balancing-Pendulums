"""Gymnasium environment for the double pendulum on a cart.

Tasks: ``balance`` (start near upright, hold it) and ``swingup`` (start hanging,
swing up and balance). Angles are from upright, positive clockwise; the physics
advances 10 RK4 substeps of ``params.dt`` per control step (``params.control_dt``).
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium
import numpy as np
from gymnasium import spaces

from src.dynamics import PendulumParams, params_to_array, rk4_step
from src.render import PendulumRenderer

TASK_CONFIG = {
    "balance": {"max_episode_steps": 500, "angle_terminate": 0.5},
    "swingup": {"max_episode_steps": 1000, "angle_terminate": None},
}

REWARD_WEIGHTS = {
    "x": 0.01,
    "velocity": 0.001,
    "action": 0.001,
    "terminal_track_exit": -100.0,
}

OBS_LOW = np.array([-3.0, -1.0, -1.0, -1.0, -1.0, -25.0, -60.0, -60.0, -1.5], dtype=np.float32)
OBS_HIGH = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 25.0, 60.0, 60.0, 1.5], dtype=np.float32)


class DoublePendulumCartEnv(gymnasium.Env):
    """Double inverted pendulum on a cart, with optional domain randomisation and obs noise."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: str = "balance",
        params: PendulumParams | None = None,
        randomize_params: bool = False,
        obs_noise_std: float = 0.0,
        render_mode: str | None = None,
        reward_weights: dict | None = None,
    ):
        if task not in TASK_CONFIG:
            raise ValueError(f"unknown task {task!r}, expected one of {tuple(TASK_CONFIG)}")
        super().__init__()
        self.task = task
        self.nominal_params = params if params is not None else PendulumParams()
        self.params = self.nominal_params
        self.randomize_params = randomize_params
        self.obs_noise_std = obs_noise_std
        self.render_mode = render_mode
        self.reward_weights = dict(REWARD_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        cfg = TASK_CONFIG[task]
        self.max_episode_steps = cfg["max_episode_steps"]
        self.angle_terminate = cfg["angle_terminate"]
        self.n_substeps = int(round(self.params.control_dt / self.params.dt))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)

        self._prm = params_to_array(self.params)
        self._state = np.zeros(6)
        self._elapsed_steps = 0
        self._renderer: PendulumRenderer | None = None

    @property
    def state(self) -> np.ndarray:
        """Raw state [x, theta1, theta2, xdot, theta1dot, theta2dot]."""
        return self._state.copy()

    def _sample_params(self) -> PendulumParams:
        scales = self.np_random.uniform(0.8, 1.2, size=4)
        return replace(
            self.nominal_params,
            m1=self.nominal_params.m1 * scales[0],
            m2=self.nominal_params.m2 * scales[1],
            l1=self.nominal_params.l1 * scales[2],
            l2=self.nominal_params.l2 * scales[3],
        )

    def _sample_init_state(self) -> np.ndarray:
        if self.task == "balance":
            th1, th2 = self.np_random.uniform(-0.1, 0.1, size=2)
            rest = self.np_random.uniform(-0.05, 0.05, size=4)
        else:
            th1 = np.pi + self.np_random.uniform(-0.05, 0.05)
            th2 = np.pi + self.np_random.uniform(-0.05, 0.05)
            rest = self.np_random.uniform(-0.05, 0.05, size=4)
            rest[1:] *= 0.2
        return np.array([rest[0], th1, th2, rest[1], rest[2], rest[3]])

    def _observation(self) -> np.ndarray:
        x, th1, th2, xd, w1, w2 = self._state
        obs = np.array(
            [x, np.sin(th1), np.cos(th1), np.sin(th2), np.cos(th2), xd, w1, w2, x / self.params.track_limit],
            dtype=np.float64,
        )
        if self.obs_noise_std > 0.0:
            obs = obs + self.np_random.normal(0.0, self.obs_noise_std, size=obs.shape)
        return obs.astype(np.float32)

    def _reward(self, action: float, terminated: bool) -> float:
        x, th1, th2, xd, w1, w2 = self._state
        w = self.reward_weights
        penalties = w["x"] * x**2 + w["velocity"] * (w1**2 + w2**2) + w["action"] * float(action) ** 2
        if self.task == "balance":
            return 1.0 - penalties
        upright = (np.cos(th1) + np.cos(th2)) / 2.0
        r = upright - penalties
        if terminated and abs(x) > self.params.track_limit:
            r += w["terminal_track_exit"]
        return float(r)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.randomize_params:
            self.params = self._sample_params()
            self._prm = params_to_array(self.params)
        if options is not None and "init_state" in options:
            self._state = np.asarray(options["init_state"], dtype=np.float64).copy()
        else:
            self._state = self._sample_init_state()
        self._elapsed_steps = 0
        return self._observation(), {"state": self.state}

    def step(self, action):
        action = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        force = action * self.params.F_max
        for _ in range(self.n_substeps):
            self._state = rk4_step(self._state, force, self.params.dt, self._prm)
        self._elapsed_steps += 1

        x, th1, th2 = self._state[0], self._state[1], self._state[2]
        terminated = bool(abs(x) > self.params.track_limit)
        if self.angle_terminate is not None:
            terminated = terminated or abs(th1) > self.angle_terminate or abs(th2) > self.angle_terminate
        truncated = bool(self._elapsed_steps >= self.max_episode_steps and not terminated)
        reward = self._reward(action, terminated)
        info = {
            "state": self.state,
            "force": force,
            "time": self._elapsed_steps * self.params.control_dt,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = PendulumRenderer(
                self.render_mode, self.params.track_limit, self.params.l1, self.params.l2
            )
        return self._renderer.render(self._state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
