import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.dynamics import PendulumParams
from src.env import TASK_CONFIG, DoublePendulumCartEnv


@pytest.mark.parametrize("task", ["balance", "swingup"])
def test_passes_env_checker(task):
    env = DoublePendulumCartEnv(task=task, render_mode="rgb_array")
    check_env(env, skip_render_check=False)


@pytest.mark.parametrize("task", ["balance", "swingup"])
def test_reset_deterministic_under_fixed_seed(task):
    env1 = DoublePendulumCartEnv(task=task, obs_noise_std=0.01)
    env2 = DoublePendulumCartEnv(task=task, obs_noise_std=0.01)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.array_equal(obs1, obs2)
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-1, 1, 1)
        o1, r1, t1, tr1, _ = env1.step(action)
        o2, r2, t2, tr2, _ = env2.step(action)
        assert np.array_equal(o1, o2) and r1 == r2 and t1 == t2 and tr1 == tr2


def test_observation_layout():
    env = DoublePendulumCartEnv(task="balance")
    init = np.array([0.3, 0.2, -0.1, 0.1, -0.2, 0.3])
    obs, _ = env.reset(options={"init_state": init})
    assert obs.shape == (9,)
    assert obs[0] == pytest.approx(0.3, abs=1e-6)
    assert obs[1] == pytest.approx(np.sin(0.2), abs=1e-6)
    assert obs[2] == pytest.approx(np.cos(0.2), abs=1e-6)
    assert obs[3] == pytest.approx(np.sin(-0.1), abs=1e-6)
    assert obs[4] == pytest.approx(np.cos(-0.1), abs=1e-6)
    assert obs[8] == pytest.approx(0.3 / env.params.track_limit, abs=1e-6)


def test_swingup_starts_hanging_balance_starts_upright():
    up, _ = DoublePendulumCartEnv(task="balance").reset(seed=0)
    hang, _ = DoublePendulumCartEnv(task="swingup").reset(seed=0)
    assert abs(np.arctan2(up[1], up[2])) < 0.15
    th1 = np.arctan2(hang[1], hang[2])
    assert abs(th1 - np.pi) < 0.1 or abs(th1 + np.pi) < 0.1


def test_random_policy_balance_terminates_quickly():
    env = DoublePendulumCartEnv(task="balance")
    lengths = []
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        done = False
        steps = 0
        while not done and steps < TASK_CONFIG["balance"]["max_episode_steps"]:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        lengths.append(steps)
    assert np.mean(lengths) < 200


def test_time_limit_truncates_never_terminates():
    env = DoublePendulumCartEnv(task="balance")
    env.reset(options={"init_state": np.zeros(6)}, seed=0)
    env.action_space.seed(0)
    for _ in range(TASK_CONFIG["balance"]["max_episode_steps"] - 1):
        _, _, term, trunc, _ = env.step(np.array([0.0]))
        assert not term and not trunc
    _, _, term, trunc, _ = env.step(np.array([0.0]))
    assert trunc and not term


def test_domain_randomization_perturbs_params():
    env = DoublePendulumCartEnv(task="balance", randomize_params=True)
    seen = set()
    nominal = env.nominal_params
    for _ in range(20):
        env.reset()
        p = env.params
        assert 0.8 - 1e-9 <= p.m1 / nominal.m1 <= 1.2 + 1e-9
        seen.add((round(p.m1, 6), round(p.l2, 6)))
    assert len(seen) > 1
    assert env.nominal_params == PendulumParams()


def test_rgb_array_render():
    env = DoublePendulumCartEnv(task="balance", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert isinstance(frame, np.ndarray) and frame.shape == (400, 800, 3)
    env.step(np.array([0.5]))
    frame = env.render()
    assert frame.dtype == np.uint8
    env.close()


def test_episode_lengths_match_config():
    for task, cfg in TASK_CONFIG.items():
        env = DoublePendulumCartEnv(task=task)
        env.reset(options={"init_state": np.zeros(6) if task == "balance" else np.array([0, np.pi, np.pi, 0, 0, 0.0])}, seed=0)
        n = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.0]))
            done = term or trunc
            n += 1
        assert n == cfg["max_episode_steps"]
