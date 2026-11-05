import numpy as np
import pytest

from src.derive import check_against_handwritten, sympy_accel
from src.dynamics import (
    PendulumParams,
    accel,
    accel_numpy,
    horizontal_momentum,
    mass_matrix,
    params_to_array,
    rk4_step,
    rk4_step_numpy,
    total_energy,
)

PARAMS = PendulumParams()
PRM = params_to_array(PARAMS)


def rollout(s0, n_steps, force=0.0, dt=0.002):
    s = np.array(s0, dtype=np.float64)
    for _ in range(n_steps):
        s = rk4_step(s, force, dt, PRM)
    return s


def test_sympy_matches_handwritten():
    err = check_against_handwritten(n_states=100, tol=1e-9)
    assert err < 1e-9


def test_numba_matches_numpy():
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(100):
        s = np.concatenate(
            [rng.uniform(-2, 2, 1), rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5, 5, 3)]
        )
        f = float(rng.uniform(-20, 20))
        err_a = np.abs(accel(s, f, PRM) - accel_numpy(s, f, PRM)).max()
        err_r = np.abs(rk4_step(s, f, 0.002, PRM) - rk4_step_numpy(s, f, 0.002, PRM)).max()
        max_err = max(max_err, float(err_a), float(err_r))
    assert max_err < 1e-12


def test_energy_conservation():
    s0 = np.array([0.0, np.pi - 0.5, np.pi + 0.3, 0.0, 0.0, 0.0])
    e0 = total_energy(s0, PRM)
    s = rollout(s0, n_steps=5000)
    e1 = total_energy(s, PRM)
    assert abs(e1 - e0) / abs(e0) < 1e-4


def test_equilibria():
    upright = np.zeros(6)
    hanging = np.array([0.3, np.pi, np.pi, 0.0, 0.0, 0.0])
    assert np.abs(accel(upright, 0.0, PRM)).max() < 1e-12
    assert np.abs(accel(hanging, 0.0, PRM)).max() < 1e-12
    s = rollout(upright, n_steps=500)
    assert np.abs(s).max() < 1e-9
    s = rollout(hanging, n_steps=500)
    assert np.abs(s - hanging).max() < 1e-9


def test_hanging_perturbation_stays_bounded():
    s0 = np.array([0.0, np.pi - 0.05, np.pi + 0.04, 0.0, 0.0, 0.0])
    s = np.array(s0)
    max_dev = 0.0
    for _ in range(5000):
        s = rk4_step(s, 0.0, 0.002, PRM)
        max_dev = max(max_dev, abs(s[1] - np.pi), abs(s[2] - np.pi))
    assert max_dev < 0.5


def test_upright_perturbation_diverges():
    s0 = np.array([0.0, 0.05, -0.03, 0.0, 0.0, 0.0])
    s = np.array(s0)
    for i in range(1500):
        s = rk4_step(s, 0.0, 0.002, PRM)
        if abs(s[1]) > 0.3:
            break
    assert abs(s[1]) > 0.3


def test_horizontal_momentum_conservation():
    s0 = np.array([0.0, np.pi - 0.4, np.pi + 0.2, 0.5, 0.3, -0.2])
    p0 = horizontal_momentum(s0, PRM)
    s = rollout(s0, n_steps=2500)
    p1 = horizontal_momentum(s, PRM)
    assert abs(p1 - p0) / max(abs(p0), 1e-9) < 1e-6


@pytest.mark.parametrize("b_cart,b_joint", [(0.0, 0.0), (0.4, 0.02)])
def test_sympy_accel_matches_jitted(b_cart, b_joint):
    prm = params_to_array(PendulumParams(b_cart=b_cart, b_joint=b_joint))
    rng = np.random.default_rng(7)
    for _ in range(20):
        s = rng.uniform(-1, 1, 6)
        f = float(rng.uniform(-20, 20))
        assert np.abs(sympy_accel(s, f, prm) - accel(s, f, prm)).max() < 1e-9
