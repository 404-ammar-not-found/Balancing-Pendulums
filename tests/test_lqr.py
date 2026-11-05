import numpy as np

from src.dynamics import PendulumParams, params_to_array
from src.derive import sympy_linearize
from src.env import DoublePendulumCartEnv
from src.lqr import LQRController, controllability_matrix, linearize


def test_controllability():
    A, B = linearize()
    ctrb = controllability_matrix(A, B)
    assert np.linalg.matrix_rank(ctrb) == 6


def test_finite_difference_matches_sympy_jacobian():
    A_fd, B_fd = linearize(PendulumParams(b_cart=0.2))
    a_fn, b_fn = sympy_linearize()
    prm = params_to_array(PendulumParams(b_cart=0.2))
    args = (0, 0, 0, 0, 0, 0, 0.0, prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[10], prm[11])
    A_sym = np.asarray(a_fn(*args), dtype=np.float64)
    B_sym = np.asarray(b_fn(*args), dtype=np.float64).reshape(6, 1)
    assert np.abs(A_fd - A_sym).max() < 1e-5
    assert np.abs(B_fd - B_sym).max() < 1e-5


def test_closed_loop_eigenvalues_stable():
    ctrl = LQRController()
    assert np.max(ctrl.eigenvalues.real) < 0


def test_lqr_stabilizes_small_perturbation():
    env = DoublePendulumCartEnv(task="balance")
    ctrl = LQRController(params=env.params)
    obs, _ = env.reset(options={"init_state": np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0])})
    thetas = []
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(ctrl.act(obs))
        done = term or trunc
        thetas.append(info["state"][1:3])
        assert not term
    thetas = np.asarray(thetas)
    settle_step = 250
    assert np.all(np.abs(thetas[settle_step:]) < 0.01), f"{np.abs(thetas[settle_step:]).max()=}"
    assert thetas.shape[0] == 500
