"""Linearisation around upright and an LQR controller for the balance task."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from src.dynamics import PendulumParams, params_to_array, accel

DEFAULT_Q = np.diag([1.0, 50.0, 50.0, 1.0, 1.0, 1.0])
DEFAULT_R = np.array([[0.1]])


def state_derivative(s: np.ndarray, F: float, prm: np.ndarray) -> np.ndarray:
    """Full state derivative sdot = [sdot_angles, qdd] used for linearisation."""
    out = np.empty(6)
    out[:3] = s[3:]
    out[3:] = accel(s, F, prm)
    return out


def linearize(
    params: PendulumParams | None = None, s_star: np.ndarray | None = None, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Central-finite-difference linearisation of sdot around ``s_star`` (upright) and F = 0."""
    prm = params_to_array(params if params is not None else PendulumParams())
    s_star = np.zeros(6) if s_star is None else np.asarray(s_star, dtype=np.float64)
    A = np.empty((6, 6))
    for j in range(6):
        sp = s_star.copy()
        sm = s_star.copy()
        sp[j] += eps
        sm[j] -= eps
        A[:, j] = (state_derivative(sp, 0.0, prm) - state_derivative(sm, 0.0, prm)) / (2 * eps)
    B = np.zeros((6, 1))
    B[3:, 0] = (accel(s_star, eps, prm) - accel(s_star, -eps, prm)) / (2 * eps)
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stack [B, AB, A²B, ..., Aⁿ⁻¹B]."""
    n = A.shape[0]
    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray = DEFAULT_Q, R: np.ndarray = DEFAULT_R) -> np.ndarray:
    """Continuous-time LQR gain K from the algebraic Riccati equation; u = -K s."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def obs_to_state(obs: np.ndarray) -> np.ndarray:
    """Map the 9-dim observation to the raw 6-dim state (angles unwrapped via atan2)."""
    obs = np.asarray(obs).ravel()
    if obs.shape[0] == 6:
        return obs.astype(np.float64)
    return np.array(
        [
            obs[0],
            np.arctan2(obs[1], obs[2]),
            np.arctan2(obs[3], obs[4]),
            obs[5],
            obs[6],
            obs[7],
        ]
    )


class LQRController:
    """Balance-only LQR controller with the same act(obs) interface as an RL policy."""

    def __init__(
        self,
        params: PendulumParams | None = None,
        Q: np.ndarray = DEFAULT_Q,
        R: np.ndarray = DEFAULT_R,
    ):
        self.params = params if params is not None else PendulumParams()
        self.A, self.B = linearize(self.params)
        self.K = lqr_gain(self.A, self.B, Q, R)
        self.eigenvalues = np.linalg.eigvals(self.A - self.B @ self.K)

    def force(self, obs_or_state: np.ndarray) -> float:
        """Raw force F = -K s for a 6-dim state or a 9-dim observation."""
        s = obs_to_state(obs_or_state)
        return float(-(self.K @ s)[0])

    def act(self, obs_or_state: np.ndarray) -> np.ndarray:
        """Action in the env's [-1, 1] action space, scaled by F_max."""
        return np.array([np.clip(self.force(obs_or_state) / self.params.F_max, -1.0, 1.0)], dtype=np.float32)
