"""Equations of motion, RK4 integrator and energy for a double pendulum on a cart.

State layout: s = [x, theta1, theta2, xdot, theta1dot, theta2dot].
Angles are measured from the upright vertical, positive clockwise, so
theta = 0 is upright and theta = pi is hanging down.

Dynamics are written in manipulator form M(q) qdd = f(q, qd, F) and solved
numerically with np.linalg.solve. The jitted functions take the parameters
as a plain float64 array (see ``PARAM_ORDER`` / ``params_to_array``) because
numba cannot consume dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np


@dataclass(frozen=True)
class PendulumParams:
    """Physical and integration parameters, all overridable."""

    M: float = 1.0
    m1: float = 0.1
    m2: float = 0.1
    l1: float = 0.5
    l2: float = 0.5
    g: float = 9.81
    dt: float = 0.002
    control_dt: float = 0.02
    F_max: float = 20.0
    track_limit: float = 2.4
    b_cart: float = 0.0
    b_joint: float = 0.0


PARAM_ORDER = (
    "M", "m1", "m2", "l1", "l2", "g",
    "dt", "control_dt", "F_max", "track_limit", "b_cart", "b_joint",
)


def params_to_array(params: PendulumParams) -> np.ndarray:
    """Pack a ``PendulumParams`` dataclass into the float64 array the jitted functions expect."""
    return np.array([getattr(params, k) for k in PARAM_ORDER], dtype=np.float64)


def _as_array(params: PendulumParams | np.ndarray) -> np.ndarray:
    if isinstance(params, PendulumParams):
        return params_to_array(params)
    return np.asarray(params, dtype=np.float64)


def mass_matrix(s: np.ndarray, params: PendulumParams | np.ndarray) -> np.ndarray:
    """Hand-written symmetric mass matrix M(q) of the manipulator form."""
    p = _as_array(params)
    M, m1, m2, l1, l2 = p[0], p[1], p[2], p[3], p[4]
    th1, th2 = s[1], s[2]
    return np.array(
        [
            [M + m1 + m2, (m1 + m2) * l1 * np.cos(th1), m2 * l2 * np.cos(th2)],
            [(m1 + m2) * l1 * np.cos(th1), (m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(th1 - th2)],
            [m2 * l2 * np.cos(th2), m2 * l1 * l2 * np.cos(th1 - th2), m2 * l2**2],
        ]
    )


def rhs(s: np.ndarray, F: float, params: PendulumParams | np.ndarray) -> np.ndarray:
    """Hand-written right-hand side f(q, qd, F) of the manipulator form."""
    p = _as_array(params)
    m1, m2, l1, l2, g = p[1], p[2], p[3], p[4], p[5]
    b_cart, b_joint = p[10], p[11]
    th1, th2, xd, w1, w2 = s[1], s[2], s[3], s[4], s[5]
    f1 = F + (m1 + m2) * l1 * w1**2 * np.sin(th1) + m2 * l2 * w2**2 * np.sin(th2) - b_cart * xd
    f2 = -m2 * l1 * l2 * w2**2 * np.sin(th1 - th2) + (m1 + m2) * g * l1 * np.sin(th1) - b_joint * w1
    f3 = m2 * l1 * l2 * w1**2 * np.sin(th1 - th2) + m2 * g * l2 * np.sin(th2) - b_joint * w2
    return np.array([f1, f2, f3])


def accel_numpy(s: np.ndarray, F: float, params: PendulumParams | np.ndarray) -> np.ndarray:
    """Pure-numpy acceleration qdd = M(q)^-1 f(q, qd, F); reference for the numba version."""
    return np.linalg.solve(mass_matrix(s, params), rhs(s, F, params))


def rk4_step_numpy(
    s: np.ndarray, F: float, dt: float, params: PendulumParams | np.ndarray
) -> np.ndarray:
    """Pure-numpy single RK4 step; reference for the numba version."""

    def deriv(sv: np.ndarray) -> np.ndarray:
        out = np.empty(6)
        out[:3] = sv[3:]
        out[3:] = accel_numpy(sv, F, params)
        return out

    k1 = deriv(s)
    k2 = deriv(s + 0.5 * dt * k1)
    k3 = deriv(s + 0.5 * dt * k2)
    k4 = deriv(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@numba.njit(cache=True)
def accel(s: np.ndarray, F: float, prm: np.ndarray) -> np.ndarray:
    """Numba-jitted acceleration qdd = M(q)^-1 f(q, qd, F). ``prm`` is a float64 array in PARAM_ORDER."""
    M, m1, m2, l1, l2 = prm[0], prm[1], prm[2], prm[3], prm[4]
    g, b_cart, b_joint = prm[5], prm[10], prm[11]
    th1 = s[1]
    th2 = s[2]
    xd = s[3]
    w1 = s[4]
    w2 = s[5]
    c1 = np.cos(th1)
    c2 = np.cos(th2)
    c12 = np.cos(th1 - th2)
    s1 = np.sin(th1)
    s2 = np.sin(th2)
    s12 = np.sin(th1 - th2)
    m11 = M + m1 + m2
    m12 = (m1 + m2) * l1 * c1
    m13 = m2 * l2 * c2
    m22 = (m1 + m2) * l1 * l1
    m23 = m2 * l1 * l2 * c12
    m33 = m2 * l2 * l2
    f1 = F + (m1 + m2) * l1 * w1 * w1 * s1 + m2 * l2 * w2 * w2 * s2 - b_cart * xd
    f2 = -m2 * l1 * l2 * w2 * w2 * s12 + (m1 + m2) * g * l1 * s1 - b_joint * w1
    f3 = m2 * l1 * l2 * w1 * w1 * s12 + m2 * g * l2 * s2 - b_joint * w2
    a = np.linalg.solve(
        np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]),
        np.array([f1, f2, f3]),
    )
    return a


@numba.njit(cache=True)
def _deriv(s: np.ndarray, F: float, prm: np.ndarray) -> np.ndarray:
    out = np.empty(6)
    a = accel(s, F, prm)
    out[0] = s[3]
    out[1] = s[4]
    out[2] = s[5]
    out[3] = a[0]
    out[4] = a[1]
    out[5] = a[2]
    return out


@numba.njit(cache=True)
def rk4_step(s: np.ndarray, F: float, dt: float, prm: np.ndarray) -> np.ndarray:
    """Numba-jitted single RK4 step for state ``s`` under constant force ``F``."""
    k1 = _deriv(s, F, prm)
    k2 = _deriv(s + 0.5 * dt * k1, F, prm)
    k3 = _deriv(s + 0.5 * dt * k2, F, prm)
    k4 = _deriv(s + dt * k3, F, prm)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def total_energy(s: np.ndarray, params: PendulumParams | np.ndarray) -> float:
    """Total energy (kinetic + potential, potential zero at the cart's height)."""
    p = _as_array(params)
    M, m1, m2, l1, l2, g = p[0], p[1], p[2], p[3], p[4], p[5]
    th1, th2, xd, w1, w2 = s[1], s[2], s[3], s[4], s[5]
    xb1 = xd + l1 * np.cos(th1) * w1
    yb1 = -l1 * np.sin(th1) * w1
    xb2 = xb1 + l2 * np.cos(th2) * w2
    yb2 = yb1 - l2 * np.sin(th2) * w2
    kinetic = 0.5 * M * xd**2 + 0.5 * m1 * (xb1**2 + yb1**2) + 0.5 * m2 * (xb2**2 + yb2**2)
    potential = m1 * g * l1 * np.cos(th1) + m2 * g * (l1 * np.cos(th1) + l2 * np.cos(th2))
    return float(kinetic + potential)


def horizontal_momentum(s: np.ndarray, params: PendulumParams | np.ndarray) -> float:
    """Total horizontal momentum of cart plus both bobs (conserved when F = 0, no damping)."""
    p = _as_array(params)
    M, m1, m2, l1, l2 = p[0], p[1], p[2], p[3], p[4]
    th1, th2, xd, w1, w2 = s[1], s[2], s[3], s[4], s[5]
    xb1 = xd + l1 * np.cos(th1) * w1
    xb2 = xb1 + l2 * np.cos(th2) * w2
    return float(M * xd + m1 * xb1 + m2 * xb2)
