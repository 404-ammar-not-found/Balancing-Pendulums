"""Independent derivation of the equations of motion from the Lagrangian via sympy.

Used to verify the hand-written dynamics in ``src/dynamics.py``. Bob positions
follow the spec convention (angles from upright, positive clockwise):

    p1 = (x + l1 sin(theta1), l1 cos(theta1))
    p2 = (x + l1 sin(theta1) + l2 sin(theta2), l1 cos(theta1) + l2 cos(theta2))

The Euler-Lagrange equations with generalised forces Q = [F - b_cart xdot,
-b_joint theta1dot, -b_joint theta2dot] are reduced to manipulator form
M(q) qdd = f(q, qd, F) and lambdified for numeric comparison.
"""

from __future__ import annotations

import sympy as sp

from src.dynamics import PARAM_ORDER

_x, _th1, _th2 = sp.symbols("x theta1 theta2", real=True)
_xd, _w1, _w2 = sp.symbols("xdot w1 w2", real=True)
_ax, _a1, _a2 = sp.symbols("xddot a1 a2", real=True)
_F = sp.Symbol("F", real=True)
_M, _m1, _m2, _l1, _l2, _g = sp.symbols("M m1 m2 l1 l2 g", positive=True)
_bc, _bj = sp.symbols("b_cart b_joint", nonnegative=True)

_q = sp.Matrix([_x, _th1, _th2])
_qd = sp.Matrix([_xd, _w1, _w2])
_qdd = sp.Matrix([_ax, _a1, _a2])

_FUNC_ARGS = (_x, _th1, _th2, _xd, _w1, _w2, _F, _M, _m1, _m2, _l1, _l2, _g, _bc, _bj)


def _build():
    y1 = _l1 * sp.cos(_th1)
    p1 = sp.Matrix([_x + _l1 * sp.sin(_th1), y1])
    p2 = sp.Matrix([_x + _l1 * sp.sin(_th1) + _l2 * sp.sin(_th2), y1 + _l2 * sp.cos(_th2)])

    def velocity(p: sp.Matrix) -> sp.Matrix:
        return p.jacobian(_q) * _qd

    v1 = velocity(p1)
    v2 = velocity(p2)
    kinetic = (
        sp.Rational(1, 2) * _M * _xd**2
        + sp.Rational(1, 2) * _m1 * (v1.dot(v1))
        + sp.Rational(1, 2) * _m2 * (v2.dot(v2))
    )
    potential = _m1 * _g * y1 + _m2 * _g * (y1 + _l2 * sp.cos(_th2))
    lagrangian = kinetic - potential

    grad_qd = lagrangian.diff(_qd.T)
    j_q = grad_qd.jacobian(_q)
    j_qd = grad_qd.jacobian(_qd)
    forces = sp.Matrix([_F - _bc * _xd, -_bj * _w1, -_bj * _w2])
    eom = j_qd * _qdd + j_q * _qd - lagrangian.diff(_q.T).reshape(3, 1) - forces

    mass_sym = sp.simplify(eom.jacobian(_qdd))
    rhs_sym = sp.simplify(-(eom.subs({_ax: 0, _a1: 0, _a2: 0})))
    return mass_sym, rhs_sym


_MASS_SYM, _RHS_SYM = _build()
_QDD_SYM = sp.simplify(_MASS_SYM.LUsolve(_RHS_SYM))


def mass_matrix_sympy() -> sp.Matrix:
    """Symbolic mass matrix M(q) in manipulator form."""
    return _MASS_SYM


def rhs_sympy() -> sp.Matrix:
    """Symbolic right-hand side f(q, qd, F) in manipulator form."""
    return _RHS_SYM


def sympy_accel(s, F: float, params) -> "np.ndarray":
    """Numpy-compatible qdd from the sympy derivation; signature matches ``dynamics.accel``."""
    import numpy as np

    fn = sp.lambdify(_FUNC_ARGS, _QDD_SYM, modules="numpy")
    p = np.asarray(params, dtype=np.float64)
    return np.asarray(
        fn(s[0], s[1], s[2], s[3], s[4], s[5], F, p[0], p[1], p[2], p[3], p[4], p[5], p[10], p[11]),
        dtype=np.float64,
    ).ravel()


def sympy_linearize():
    """Return lambdified (A_fn, B_fn) Jacobians of sdot = f(s, F) from the sympy derivation."""
    state = sp.Matrix([_xd, _w1, _w2, _QDD_SYM[0], _QDD_SYM[1], _QDD_SYM[2]])
    a_expr = state.jacobian(sp.Matrix([_x, _th1, _th2, _xd, _w1, _w2]))
    b_expr = state.diff(_F)
    a_fn = sp.lambdify(_FUNC_ARGS, a_expr, modules="numpy")
    b_fn = sp.lambdify(_FUNC_ARGS, b_expr, modules="numpy")
    return a_fn, b_fn


def check_against_handwritten(n_states: int = 100, tol: float = 1e-9, seed: int = 0) -> float:
    """Compare M and f from sympy against ``dynamics.mass_matrix`` / ``dynamics.rhs``.

    Returns the maximum absolute discrepancy over ``n_states`` random states and
    raises ``AssertionError`` if it exceeds ``tol``.
    """
    import numpy as np

    from src.dynamics import PendulumParams, mass_matrix, params_to_array, rhs

    params = PendulumParams(b_cart=0.31, b_joint=0.07)
    prm = params_to_array(params)
    m_fn = sp.lambdify(_FUNC_ARGS, _MASS_SYM, modules="numpy")
    f_fn = sp.lambdify(_FUNC_ARGS, _RHS_SYM, modules="numpy")
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_states):
        s = np.concatenate(
            [rng.uniform(-2.0, 2.0, 1), rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5.0, 5.0, 3)]
        )
        force = float(rng.uniform(-20.0, 20.0))
        args = (
            s[0], s[1], s[2], s[3], s[4], s[5], force,
            prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[10], prm[11],
        )
        err = np.maximum(
            np.abs(np.asarray(m_fn(*args)) - mass_matrix(s, prm)).max(),
            np.abs(np.asarray(f_fn(*args)).ravel() - rhs(s, force, prm)).max(),
        )
        max_err = max(max_err, float(err))
    assert max_err < tol, f"sympy and hand-written dynamics disagree: max err {max_err:.3e}"
    return max_err
