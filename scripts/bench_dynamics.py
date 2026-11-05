"""Benchmark physics throughput: env-style control steps (10 RK4 substeps) with and without numba."""

import argparse
import time

import numpy as np

from src.dynamics import PendulumParams, params_to_array, rk4_step, rk4_step_numpy


def bench(step_fn, prm, n_steps):
    rng = np.random.default_rng(0)
    s = np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0])
    forces = rng.uniform(-20, 20, n_steps)
    start = time.perf_counter()
    for f in forces:
        for _ in range(10):
            s = step_fn(s, f, 0.002, prm)
    elapsed = time.perf_counter() - start
    return n_steps / elapsed, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    prm = params_to_array(PendulumParams())
    rk4_step(np.zeros(6), 0.0, 0.002, prm)

    rate_jit, t_jit = bench(rk4_step, prm, args.steps)
    rate_np, t_np = bench(rk4_step_numpy, prm, args.steps)
    print(f"numba : {rate_jit:10.0f} control steps/s  ({t_jit:.2f} s for {args.steps} steps)")
    print(f"numpy : {rate_np:10.0f} control steps/s  ({t_np:.2f} s for {args.steps} steps)")
    print(f"speedup: {rate_jit / rate_np:.1f}x")


if __name__ == "__main__":
    main()
