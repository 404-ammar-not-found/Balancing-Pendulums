# balancing-pendulms

Double inverted pendulum on a cart, built from first principles: Lagrangian
physics (verified with sympy), a Gymnasium environment, an LQR baseline, and an
SAC agent trained to balance — with comparison and robustness analysis.
Full spec in PROJECT.md.

## Setup

```bash
uv sync                      # Python 3.12 venv via uv (or: python3.12 -m venv .venv && pip install -r requirements.txt)
uv run pytest                # 24 tests: physics, env, LQR
uv run python scripts/bench_dynamics.py   # numba vs numpy physics throughput
```

## Usage

```bash
# Train SAC (balance task, 3 seeds in parallel; caffeinate keeps the Mac awake)
scripts/run_seeds.sh balance

# Or a single run
caffeinate -i uv run python scripts/train_sac.py --task balance --seed 0

# Swing-up (hours per seed)
scripts/run_seeds.sh swingup

# Train with domain randomisation
uv run python scripts/train_sac.py --task balance --seed 0 --randomize --out results/sac/balance_seed0_rand

# Evaluate any controller (100 episodes, GIF of the first episode)
uv run python scripts/evaluate.py --controller lqr --task balance --episodes 100 --video
uv run python scripts/evaluate.py --controller results/sac/balance_seed0 --task balance --episodes 100 --video

# Comparison plots (learning curves, region of attraction, time traces)
uv run python scripts/compare.py

# Robustness sweeps (m2/l2 scale 0.5x-1.5x, observation noise)
uv run python scripts/robustness.py

# Watch a trained policy live
uv run python scripts/evaluate.py --controller results/sac/balance_seed0 --task balance --episodes 1 --video
```

Artifacts: `results/plots/` (committed), `results/sac/` checkpoints,
`results/videos/` GIFs, `results/tb/` tensorboard logs
(`uv run tensorboard --logdir results/tb`).

## Results

### Balance task (500 steps = 10 s, start near upright)

**Physics.** Hand-written equations of motion match the independent sympy
Lagrangian derivation at 100 random states to < 1e-9 (with damping exercised),
RK4 conserves energy to a relative drift < 1e-4 over 10 s, and both equilibria,
hanging stability, upright instability and momentum conservation are tested.
Numba gives ~11x faster physics (≈ 37k control steps/s per core).

**Controllers** (100 episodes each, success = both |theta| < 0.1 rad sustained
2 s and no early termination):

| Controller | Success | Mean return | Time to upright | Control effort (sum F² dt) |
|---|---|---|---|---|
| LQR | 100% | 499.9 | 0.2 s | 17.3 |
| SAC seed 0 | 100% | 499.6 | — | — |
| SAC seed 1 | 100% | 499.7 | — | — |
| SAC seed 2 | 96% | 483.3 | — | — |

All three SAC seeds beat the 95% acceptance bar; final eval rewards during
training were 497.7 / 495.6 / 489.6 (± ~1) out of a 500-step maximum.
Training took ~7 min per seed on an M4 Pro (CPU only, 8 SubprocVecEnv workers,
2 torch threads), well under the spec's 15–30 min estimate.

**Where RL beat LQR, where it didn't, and why:**

- *Region of attraction* (`results/plots/roa_balance.png`): essentially a tie —
  LQR recovers 17.0% of the (theta1, theta2) grid in [-0.6, 0.6]² with zero
  velocity, SAC 16.3%. Both are local controllers: the double pendulum is
  simply not stabilisable from most of that grid with a 20 N force limit, and
  RL found nothing bigger.
- *Robustness to model mismatch* (`robustness_m2.png`, `robustness_l2.png`):
  both LQR and nominal SAC hold 90–100% success across m2 in [0.05, 0.15] kg
  and l2 in [0.25, 0.75] m. The single domain-randomised SAC seed survives
  everywhere (0/10 early terminations at the extremes) but hovers slightly
  outside the strict 0.1 rad window at 0.5x m2 and 1.3–1.5x l2, so it scores
  lower under the strict metric — randomisation traded steady-state tightness
  for never falling (n = 1 seed).
- *Observation noise* (`robustness_noise.png`): both controllers are flawless
  at noise std 0.01 and both fail at 0.05. The LQR gains are aggressive
  (R = 0.1), so 0.05 rad-equivalent noise injects several newtons of white
  force jitter and the pendulum wobbles past 0.5 rad. SAC, trained without
  noise, inherits the same sensitivity. Robustness to noise would need noise
  injected during training (or a higher R for LQR).
- *Swing-up*: not yet trained in this run — `scripts/run_seeds.sh swingup`
  (2M steps/seed, hours) reproduces it; `compare.py` picks up the checkpoints
  automatically and writes `traces_swingup.png` + a swing-up GIF via
  `evaluate.py --task swingup`.

Plots: `learning_curves_balance.png`, `roa_balance.png`, `traces_balance.png`,
`robustness_*.png`. Episode GIFs in `results/videos/`.

## Spec deviations

- Repository name is `balancing-pendulms`; the spec's tree says
  `double-pendulum-rl/` (structure inside matches otherwise).
- Dependency management via uv (`pyproject.toml` + `uv.lock`);
  `requirements.txt` kept in sync for the spec.
- `robustness.py` and `compare.py` load a domain-randomised SAC checkpoint if
  `results/sac/balance_seed0_rand` exists (trained with `--randomize`).
- Success metric everywhere: both |theta| < 0.1 rad sustained 2 s and no early
  termination.
