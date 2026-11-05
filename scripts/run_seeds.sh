#!/usr/bin/env bash
# Launch SAC training for seeds 0, 1, 2 in parallel, each with its own log file.
# Usage: scripts/run_seeds.sh [task] [timesteps]
set -euo pipefail

TASK="${1:-balance}"
TIMESTEPS="${2:-}"

mkdir -p results/logs
pids=()
for seed in 0 1 2; do
  extra=()
  if [[ -n "$TIMESTEPS" ]]; then
    extra+=(--timesteps "$TIMESTEPS")
  fi
  echo "Launching seed ${seed} (log: results/logs/train_${TASK}_seed${seed}.log)"
  caffeinate -i uv run python scripts/train_sac.py --task "$TASK" --seed "$seed" "${extra[@]}" \
    > "results/logs/train_${TASK}_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one seed failed; check results/logs/." >&2
  exit 1
fi
echo "All seeds finished."
