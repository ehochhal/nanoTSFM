#!/usr/bin/env bash
# Run the random-25% vs smart-25% subset test on a GPU machine.
#
# Usage (from the repo root, after ./run.sh setup and ./run.sh data):
#   STATS=path/to/series_stats_GEP-M_train.parquet ./subset-test/run_all.sh
#
# Set FULL=1 to also train full GEP-M at the same seeds as a reference.
# Each run is about 2 minutes of training plus about 5 minutes of GIFT-Eval.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STATS=${STATS:?Set STATS to the GEP-M stats parquet}
SEEDS=${SEEDS:-"7 1 2"}
RUNS=${RUNS:-runs}
extras=(--extra gift)
[[ -n "${TORCH:-}" ]] && extras+=(--extra "$TORCH")

uv run --locked "${extras[@]}" python subset-test/make_subsets.py --stats "$STATS" --seeds $SEEDS

for seed in $SEEDS; do
  for method in random smart; do
    name="$method-25-s$seed"
    uv run --locked "${extras[@]}" python -m nanotsfm.train --config configs/series.yaml \
      --output "$RUNS/$name" --data "subsets/$name" --seed "$seed"
    ./run.sh eval "$name"
  done
  if [[ "${FULL:-0}" == 1 ]]; then
    ./run.sh train "full-s$seed" configs/series.yaml "$seed"
    ./run.sh eval "full-s$seed"
  fi
done
