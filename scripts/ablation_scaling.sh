#!/usr/bin/env bash
# Ablation: pool-size scaling. Sweeps `num_questions` ∈ {500, 1000, 2000, 5000}.
# Each run is independent so they share no cache (a smaller pool can't be
# subsetted into a fixed bot50 cleanly without re-filtering). Run sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-8B-Base}
RENDERER=${RENDERER:-qwen3}

for NQ in 500 1000 2000 5000; do
  bash run.sh \
    --tag "ablation_scaling/n${NQ}" \
    --model_name "$MODEL" \
    --renderer_name "$RENDERER" \
    --run_phase1 false \
    --num_questions "$NQ" \
    --pct_low 0 --pct_high 50 \
    --eval_tasks aime24,aime25
done
