#!/usr/bin/env bash
# Ablation: δ-percentile cutoff sweep.
#
# Reuses one Phase-2 raw_pool across cutoffs by only re-running Phase 2's
# *filter* + Phase 3 (DPO) + eval per cutoff. We do this by:
#   1. Generating the pool once under a "_pool" tag.
#   2. Symlinking raw_pool.jsonl into each cutoff tag.
#   3. Running with that tag — main.py sees the cached pool and skips
#      generation, then re-filters according to --pct_low/--pct_high.
#
# Cutoffs studied in the paper (all positive on AIME, bot50 best on average):
#   bot10  [0,10],  bot25 [0,25],  bot50 [0,50],  bot75 [0,75],  top50 [50,100]
set -euo pipefail
cd "$(dirname "$0")/.."

POOL_TAG="ablation_cutoffs/_pool"
MODEL=${MODEL:-Qwen/Qwen3-8B-Base}
RENDERER=${RENDERER:-qwen3}
NQ=${NQ:-3000}

# 1. Build the pool once.
bash run.sh \
  --tag "$POOL_TAG" \
  --model_name "$MODEL" \
  --renderer_name "$RENDERER" \
  --run_phase1 true \
  --num_questions "$NQ" \
  --pct_low 0 --pct_high 50 \
  --eval_tasks aime24,aime25

POOL_FILE="runs/$POOL_TAG/raw_pool.jsonl"
if [[ ! -f "$POOL_FILE" ]]; then
  echo "ERROR: expected pool at $POOL_FILE" >&2
  exit 1
fi

# 2. + 3. Per-cutoff: link the cached pool, then run.
for cut in "0:10" "0:25" "0:50" "0:75" "50:100"; do
  lo=${cut%:*}; hi=${cut#*:}
  TAG="ablation_cutoffs/p${lo}_${hi}"
  mkdir -p "runs/$TAG"
  ln -sf "$(pwd)/$POOL_FILE" "runs/$TAG/raw_pool.jsonl"
  bash run.sh \
    --tag "$TAG" \
    --model_name "$MODEL" \
    --renderer_name "$RENDERER" \
    --run_phase1 false \
    --num_questions "$NQ" \
    --pct_low "$lo" --pct_high "$hi" \
    --eval_tasks aime24,aime25
done
