#!/usr/bin/env bash
# Ablation: 3 rounds of Challenger ↔ Solver co-evolution.
# Round 1 captures most of the gain on Qwen3-8B-Base; round 2/3 either
# regress or drift. Included so the curve can be reproduced.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set." >&2
  exit 1
fi

# punkt for the BLEU diversity penalty (Phase 1)
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" 2>/dev/null || true

exec python -m g_zero.multi_round \
  --tag ablation_multi_round \
  --model_name Qwen/Qwen3-8B-Base \
  --renderer_name qwen3 \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  --num_rounds 3 \
  "$@"
