#!/usr/bin/env bash
# Paper-main: Qwen3-8B-Base, single round, bot50 δ-cutoff.
# ~$5–8 of Tinker compute, 5–7h wall-clock.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag main_qwen3_8b_base \
  --model_name Qwen/Qwen3-8B-Base \
  --renderer_name qwen3 \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
