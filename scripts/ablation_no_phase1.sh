#!/usr/bin/env bash
# Ablation: skip Phase 1; use the base model directly as the Challenger.
# On Qwen3-8B-Base this matches Phase-1-on R1 within noise — i.e. the GRPO
# training of the Challenger contributes very little once the length and
# diversity penalties are in place. The role of Phase 1 is mainly to keep
# the Challenger from collapsing on instruct models that drift.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag ablation_no_phase1 \
  --model_name Qwen/Qwen3-8B-Base \
  --renderer_name qwen3 \
  --run_phase1 false \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
