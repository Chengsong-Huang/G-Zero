#!/usr/bin/env bash
# Paper-main on Llama-3.1-8B-Instruct.
# Same recipe as Qwen3-8B-Base, with the llama3_instruct renderer.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag main_llama3_8b_instruct \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --renderer_name llama3_instruct \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
