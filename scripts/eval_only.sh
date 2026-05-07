#!/usr/bin/env bash
# Re-run the full eval (AIME24+AIME25+IFEval+AlpacaEval) on an existing
# Tinker sampler URI without retraining.
#
# Usage:
#   bash scripts/eval_only.sh <tinker_sampler_uri> [--eval_tasks aime24,aime25,...]
#
# The base model is whatever the sampler was trained on; pass
# --model_name / --renderer_name to override.
set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLER=${1:?usage: eval_only.sh <sampler_uri> [extra args]}
shift

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set." >&2
  exit 1
fi

python - "$SAMPLER" "$@" <<'PY'
import sys, logging
from g_zero.config import parse_cli_overrides
from g_zero.eval import evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

sampler = sys.argv[1]
config = parse_cli_overrides(sys.argv[2:])
if not getattr(config, "tag", None) or config.tag == "g_zero_demo":
    config.tag = "eval_only"
evaluate(config, sampler_path=sampler)
PY
