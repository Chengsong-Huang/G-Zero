#!/usr/bin/env bash
# One-command driver for the G-Zero pipeline.
#
# Usage:
#   export TINKER_API_KEY=...                     # required
#   bash run.sh                                   # default: paper-main R1
#   bash run.sh --run_phase1 false                # no_phase1 ablation
#   bash run.sh --tag big --num_questions 5000    # custom tag / pool size
#
# All flags after `bash run.sh` are forwarded verbatim to g_zero/main.py.
# See scripts/ for canned single-command experiments.

set -euo pipefail

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set." >&2
  echo "  export TINKER_API_KEY=<your-key>     # from https://tinker.thinkingmachines.ai/" >&2
  exit 1
fi

cd "$(dirname "$0")"

# NLTK data needed by the BLEU duplication penalty (Phase 1 only).
# Idempotent — fast no-op when already cached.
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" 2>/dev/null || true

exec python -m g_zero.main "$@"
