#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./eval_structured_v2.sh [CHECKPOINT] [extra eval args...]

Purpose:
  Evaluate the structured_v2 finetune line against the fixed QA prompt set.

Defaults:
  CHECKPOINT  learned/gpt_corpus_structured_v2/checkpoint_latest.pt
  TOKEN_DIR   learned/corpus_structured_v2_compatible

Examples:
  ./eval_structured_v2.sh
  ./eval_structured_v2.sh learned/gpt_corpus_structured_v2/checkpoint_step_6000.pt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

CHECKPOINT="${1:-}"
if (($# > 0)); then
  shift 1
fi

if [[ -z "$CHECKPOINT" ]]; then
  if [[ -f learned/gpt_corpus_structured_v2/checkpoint_latest.pt ]]; then
    CHECKPOINT="learned/gpt_corpus_structured_v2/checkpoint_latest.pt"
  elif [[ -f learned/gpt_corpus_structured_v2/checkpoint.pt ]]; then
    CHECKPOINT="learned/gpt_corpus_structured_v2/checkpoint.pt"
  else
    echo "No structured_v2 checkpoint exists yet in learned/gpt_corpus_structured_v2" >&2
    echo "Wait for the first validation/save pass, then rerun ./eval_structured_v2.sh" >&2
    exit 1
  fi
fi

exec python python/4_eval_gpt_qa.py "$CHECKPOINT" --token-dir learned/corpus_structured_v2_compatible "$@"
