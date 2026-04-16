#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./eval_qa.sh [CHECKPOINT] [--token-dir DIR] [extra eval args...]

Purpose:
  Run a fixed QA prompt set against a checkpoint and save reports.

Defaults:
  CHECKPOINT  learned/gpt_corpus_mixed_128k/checkpoint_latest.pt
  TOKEN_DIR   learned/corpus_mixed_tokenized_128k

Examples:
  ./eval_qa.sh
  ./eval_qa.sh learned/gpt_corpus_mixed_128k/checkpoint_step_108000.pt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

CHECKPOINT="${1:-learned/gpt_corpus_mixed_128k/checkpoint_latest.pt}"
if (($# > 0)); then
  shift 1
fi

TOKEN_DIR="${EVAL_QA_TOKEN_DIR:-learned/corpus_mixed_tokenized_128k}"
EXTRA_ARGS=()
while (($# > 0)); do
  case "$1" in
    --token-dir)
      TOKEN_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

exec python python/4_eval_gpt_qa.py "$CHECKPOINT" --token-dir "$TOKEN_DIR" "${EXTRA_ARGS[@]}"
