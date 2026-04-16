#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./train_all_works.sh [extra training args...]

Purpose:
  Start training with sane defaults for the 12 GB GPU and resume automatically
  if an existing checkpoint is present.

Automatic resume order:
  1. learned/gpt_all_works/checkpoint_latest.pt
  2. learned/gpt_all_works/checkpoint.pt

Defaults:
  --max-vocab-size 32768
  --batch-size 4
  --grad-accum-steps 4
  --eval-batches 100

Examples:
  ./train_all_works.sh
  ./train_all_works.sh --epochs 2
  ./train_all_works.sh --token-dir learned/corpus_chat_tokenized --epochs 3
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

RESUME_FROM=""
if [[ -f learned/gpt_all_works/checkpoint_latest.pt ]]; then
  RESUME_FROM="learned/gpt_all_works/checkpoint_latest.pt"
elif [[ -f learned/gpt_all_works/checkpoint.pt ]]; then
  RESUME_FROM="learned/gpt_all_works/checkpoint.pt"
fi

CMD=(
  ./python/1_train_gpt_all_works.py
  --max-vocab-size 32768
  --batch-size 4
  --grad-accum-steps 4
  --eval-batches 100
)

if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume-from "$RESUME_FROM")
fi

CMD+=("$@")
exec "${CMD[@]}"
