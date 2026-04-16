#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./finetune_chat_corpus.sh [extra training args...]

Purpose:
  Resume from the current all_works checkpoint and continue training on the
  chat/dictionary corpus encoded with the same vocabulary.

Defaults:
  --resume-from learned/gpt_all_works/checkpoint_latest.pt
  --token-dir    learned/corpus_chat_compatible_tokenized
  --output-dir   learned/gpt_chat_finetune
  --learning-rate 0.0001
  --max-vocab-size 32768
  --batch-size 4
  --grad-accum-steps 4
  --eval-batches 100

Examples:
  ./finetune_chat_corpus.sh
  ./finetune_chat_corpus.sh --epochs 1
  ./finetune_chat_corpus.sh --token-dir learned/corpus_mixed_compatible_tokenized --output-dir learned/gpt_mixed_finetune
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
else
  echo "No base checkpoint found in learned/gpt_all_works" >&2
  exit 1
fi

exec ./python/1_train_gpt_all_works.py \
  --resume-from "$RESUME_FROM" \
  --token-dir learned/corpus_chat_compatible_tokenized \
  --output-dir learned/gpt_chat_finetune \
  --learning-rate 0.0001 \
  --max-vocab-size 32768 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --eval-batches 100 \
  "$@"
