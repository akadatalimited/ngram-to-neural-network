#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./continue_structured_v2.sh [extra training args...]

Purpose:
  Continue the current mixed-model checkpoint on the cleaner structured_v2
  corpus without changing model shape, token IDs, or vocabulary.

Defaults:
  --resume-from   learned/gpt_corpus_mixed_128k/checkpoint_latest.pt
  --token-dir     learned/corpus_structured_v2_compatible
  --output-dir    learned/gpt_corpus_structured_v2
  --context-len   256
  --d-model       384
  --n-heads       8
  --n-layers      6
  --d-ff          1024
  --max-vocab-size 131072
  --batch-size    16
  --grad-accum-steps 1
  --num-workers   6
  --prefetch-factor 6
  --eval-batches  50
  --log-every     2000
  --save-every    6000

Examples:
  ./continue_structured_v2.sh
  ./continue_structured_v2.sh --epochs 2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

RESUME_FROM=""
if [[ -f learned/gpt_corpus_mixed_128k/checkpoint_latest.pt ]]; then
  RESUME_FROM="learned/gpt_corpus_mixed_128k/checkpoint_latest.pt"
elif [[ -f learned/gpt_corpus_mixed_128k/checkpoint.pt ]]; then
  RESUME_FROM="learned/gpt_corpus_mixed_128k/checkpoint.pt"
else
  echo "No base checkpoint found in learned/gpt_corpus_mixed_128k" >&2
  exit 1
fi

exec ./python/1_train_gpt_all_works.py \
  --token-dir learned/corpus_structured_v2_compatible \
  --output-dir learned/gpt_corpus_structured_v2 \
  --resume-from "$RESUME_FROM" \
  --context-len 256 \
  --d-model 384 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 1024 \
  --max-vocab-size 131072 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --num-workers 6 \
  --persistent-workers \
  --prefetch-factor 6 \
  --eval-batches 50 \
  --log-every 2000 \
  --save-every 6000 \
  --no-compile-model \
  "$@"
