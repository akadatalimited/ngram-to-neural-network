#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./tokenize_corpus.sh [INPUT] [OUTPUT] [extra tokenizer args...]

Purpose:
  Tokenize either:
    - a single text file, or
    - a directory of shard .txt files

Defaults:
  INPUT   training/corpus_chat
  OUTPUT  learned/all_works_tokenized

Examples:
  ./tokenize_corpus.sh
  ./tokenize_corpus.sh training/corpus_mixed learned/corpus_mixed_tokenized
  ./tokenize_corpus.sh training/corpus_chat learned/chat_tokenized --max-vocab-size 16384
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

INPUT="${1:-training/corpus_chat}"
OUTPUT="${2:-learned/all_works_tokenized}"

if (($# >= 2)); then
  shift 2
else
  shift $#
fi

exec python python/0_tokenize_all_works.py "$INPUT" "$OUTPUT" "$@"
