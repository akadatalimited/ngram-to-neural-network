#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./tokenize_compatible_corpus.sh [INPUT] [OUTPUT] [--vocab-json PATH] [extra tokenizer args...]

Purpose:
  Tokenize a new corpus with the current model's existing vocabulary so that
  training can resume from an existing checkpoint.

Defaults:
  INPUT   training/corpus_chat
  OUTPUT  learned/corpus_chat_compatible_tokenized
  VOCAB   learned/gpt_corpus_mixed_128k/effective_vocab.json
          then learned/gpt_all_works/effective_vocab.json as fallback

Examples:
  ./tokenize_compatible_corpus.sh
  ./tokenize_compatible_corpus.sh training/corpus_mixed learned/corpus_mixed_compatible_tokenized
  ./tokenize_compatible_corpus.sh training/corpus_structured_v2 learned/corpus_structured_v2_compatible
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

INPUT="${1:-training/corpus_chat}"
OUTPUT="${2:-learned/corpus_chat_compatible_tokenized}"
VOCAB_JSON="${COMPAT_VOCAB_JSON:-}"

if (($# >= 2)); then
  shift 2
else
  shift $#
fi

EXTRA_ARGS=()
while (($# > 0)); do
  case "$1" in
    --vocab-json)
      VOCAB_JSON="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$VOCAB_JSON" ]]; then
  if [[ -f learned/gpt_corpus_mixed_128k/effective_vocab.json ]]; then
    VOCAB_JSON="learned/gpt_corpus_mixed_128k/effective_vocab.json"
  elif [[ -f learned/gpt_all_works/effective_vocab.json ]]; then
    VOCAB_JSON="learned/gpt_all_works/effective_vocab.json"
  else
    echo "No compatible vocab JSON found. Pass --vocab-json PATH." >&2
    exit 1
  fi
fi

exec python python/0_tokenize_all_works.py \
  "$INPUT" \
  "$OUTPUT" \
  --vocab-json "$VOCAB_JSON" \
  "${EXTRA_ARGS[@]}"
