#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./prepare_chat_corpus.sh [--root DIR] [--output DIR] [extra corpus args...]

Purpose:
  Build a chat-style supervision corpus with records like:
    STARTTOK User: ... Assistant: ... ENDTOK

Defaults:
  --root   literature
  --output training/corpus_chat

Included sources:
  - english_dict.json.txt
  - English Bible CSV files:
    ASV, BBE, Darby, KJV, Webster, YLT

Examples:
  ./prepare_chat_corpus.sh
  ./prepare_chat_corpus.sh --output training/corpus_chat_small --max-records 50000
EOF
}

ROOT="literature"
OUTPUT="training/corpus_chat"
EXTRA_ARGS=()

while (($# > 0)); do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

exec python python/0_prepare_literature_corpus.py "$ROOT" "$OUTPUT" \
  --mode qa \
  --qa-style chat \
  --include-glob 'english_dict.json.txt' \
  --include-glob 'bible_databases/formats/csv/ASV.csv' \
  --include-glob 'bible_databases/formats/csv/BBE.csv' \
  --include-glob 'bible_databases/formats/csv/Darby.csv' \
  --include-glob 'bible_databases/formats/csv/KJV.csv' \
  --include-glob 'bible_databases/formats/csv/Webster.csv' \
  --include-glob 'bible_databases/formats/csv/YLT.csv' \
  "${EXTRA_ARGS[@]}"
