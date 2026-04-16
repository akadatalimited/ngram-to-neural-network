#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./prepare_mixed_corpus.sh [--root DIR] [--output DIR] [extra corpus args...]

Purpose:
  Build a mixed corpus with:
    - chat-style User/Assistant records from structured sources
    - plain cleaned literary records from selected English prose/drama

Defaults:
  --root   literature
  --output training/corpus_mixed

Included sources:
  - english_dict.json.txt
  - English Bible CSV files:
    ASV, BBE, Darby, KJV, Webster, YLT
  - Shakespeare HTML plays

Examples:
  ./prepare_mixed_corpus.sh
  ./prepare_mixed_corpus.sh --output training/corpus_mixed_small --max-records 100000
EOF
}

ROOT="literature"
OUTPUT="training/corpus_mixed"
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
  --mode mixed \
  --qa-style chat \
  --include-glob 'english_dict.json.txt' \
  --include-glob 'bible_databases/formats/csv/ASV.csv' \
  --include-glob 'bible_databases/formats/csv/BBE.csv' \
  --include-glob 'bible_databases/formats/csv/Darby.csv' \
  --include-glob 'bible_databases/formats/csv/KJV.csv' \
  --include-glob 'bible_databases/formats/csv/Webster.csv' \
  --include-glob 'bible_databases/formats/csv/YLT.csv' \
  --include-glob 'shakespeare/**/*.html' \
  "${EXTRA_ARGS[@]}"
