#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./prepare_structured_v2_corpus.sh [--root DIR] [--output DIR] [--bible-csv NAME.csv] [extra corpus args...]

Purpose:
  Build a cleaner continuation corpus in separate source buckets under one new
  output tree. This preserves the current checkpoint line while improving the
  presentation of the same source material.

Output layout:
  training/corpus_structured_v2/
    dictionary/
    bible/
    fables/
    stories/
    literature/
    manifest.json

Defaults:
  --root       literature
  --output     training/corpus_structured_v2
  --bible-csv  KJV.csv

Examples:
  ./prepare_structured_v2_corpus.sh
  ./prepare_structured_v2_corpus.sh --output training/corpus_structured_v2_small --max-records 50000
EOF
}

ROOT="literature"
OUTPUT="training/corpus_structured_v2"
BIBLE_CSV="KJV.csv"
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
    --bible-csv)
      BIBLE_CSV="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

run_builder() {
  local subdir="$1"
  shift
  python python/0_prepare_literature_corpus.py "$ROOT" "$OUTPUT/$subdir" "$@"
}

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

run_builder dictionary \
  --mode qa \
  --qa-style chat \
  --min-words 3 \
  --include-glob 'english_dict.json.txt' \
  "${EXTRA_ARGS[@]}"

run_builder bible \
  --mode qa \
  --qa-style chat \
  --min-words 3 \
  --include-glob "bible_databases/formats/csv/$BIBLE_CSV" \
  "${EXTRA_ARGS[@]}"

run_builder fables \
  --mode mixed \
  --qa-style chat \
  --min-words 4 \
  --include-glob 'classics/Aesop/fab.mb.txt' \
  --include-glob 'bilingual-literature/Grimm/The_Fox_and_the_Cat/*.html' \
  --include-glob 'bilingual-literature/Grimm/The_Wolf_and_the_Fox/*.html' \
  --include-glob 'bilingual-literature/Charles_Perrault/Cinderella/*.html' \
  --include-glob 'bilingual-literature/Charles_Perrault/Little_Red_Riding_Hood/*.html' \
  --include-glob 'bilingual-literature/Charles_Perrault/Puss_in_Boots/*.html' \
  --include-glob 'bilingual-literature/Charles_Perrault/The_Sleeping_Beauty/*.html' \
  "${EXTRA_ARGS[@]}"

run_builder stories \
  --mode plain \
  --min-words 4 \
  --include-glob 'bilingual-literature/Lewis_Carroll/Alice_s_Adventures_in_Wonderland/*.en.xml' \
  --include-glob 'bilingual-literature/A_C_Doyle/The_Adventures_of_Sherlock_Holmes/*.html' \
  --exclude-glob 'bilingual-literature/Lewis_Carroll/Alice_s_Adventures_in_Wonderland/*-00.en.xml' \
  --exclude-glob 'bilingual-literature/A_C_Doyle/The_Adventures_of_Sherlock_Holmes/*.all.html' \
  --exclude-glob 'bilingual-literature/A_C_Doyle/The_Adventures_of_Sherlock_Holmes/*.index.html' \
  --exclude-glob 'bilingual-literature/A_C_Doyle/The_Adventures_of_Sherlock_Holmes/The_Adventures_of_Sherlock_Holmes.html' \
  "${EXTRA_ARGS[@]}"

run_builder literature \
  --mode plain \
  --min-words 5 \
  --include-glob 'shakespeare/hamlet/full.html' \
  --include-glob 'shakespeare/macbeth/full.html' \
  --include-glob 'shakespeare/tempest/full.html' \
  "${EXTRA_ARGS[@]}"

python - "$OUTPUT" "$BIBLE_CSV" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
bible_csv = sys.argv[2]
categories = ["dictionary", "bible", "fables", "stories", "literature"]
combined = {
    "root": str(output),
    "bible_csv": bible_csv,
    "categories": {},
    "next_steps": {
        "tokenize": f"./tokenize_compatible_corpus.sh {output} learned/corpus_structured_v2_compatible",
        "train": "./continue_structured_v2.sh",
        "eval": "./eval_structured_v2.sh",
    },
}

for category in categories:
    manifest_path = output / category / "manifest.json"
    if not manifest_path.exists():
        continue
    combined["categories"][category] = json.loads(manifest_path.read_text(encoding="utf-8"))

(output / "manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"combined manifest: {output / 'manifest.json'}")
PY
