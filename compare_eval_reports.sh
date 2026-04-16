#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  ./compare_eval_reports.sh BASE_REPORT CANDIDATE_REPORT

Example:
  ./compare_eval_reports.sh \
    evaluation/reports/gpt_corpus_mixed_128k_checkpoint_latest.json \
    evaluation/reports/gpt_corpus_structured_v2_checkpoint_latest.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
  show_help
  exit 0
fi

exec python python/5_compare_eval_reports.py "$1" "$2"
