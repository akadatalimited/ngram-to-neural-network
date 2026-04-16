#!/usr/bin/env python3
"""
Compare two QA evaluation JSON reports and print a compact checkpoint-to-checkpoint summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two QA evaluation reports")
    parser.add_argument("base_report")
    parser.add_argument("candidate_report")
    return parser.parse_args()


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(report: dict) -> tuple[int, int]:
    results = report.get("results", [])
    total_hits = sum(int(item.get("keyword_hits", 0)) for item in results)
    total_keywords = sum(int(item.get("keyword_total", 0)) for item in results)
    return total_hits, total_keywords


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_report)
    candidate_path = Path(args.candidate_report)

    base = load_report(base_path)
    candidate = load_report(candidate_path)
    base_results = {item["prompt"]: item for item in base.get("results", [])}
    candidate_results = {item["prompt"]: item for item in candidate.get("results", [])}

    base_hits, base_total = summarize(base)
    cand_hits, cand_total = summarize(candidate)

    print(f"base report:      {base_path}")
    print(f"candidate report: {candidate_path}")
    print(f"base score:       {base_hits}/{base_total}")
    print(f"candidate score:  {cand_hits}/{cand_total}")
    print("")

    prompts = sorted(set(base_results) | set(candidate_results))
    for prompt in prompts:
        base_item = base_results.get(prompt, {})
        cand_item = candidate_results.get(prompt, {})
        print(prompt)
        print(
            "  base:      "
            f"{base_item.get('keyword_hits', 0)}/{base_item.get('keyword_total', 0)} "
            f"echo={base_item.get('echoed_prompt', False)} "
            f"answer={base_item.get('answer', '')!r}"
        )
        print(
            "  candidate: "
            f"{cand_item.get('keyword_hits', 0)}/{cand_item.get('keyword_total', 0)} "
            f"echo={cand_item.get('echoed_prompt', False)} "
            f"answer={cand_item.get('answer', '')!r}"
        )
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
