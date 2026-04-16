#!/usr/bin/env python3
"""
Run a fixed prompt set against a GPT checkpoint and save a small QA report.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT QA outputs on a fixed prompt set")
    parser.add_argument("checkpoint", nargs="?", default="learned/gpt_corpus_mixed_128k/checkpoint_latest.pt")
    parser.add_argument("--token-dir", default="learned/corpus_mixed_tokenized_128k")
    parser.add_argument("--prompt-file", default="evaluation/qa_prompts.jsonl")
    parser.add_argument("--output-dir", default="evaluation/reports")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--mode", choices=["greedy", "random"], default="random")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(json.loads(line))
    return prompts


def run_generator(args: argparse.Namespace, prompt: str) -> str:
    cmd = [
        sys.executable,
        "python/2_generate_gpt_all_works.py",
        args.checkpoint,
        prompt,
        str(args.max_new_tokens),
        str(args.temperature),
        args.mode,
        str(args.seed),
        "--token-dir",
        args.token_dir,
        "--top-k",
        str(args.top_k),
    ]
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def score_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    lowered = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits, len(keywords)


def normalize_text_for_compare(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered.strip()


def write_reports(output_dir: Path, checkpoint: Path, results: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.parent.name + "_" + checkpoint.stem
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    summary = {
        "checkpoint": str(checkpoint),
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# QA Eval for `{checkpoint}`", ""]
    total_hits = 0
    total_keywords = 0
    for result in results:
        total_hits += result["keyword_hits"]
        total_keywords += result["keyword_total"]
        lines.append(f"## {result['prompt']}")
        lines.append(f"- category: `{result['category']}`")
        lines.append(f"- keyword score: `{result['keyword_hits']}/{result['keyword_total']}`")
        lines.append(f"- echoed prompt: `{result['echoed_prompt']}`")
        lines.append(f"- answer: `{result['answer']}`")
        lines.append("")
    if total_keywords > 0:
        lines.insert(2, f"Overall keyword score: `{total_hits}/{total_keywords}`")
        lines.insert(3, "")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()
    prompt_file = Path(args.prompt_file)
    prompts = load_prompts(prompt_file)
    checkpoint = Path(args.checkpoint)

    results = []
    for item in prompts:
        prompt = item["prompt"]
        answer = run_generator(args, prompt)
        keywords = item.get("keywords", [])
        hits, total = score_keywords(answer, keywords)
        echoed_prompt = normalize_text_for_compare(answer) == normalize_text_for_compare(prompt)
        if echoed_prompt:
            hits = 0
        results.append(
            {
                "category": item.get("category", "unknown"),
                "prompt": prompt,
                "answer": answer,
                "keywords": keywords,
                "keyword_hits": hits,
                "keyword_total": total,
                "echoed_prompt": echoed_prompt,
            }
        )

    json_path, md_path = write_reports(Path(args.output_dir), checkpoint, results)

    total_hits = sum(result["keyword_hits"] for result in results)
    total_keywords = sum(result["keyword_total"] for result in results)
    print(f"checkpoint:      {checkpoint}")
    print(f"prompts:         {len(results)}")
    print(f"keyword score:   {total_hits}/{total_keywords}")
    print(f"json report:     {json_path}")
    print(f"markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
