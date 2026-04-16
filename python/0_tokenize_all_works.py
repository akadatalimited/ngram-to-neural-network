#!/usr/bin/env python3
"""
Tokenize training/all_works.txt for a small GPT-style model.

Goals:
- keep the pipeline readable
- preserve STARTTOK and ENDTOK markers
- split words and punctuation into separate tokens
- build a vocabulary from the corpus itself
- save token IDs and metadata for later training/generation

Output files are written under learned/all_works_tokenized/ by default:
- all_works_vocab.json
- all_works_token_ids.json
- all_works_meta.json
- all_works_tokens_preview.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"STARTTOK|ENDTOK|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,;:!?()\[\]{}\-\"“”‘’]", re.UNICODE)

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
]


def read_text(path: Path) -> str:
    if path.is_dir():
        parts = []
        for child in sorted(path.rglob("*.txt")):
            parts.append(child.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_vocab(tokens: list[str], min_freq: int, max_vocab_size: int | None) -> tuple[list[str], dict[str, int]]:
    counts = Counter(tokens)

    vocab = list(SPECIAL_TOKENS)
    for token, count in counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab.append(token)
        if max_vocab_size is not None and len(vocab) >= max_vocab_size:
            break

    stoi = {token: idx for idx, token in enumerate(vocab)}
    return vocab, stoi


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return vocab["itos"], vocab["stoi"]


def encode_tokens(tokens: list[str], stoi: dict[str, int]) -> list[int]:
    unk_id = stoi["<UNK>"]
    return [stoi.get(token, unk_id) for token in tokens]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_preview(path: Path, tokens: list[str], limit: int = 500) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = "\n".join(tokens[:limit]) + "\n"
    path.write_text(preview, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize training/all_works.txt for GPT-style training")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="training/all_works.txt",
        help="Input text file or directory of shard .txt files (default: training/all_works.txt)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="learned/all_works_tokenized",
        help="Output directory (default: learned/all_works_tokenized)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum token frequency to include in vocab before capping (default: 2)",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=32768,
        help="Maximum vocabulary size including special tokens (default: 32768)",
    )
    parser.add_argument(
        "--vocab-json",
        default=None,
        help="Reuse an existing vocab JSON instead of building a new one",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    text = read_text(input_path)
    tokens = tokenize(text)
    if args.vocab_json is not None:
        vocab, stoi = load_vocab(Path(args.vocab_json))
    else:
        vocab, stoi = build_vocab(tokens, args.min_freq, args.max_vocab_size)
    ids = encode_tokens(tokens, stoi)

    itos = vocab
    unk_id = stoi["<UNK>"]
    unk_count = sum(1 for token_id in ids if token_id == unk_id)
    unk_fraction = (unk_count / len(ids)) if ids else 0.0

    meta = {
        "input_file": str(input_path),
        "token_count": len(tokens),
        "vocab_size": len(vocab),
        "min_freq": args.min_freq,
        "max_vocab_size": args.max_vocab_size,
        "vocab_json": args.vocab_json,
        "special_tokens": SPECIAL_TOKENS,
        "starttok_id": stoi.get("STARTTOK"),
        "endtok_id": stoi.get("ENDTOK"),
        "pad_id": stoi["<PAD>"],
        "unk_id": unk_id,
        "unk_count": unk_count,
        "unk_fraction": unk_fraction,
    }

    write_json(output_dir / "all_works_vocab.json", {
        "stoi": stoi,
        "itos": itos,
    })
    write_json(output_dir / "all_works_token_ids.json", ids)
    write_json(output_dir / "all_works_meta.json", meta)
    write_preview(output_dir / "all_works_tokens_preview.txt", tokens)

    print(f"input file:   {input_path}")
    print(f"token count:  {len(tokens)}")
    print(f"vocab size:   {len(vocab)}")
    print(f"unk id:       {unk_id}")
    print(f"unk count:    {unk_count}")
    print(f"unk fraction: {unk_fraction:.6f}")
    print(f"output dir:   {output_dir}")
    print(f"STARTTOK id:  {meta['starttok_id']}")
    print(f"ENDTOK id:    {meta['endtok_id']}")
    print(f"preview file: {output_dir / 'all_works_tokens_preview.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
