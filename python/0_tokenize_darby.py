#!/usr/bin/env python3
"""
Tokenize training/Darby.verses.txt for a small GPT-style model.

Goals:
- keep the pipeline readable
- preserve STARTTOK and ENDTOK markers
- split words and punctuation into separate tokens
- build a vocabulary from the corpus itself
- save token IDs and metadata for later training/generation

Output files are written under learned/darby_tokenized/ by default:
- vocab.json
- token_ids.json
- meta.json
- tokens_preview.txt
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
    return path.read_text(encoding="utf-8", errors="replace")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_vocab(tokens: list[str], min_freq: int) -> tuple[list[str], dict[str, int]]:
    counts = Counter(tokens)

    vocab = list(SPECIAL_TOKENS)
    for token, count in counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab.append(token)

    stoi = {token: idx for idx, token in enumerate(vocab)}
    return vocab, stoi


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
    parser = argparse.ArgumentParser(description="Tokenize Darby.verses.txt for GPT-style training")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="training/Darby.verses.txt",
        help="Input text file (default: training/Darby.verses.txt)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="learned/darby_tokenized",
        help="Output directory (default: learned/darby_tokenized)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum token frequency to include in vocab (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    text = read_text(input_path)
    tokens = tokenize(text)
    vocab, stoi = build_vocab(tokens, args.min_freq)
    ids = encode_tokens(tokens, stoi)

    itos = vocab

    meta = {
        "input_file": str(input_path),
        "token_count": len(tokens),
        "vocab_size": len(vocab),
        "min_freq": args.min_freq,
        "special_tokens": SPECIAL_TOKENS,
        "starttok_id": stoi.get("STARTTOK"),
        "endtok_id": stoi.get("ENDTOK"),
        "pad_id": stoi["<PAD>"],
        "unk_id": stoi["<UNK>"],
    }

    write_json(output_dir / "vocab.json", {
        "stoi": stoi,
        "itos": itos,
    })
    write_json(output_dir / "token_ids.json", ids)
    write_json(output_dir / "meta.json", meta)
    write_preview(output_dir / "tokens_preview.txt", tokens)

    print(f"input file:   {input_path}")
    print(f"token count:  {len(tokens)}")
    print(f"vocab size:   {len(vocab)}")
    print(f"output dir:   {output_dir}")
    print(f"STARTTOK id:  {meta['starttok_id']}")
    print(f"ENDTOK id:    {meta['endtok_id']}")
    print(f"preview file: {output_dir / 'tokens_preview.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

