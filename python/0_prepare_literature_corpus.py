#!/usr/bin/env python3
"""
Build a cleaner training corpus from literature/ with optional Q/A records.

This script is designed to replace the "one giant raw text file" approach with
format-aware extraction and sharded outputs. It can emit:

- plain: STARTTOK <clean sentence> ENDTOK
- qa:    STARTTOK Question: ... Answer: ... ENDTOK
- mixed: both plain and qa records when structured sources permit it

Outputs:
- <output_dir>/train/shard_00000.txt
- <output_dir>/val/shard_00000.txt
- <output_dir>/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".xml", ".csv", ".json"}
TEXT_KEYS = {"text", "content", "body", "description", "definition", "paragraph", "line"}
PATH_SKIP_TERMS = (
    "/.git/",
    "readme",
    "license",
    "package.json",
    "package-lock.json",
    "poetry.lock",
    "yarn.lock",
    "tree.txt",
    "rules.md",
    "/docs/",
    "/main_readme/",
    "/node_modules/",
    "__pycache__",
)
BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"provided by the internet classics archive",
        r"available online at",
        r"click here for",
        r"google cache",
        r"see bottom for copyright",
        r"if you have any questions about the perseus project texts",
        r"translation list",
        r"project layout",
        r"adding texts",
        r"scripting",
        r"schema",
        r"shakespeare homepage",
        r"entire play",
        r"project gutenberg",
        r"ebook #\d+",
        r"produced by",
        r"translation by",
        r"translated by",
        r"table of contents",
        r"dependencies",
        r"devdependencies",
        r"lint-staged",
    ]
]

SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?>.*?</\1>", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BLOCK_TAG_RE = re.compile(r"</?(?:p|div|br|li|tr|td|th|h[1-6]|blockquote|section|article)\b[^>]*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
ENGLISH_HTML_CELL_RE = re.compile(
    r"<td\b[^>]*class=['\"][^'\"]*\ben\b[^'\"]*['\"][^>]*>(.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)
ENGLISH_HTML_BLOCK_RE = re.compile(
    r"<(?:div|p|h[1-6]|span)\b[^>]*class=['\"][^'\"]*\ben\b[^'\"]*['\"][^>]*>(.*?)</(?:div|p|h[1-6]|span)>",
    re.IGNORECASE | re.DOTALL,
)
URL_RE = re.compile(r"\b(?:https?|ftp)://\S+|\bwww\.\S+", re.IGNORECASE)
VERSE_REF_RE = re.compile(r"\[(?:\d+:\d+|\d+)\]")
BRACKET_TAG_RE = re.compile(r"\[(?:gap[^\]]*|FI|Fi|font[^\]]*|/?[A-Za-z]+=[^\]]*)\]")
REPEATED_DASH_RE = re.compile(r"-{2,}")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
NON_ALLOWED_ASCII_RE = re.compile(r"[^A-Za-z0-9\s\.,:\-\'\"()\?!;]")
LOWER_TO_CAPITALIZED_WORD_RE = re.compile(r"\b([a-z]{2,})([A-Z][a-z]+)\b")
HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\s+([a-z])")
OCR_NOISE_RE = re.compile(r"\b(?:FI|Fi)\b")

TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


@dataclass
class BuildConfig:
    root_dir: str
    output_dir: str
    mode: str
    qa_style: str
    records_per_shard: int
    val_fraction: float
    min_words: int
    ascii_only: bool
    limit_files: int | None
    max_records: int | None
    max_file_bytes: int
    include_glob: list[str]
    exclude_glob: list[str]


class ShardWriter:
    def __init__(self, output_dir: Path, records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.records_per_shard = records_per_shard
        self.handles: dict[str, object] = {}
        self.counts = Counter()
        self.shard_indices = defaultdict(int)

    def _open_handle(self, split: str):
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = self.shard_indices[split]
        path = split_dir / f"shard_{shard_idx:05d}.txt"
        self.handles[split] = path.open("w", encoding="utf-8")
        self.counts[f"{split}_in_shard"] = 0

    def write(self, split: str, record: str) -> None:
        if split not in self.handles:
            self._open_handle(split)
        if self.counts[f"{split}_in_shard"] >= self.records_per_shard:
            self.handles[split].close()
            self.shard_indices[split] += 1
            self._open_handle(split)

        self.handles[split].write(record + "\n")
        self.counts[f"{split}_records"] += 1
        self.counts[f"{split}_in_shard"] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build cleaned literature corpus shards")
    parser.add_argument("root_dir", nargs="?", default="literature")
    parser.add_argument("output_dir", nargs="?", default="training/literature_corpus")
    parser.add_argument("--mode", choices=["plain", "qa", "mixed"], default="mixed")
    parser.add_argument("--qa-style", choices=["chat", "qa"], default="chat")
    parser.add_argument("--records-per-shard", type=int, default=50000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--ascii-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-file-bytes", type=int, default=25_000_000)
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help="Optional glob relative to root_dir. Repeat to target subsets.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Optional glob relative to root_dir to remove unwanted subsets.",
    )
    args = parser.parse_args()
    return BuildConfig(**vars(args))


def should_skip_path(path: Path, root_dir: Path, include_glob: list[str], exclude_glob: list[str]) -> bool:
    relative = path.relative_to(root_dir).as_posix()
    lowered = relative.lower()

    if include_glob and not any(fnmatch.fnmatch(relative, pattern) for pattern in include_glob):
        return True
    if exclude_glob and any(fnmatch.fnmatch(relative, pattern) for pattern in exclude_glob):
        return True

    if path.suffix.lower() not in ALLOWED_EXTENSIONS and not relative.endswith(".json.txt"):
        return True

    return any(term in lowered for term in PATH_SKIP_TERMS)


def iter_candidate_files(config: BuildConfig, root_dir: Path) -> Iterable[Path]:
    count = 0
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if should_skip_path(path, root_dir, config.include_glob, config.exclude_glob):
            continue
        yield path
        count += 1
        if config.limit_files is not None and count >= config.limit_files:
            break


def stable_split_key(source_id: str, val_fraction: float) -> str:
    digest = hashlib.md5(source_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return "val" if bucket < val_fraction else "train"


def strip_markup(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = BLOCK_TAG_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def extract_english_bilingual_html(text: str) -> str:
    blocks = ENGLISH_HTML_CELL_RE.findall(text)
    if not blocks:
        blocks = ENGLISH_HTML_BLOCK_RE.findall(text)
    if not blocks:
        return text
    return "\n".join(blocks)


def preprocess_source_text(path: Path, text: str) -> str:
    relative = path.as_posix().lower()
    if "bilingual-literature/" in relative and path.suffix.lower() == ".html":
        return extract_english_bilingual_html(text)
    return text


def normalize_text(text: str, ascii_only: bool) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATION_TABLE)
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = URL_RE.sub(" ", text)
    text = VERSE_REF_RE.sub(" ", text)
    text = BRACKET_TAG_RE.sub(" ", text)
    text = OCR_NOISE_RE.sub(" ", text)
    text = REPEATED_DASH_RE.sub("-", text)
    text = LOWER_TO_CAPITALIZED_WORD_RE.sub(r"\1 \2", text)

    if ascii_only:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = NON_ALLOWED_ASCII_RE.sub(" ", text)

    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def remove_boilerplate_lines(text: str) -> str:
    kept = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def split_sentences(text: str, min_words: int) -> list[str]:
    chunks = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    for chunk in chunks:
        sentence = chunk.strip(" -")
        if not sentence:
            continue
        words = WORD_RE.findall(sentence)
        if len(words) < min_words:
            continue
        alpha_chars = sum(ch.isalpha() for ch in sentence)
        non_space_chars = sum(not ch.isspace() for ch in sentence)
        if non_space_chars == 0:
            continue
        if alpha_chars / non_space_chars < 0.55:
            continue
        if any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS):
            continue
        sentences.append(sentence)
    return sentences


def make_plain_record(sentence: str) -> str:
    return f"STARTTOK {sentence} ENDTOK"


def make_qa_record(question: str, answer: str, qa_style: str) -> str:
    if qa_style == "chat":
        return f"STARTTOK User: {question} Assistant: {answer} ENDTOK"
    return f"STARTTOK Question: {question} Answer: {answer} ENDTOK"


def clean_and_split_text(text: str, ascii_only: bool, min_words: int, markup: bool) -> list[str]:
    if markup:
        text = strip_markup(text)
    text = remove_boilerplate_lines(text)
    text = normalize_text(text, ascii_only=ascii_only)
    return split_sentences(text, min_words=min_words)


def extract_generic_records(path: Path, config: BuildConfig) -> list[tuple[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = preprocess_source_text(path, raw)
    markup = path.suffix.lower() in {".html", ".xml"}

    if path.suffix.lower() == ".md":
        raw = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
        raw = re.sub(r"`[^`]+`", " ", raw)
        raw = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw, flags=re.MULTILINE)

    sentences = clean_and_split_text(raw, ascii_only=config.ascii_only, min_words=config.min_words, markup=markup)
    records = []
    if config.mode in {"plain", "mixed"}:
        records.extend(("plain", make_plain_record(sentence)) for sentence in sentences)
    return records


def extract_csv_records(path: Path, config: BuildConfig) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return records

        fieldnames = {name.lower() for name in reader.fieldnames if name}
        is_bible = {"book", "chapter", "verse", "text"}.issubset(fieldnames)

        for row in reader:
            if is_bible:
                book = (row.get("Book") or row.get("book") or "").strip()
                chapter = (row.get("Chapter") or row.get("chapter") or "").strip()
                verse = (row.get("Verse") or row.get("verse") or "").strip()
                raw_text = row.get("Text") or row.get("text") or ""
                sentences = clean_and_split_text(raw_text, ascii_only=config.ascii_only, min_words=config.min_words, markup=False)
                joined = " ".join(sentences).strip()
                if not joined:
                    continue
                if config.mode in {"plain", "mixed"}:
                    records.append(("bible_plain", make_plain_record(joined)))
                if config.mode in {"qa", "mixed"} and book and chapter and verse:
                    question = f"What does {book} {chapter}:{verse} say?"
                    records.append(("bible_qa", make_qa_record(question, joined, config.qa_style)))
                continue

            text_field = next((name for name in reader.fieldnames if name and name.lower() in TEXT_KEYS), None)
            if text_field is None:
                continue
            raw_text = row.get(text_field) or ""
            sentences = clean_and_split_text(raw_text, ascii_only=config.ascii_only, min_words=config.min_words, markup=False)
            if config.mode in {"plain", "mixed"}:
                records.extend(("csv_plain", make_plain_record(sentence)) for sentence in sentences)

    return records


def extract_json_records(path: Path, config: BuildConfig) -> list[tuple[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []

    records: list[tuple[str, str]] = []

    if isinstance(data, dict):
        dictionary_like = all(isinstance(key, str) for key in data)
        definition_like = dictionary_like and all(isinstance(value, (str, list)) for value in data.values())
        if definition_like:
            for term, value in data.items():
                definitions = value if isinstance(value, list) else [value]
                for definition in definitions:
                    if not isinstance(definition, str):
                        continue
                    answers = clean_and_split_text(definition, ascii_only=config.ascii_only, min_words=config.min_words, markup=False)
                    joined = " ".join(answers).strip()
                    if not joined:
                        continue
                    if config.mode in {"plain", "mixed"}:
                        records.append(("dict_plain", make_plain_record(f"{term}: {joined}")))
                    if config.mode in {"qa", "mixed"}:
                        records.append(("dict_qa", make_qa_record(f"What is {term}?", joined, config.qa_style)))
            return records

    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue

        lowered_keys = {key.lower(): key for key in item}
        if {"book", "chapter", "verse", "text"}.issubset(lowered_keys):
            book = str(item[lowered_keys["book"]]).strip()
            chapter = str(item[lowered_keys["chapter"]]).strip()
            verse = str(item[lowered_keys["verse"]]).strip()
            raw_text = str(item[lowered_keys["text"]])
            answers = clean_and_split_text(raw_text, ascii_only=config.ascii_only, min_words=config.min_words, markup=False)
            joined = " ".join(answers).strip()
            if not joined:
                continue
            if config.mode in {"plain", "mixed"}:
                records.append(("json_bible_plain", make_plain_record(joined)))
            if config.mode in {"qa", "mixed"}:
                records.append(("json_bible_qa", make_qa_record(f"What does {book} {chapter}:{verse} say?", joined, config.qa_style)))
            continue

        text_values = [
            str(value)
            for key, value in item.items()
            if isinstance(value, str) and key.lower() in TEXT_KEYS
        ]
        for value in text_values:
            sentences = clean_and_split_text(value, ascii_only=config.ascii_only, min_words=config.min_words, markup=False)
            if config.mode in {"plain", "mixed"}:
                records.extend(("json_plain", make_plain_record(sentence)) for sentence in sentences)

    return records


def extract_records(path: Path, config: BuildConfig) -> list[tuple[str, str]]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".json", ".txt"] or path.suffix.lower() == ".json":
        return extract_json_records(path, config)
    if path.suffix.lower() == ".csv":
        return extract_csv_records(path, config)
    return extract_generic_records(path, config)


def main() -> int:
    config = parse_args()
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")
    if config.records_per_shard < 1:
        raise ValueError("records_per_shard must be at least 1")

    root_dir = Path(config.root_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(output_dir, records_per_shard=config.records_per_shard)
    stats = Counter()
    by_kind = Counter()
    skipped = Counter()

    try:
        for path in iter_candidate_files(config, root_dir):
            stats["files_seen"] += 1

            try:
                if path.stat().st_size > config.max_file_bytes:
                    skipped["too_large"] += 1
                    continue
                extracted = extract_records(path, config)
            except Exception:
                skipped["extract_error"] += 1
                continue

            if not extracted:
                skipped["no_records"] += 1
                continue

            stats["files_used"] += 1
            relative = path.relative_to(root_dir).as_posix()
            for index, (kind, record) in enumerate(extracted):
                split = stable_split_key(f"{relative}:{index}", config.val_fraction)
                writer.write(split, record)
                stats["records_written"] += 1
                by_kind[kind] += 1
                if config.max_records is not None and stats["records_written"] >= config.max_records:
                    break

            if config.max_records is not None and stats["records_written"] >= config.max_records:
                break
    finally:
        writer.close()

    manifest = {
        "config": vars(config),
        "stats": dict(stats),
        "writer_counts": dict(writer.counts),
        "record_kinds": dict(by_kind),
        "skipped": dict(skipped),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"root dir:         {root_dir}")
    print(f"output dir:       {output_dir}")
    print(f"mode:             {config.mode}")
    print(f"files seen:       {stats['files_seen']}")
    print(f"files used:       {stats['files_used']}")
    print(f"records written:  {stats['records_written']}")
    print(f"train records:    {writer.counts['train_records']}")
    print(f"val records:      {writer.counts['val_records']}")
    print(f"skipped:          {dict(skipped)}")
    print(f"manifest:         {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
