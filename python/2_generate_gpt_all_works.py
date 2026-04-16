#!/usr/bin/env python3
"""
Generate text from a trained GPT-style all_works / mixed-corpus checkpoint.

This supports the newer checkpoint layout:
- learned/gpt_*/checkpoint.pt
- learned/gpt_*/effective_vocab.json

It can also fall back to token_dir/all_works_vocab.json when needed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"STARTTOK|ENDTOK|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,;:!?()\[\]{}\-\"“”‘’]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_prompt_tokens(prompt: str) -> list[str]:
    prompt = prompt.strip()
    if "STARTTOK" in prompt or "Assistant:" in prompt or "User:" in prompt:
        return tokenize(prompt)
    return ["STARTTOK", "User", ":"] + tokenize(prompt) + ["Assistant", ":"]


def strip_leading_answer_markers(tokens: list[str]) -> list[str]:
    cleaned = list(tokens)
    while cleaned and cleaned[0] in {":", "-", ";"}:
        cleaned.pop(0)
    if len(cleaned) >= 2 and cleaned[0] == "Assistant" and cleaned[1] == ":":
        cleaned = cleaned[2:]
    while cleaned and cleaned[0] in {":", "-", ";"}:
        cleaned.pop(0)
    return cleaned


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, context_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        mask = self.causal_mask[:seq_len, :seq_len]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, context_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, context_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTSmall(nn.Module):
    def __init__(self, vocab_size: int, context_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float, tie_embeddings: bool) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, context_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        tok = self.token_emb(x)
        pos = self.pos_emb(positions)
        h = self.dropout(tok + pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt, ckpt["train_config"]


def resolve_vocab(args: argparse.Namespace, checkpoint_path: Path) -> dict:
    if args.vocab_json is not None:
        return json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))

    checkpoint_vocab = checkpoint_path.parent / "effective_vocab.json"
    if checkpoint_vocab.exists():
        return json.loads(checkpoint_vocab.read_text(encoding="utf-8"))

    token_dir = Path(args.token_dir)
    vocab_path = token_dir / "all_works_vocab.json"
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def build_model(train_config: dict, vocab_size: int) -> GPTSmall:
    return GPTSmall(
        vocab_size=vocab_size,
        context_len=train_config["context_len"],
        d_model=train_config["d_model"],
        n_heads=train_config["n_heads"],
        n_layers=train_config["n_layers"],
        d_ff=train_config["d_ff"],
        dropout=train_config["dropout"],
        tie_embeddings=train_config.get("tie_embeddings", True),
    )


def sample_next_token(logits: torch.Tensor, temperature: float, mode: str, top_k: int | None) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = F.softmax(values, dim=-1)
        sampled = int(torch.multinomial(probs, num_samples=1).item())
        return int(indices[sampled].item())
    if mode == "greedy":
        return int(torch.argmax(logits, dim=-1).item())
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def tokens_to_text(tokens: list[str]) -> str:
    out = []
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}", "”", "’"}
    no_space_after = {"(", "[", "{", "“", "‘"}

    for token in tokens:
        if not out:
            out.append(token)
            continue
        if token in no_space_before:
            out[-1] = out[-1] + token
        elif out[-1] in no_space_after:
            out[-1] = out[-1] + token
        else:
            out.append(" " + token)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a GPT-style all_works / mixed-corpus checkpoint")
    parser.add_argument("checkpoint", nargs="?", default="learned/gpt_all_works/checkpoint_latest.pt")
    parser.add_argument("prompt", nargs="?", default="What is love?")
    parser.add_argument("max_new_tokens", nargs="?", type=int, default=80)
    parser.add_argument("temperature", nargs="?", type=float, default=0.9)
    parser.add_argument("mode", nargs="?", choices=["greedy", "random"], default="random")
    parser.add_argument("seed", nargs="?", type=int, default=12345)
    parser.add_argument("--token-dir", default="learned/all_works_tokenized")
    parser.add_argument("--vocab-json", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--show-full", action="store_true", help="Print the full generated record instead of only the first assistant answer")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint)
    vocab = resolve_vocab(args, checkpoint_path)
    stoi = vocab["stoi"]
    itos = vocab["itos"]

    ckpt, train_config = load_checkpoint(checkpoint_path)
    model = build_model(train_config, len(itos))
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_tokens = build_prompt_tokens(args.prompt)
    unk_id = stoi.get("<UNK>", 1)
    generated_ids = [stoi.get(tok, unk_id) for tok in prompt_tokens]
    prompt_len = len(generated_ids)

    context_len = train_config["context_len"]
    stop_reason = None
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x_tensor = torch.tensor([generated_ids[-context_len:]], dtype=torch.long, device=device)
            logits = model(x_tensor)
            next_logits = logits[0, -1, :]
            next_id = sample_next_token(next_logits, args.temperature, args.mode, args.top_k)
            generated_ids.append(next_id)

            generated_tokens = [itos[token_id] for token_id in generated_ids[prompt_len:]]
            if generated_tokens and generated_tokens[-1] == "ENDTOK":
                stop_reason = "endtok"
                break

            if len(generated_tokens) >= 2 and generated_tokens[-2:] == ["User", ":"]:
                generated_ids = generated_ids[:-2]
                stop_reason = "new_user_turn"
                break

    full_tokens = []
    for token_id in generated_ids:
        tok = itos[token_id]
        if tok == "STARTTOK":
            continue
        if tok == "ENDTOK":
            full_tokens.append("\n")
            continue
        full_tokens.append(tok)

    answer_tokens = []
    for token_id in generated_ids[prompt_len:]:
        tok = itos[token_id]
        if tok in {"STARTTOK", "ENDTOK"}:
            continue
        answer_tokens.append(tok)
    answer_tokens = strip_leading_answer_markers(answer_tokens)

    text_parts = []
    line_tokens = []
    source_tokens = full_tokens if args.show_full else answer_tokens
    for tok in source_tokens:
        if tok == "\n":
            if line_tokens:
                text_parts.append(tokens_to_text(line_tokens))
                line_tokens = []
        else:
            line_tokens.append(tok)
    if line_tokens:
        text_parts.append(tokens_to_text(line_tokens))

    output = "\n".join(text_parts).strip()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
