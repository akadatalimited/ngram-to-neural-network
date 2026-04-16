#!/usr/bin/env python3
"""
Expand a GPT checkpoint to a larger vocabulary by copying existing token rows
and randomly initializing new rows.

This is the safe way to "add more vocab later" for the current word-level model.
It does not preserve optimizer state, because parameter shapes change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand GPT checkpoint to a larger vocabulary")
    parser.add_argument("checkpoint", help="Existing checkpoint.pt path")
    parser.add_argument("old_vocab_json", help="Old effective vocab JSON used by the checkpoint")
    parser.add_argument("new_vocab_json", help="New larger vocab JSON to expand into")
    parser.add_argument("output_checkpoint", help="Output expanded checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state = checkpoint["model_state"]

    old_vocab = json.loads(Path(args.old_vocab_json).read_text(encoding="utf-8"))
    new_vocab = json.loads(Path(args.new_vocab_json).read_text(encoding="utf-8"))
    old_itos = old_vocab["itos"]
    new_itos = new_vocab["itos"]
    old_stoi = old_vocab["stoi"]

    if len(new_itos) < len(old_itos):
        raise ValueError("new vocab must be at least as large as old vocab")

    token_weight = model_state["token_emb.weight"]
    head_weight = model_state["head.weight"]

    if token_weight.shape[0] != len(old_itos) or head_weight.shape[0] != len(old_itos):
        raise ValueError("checkpoint embedding/head shape does not match old vocab size")

    d_model = token_weight.shape[1]
    new_token_weight = torch.empty(len(new_itos), d_model, dtype=token_weight.dtype)
    torch.nn.init.normal_(new_token_weight, mean=0.0, std=0.02)
    new_head_weight = new_token_weight.clone()

    copied = 0
    for new_idx, token in enumerate(new_itos):
        old_idx = old_stoi.get(token)
        if old_idx is None:
            continue
        new_token_weight[new_idx] = token_weight[old_idx]
        new_head_weight[new_idx] = head_weight[old_idx]
        copied += 1

    model_state["token_emb.weight"] = new_token_weight
    model_state["head.weight"] = new_head_weight

    checkpoint["model_state"] = model_state
    checkpoint["optimizer_state"] = {}
    checkpoint["scaler_state"] = {}
    checkpoint["train_config"]["max_vocab_size"] = len(new_itos)
    checkpoint["meta"]["expanded_from_vocab_size"] = len(old_itos)
    checkpoint["meta"]["expanded_to_vocab_size"] = len(new_itos)
    checkpoint["meta"]["copied_vocab_rows"] = copied

    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"old vocab size:    {len(old_itos)}")
    print(f"new vocab size:    {len(new_itos)}")
    print(f"copied rows:       {copied}")
    print(f"output checkpoint: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
