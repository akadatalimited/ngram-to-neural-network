#!/usr/bin/env python3
"""
Train a small GPT-style decoder-only Transformer on Darby token IDs.

Inputs expected from step 0:
- learned/darby_tokenized/token_ids.json
- learned/darby_tokenized/vocab.json
- learned/darby_tokenized/meta.json

Outputs written under learned/gpt_darby_small/ by default:
- checkpoint.pt
- config.json
- train_log.jsonl

This is intentionally small and readable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainConfig:
    token_dir: str = "learned/darby_tokenized"
    output_dir: str = "learned/gpt_darby_small"
    context_len: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_fraction: float = 0.98
    log_every: int = 200
    save_every: int = 2000
    seed: int = 12345


class TokenSequenceDataset(Dataset):
    def __init__(self, token_ids: list[int], context_len: int) -> None:
        self.token_ids = token_ids
        self.context_len = context_len
        self.length = len(token_ids) - context_len
        if self.length <= 0:
            raise ValueError("token stream too short for chosen context length")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx:idx + self.context_len]
        y = self.token_ids[idx + 1:idx + self.context_len + 1]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


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
    def __init__(self, vocab_size: int, context_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float) -> None:
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

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = x.shape
        if seq_len > self.context_len:
            raise ValueError("sequence length exceeds context length")

        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
        tok = self.token_emb(x)
        pos = self.pos_emb(positions)
        h = self.dropout(tok + pos)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_token_data(token_dir: Path) -> tuple[list[int], dict, dict]:
    token_ids = json.loads((token_dir / "token_ids.json").read_text(encoding="utf-8"))
    vocab = json.loads((token_dir / "vocab.json").read_text(encoding="utf-8"))
    meta = json.loads((token_dir / "meta.json").read_text(encoding="utf-8"))
    return token_ids, vocab, meta


def split_token_ids(token_ids: list[int], train_fraction: float) -> tuple[list[int], list[int]]:
    split_at = int(len(token_ids) * train_fraction)
    split_at = max(2, min(split_at, len(token_ids) - 2))
    return token_ids[:split_at], token_ids[split_at:]


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainConfig, meta: dict, epoch: int, step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": asdict(config),
            "meta": meta,
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    losses = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model on Darby tokens")
    parser.add_argument("--token-dir", default="learned/darby_tokenized")
    parser.add_argument("--output-dir", default="learned/gpt_darby_small")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> int:
    config = parse_args()
    set_seed(config.seed)

    token_dir = Path(config.token_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_ids, vocab, meta = load_token_data(token_dir)
    train_ids, val_ids = split_token_ids(token_ids, config.train_fraction)

    train_ds = TokenSequenceDataset(train_ids, config.context_len)
    val_ds = TokenSequenceDataset(val_ids, config.context_len)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory)

    vocab_size = len(vocab["itos"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = GPTSmall(
        vocab_size=vocab_size,
        context_len=config.context_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_config = {
        "train_config": asdict(config),
        "vocab_size": vocab_size,
        "token_meta": meta,
        "device": str(device),
    }
    save_json(output_dir / "config.json", model_config)

    print(f"device:           {device}")
    print(f"cuda enabled:     {torch.cuda.is_available()}")
    print(f"token count:      {len(token_ids)}")
    print(f"train tokens:     {len(train_ids)}")
    print(f"val tokens:       {len(val_ids)}")
    print(f"vocab size:       {vocab_size}")
    print(f"context len:      {config.context_len}")
    print(f"d_model:          {config.d_model}")
    print(f"heads:            {config.n_heads}")
    print(f"layers:           {config.n_layers}")
    print(f"batch size:       {config.batch_size}")
    print(f"epochs:           {config.epochs}")
    print(f"learning rate:    {config.learning_rate}")

    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_steps = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=amp_device, enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running_loss += loss.item()
            running_steps += 1

            if global_step % config.log_every == 0:
                avg_train_loss = running_loss / max(1, running_steps)
                val_loss = evaluate(model, val_loader, device, use_amp)
                elapsed = time.time() - start_time
                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "elapsed_sec": elapsed,
                }
                append_jsonl(output_dir / "train_log.jsonl", record)
                print(
                    f"epoch {epoch + 1}/{config.epochs} step {global_step} "
                    f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} elapsed={elapsed:.1f}s"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "checkpoint.pt",
                        model,
                        optimizer,
                        config,
                        meta,
                        epoch + 1,
                        global_step,
                        best_val_loss,
                    )

            if global_step % config.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_step_{global_step}.pt",
                    model,
                    optimizer,
                    config,
                    meta,
                    epoch + 1,
                    global_step,
                    best_val_loss,
                )

        epoch_train_loss = running_loss / max(1, running_steps)
        epoch_val_loss = evaluate(model, val_loader, device, use_amp)
        elapsed = time.time() - start_time

        record = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "elapsed_sec": elapsed,
            "epoch_end": True,
        }
        append_jsonl(output_dir / "train_log.jsonl", record)
        print(
            f"epoch {epoch + 1}/{config.epochs} complete "
            f"train_loss={epoch_train_loss:.6f} val_loss={epoch_val_loss:.6f} elapsed={elapsed:.1f}s"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                optimizer,
                config,
                meta,
                epoch + 1,
                global_step,
                best_val_loss,
            )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        model,
        optimizer,
        config,
        meta,
        config.epochs,
        global_step,
        best_val_loss,
    )

    total_time = time.time() - start_time
    print(f"training complete in {total_time:.1f}s")
    print(f"best val loss: {best_val_loss:.6f}")
    print(f"checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

