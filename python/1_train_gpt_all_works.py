#!/usr/bin/env python3
"""
Train a small GPT-style decoder-only Transformer on all_works token IDs.

Inputs expected from step 0:
- learned/all_works_tokenized/all_works_token_ids.json
- learned/all_works_tokenized/all_works_vocab.json
- learned/all_works_tokenized/all_works_meta.json

Outputs written under learned/gpt_all_works/ by default:
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
    token_dir: str = "learned/all_works_tokenized"
    output_dir: str = "learned/gpt_all_works"
    resume_from: str | None = None
    context_len: int = 512
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    train_fraction: float = 0.98
    max_vocab_size: int = 65536
    tie_embeddings: bool = True
    eval_batches: int = 800
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    compile_model: bool = False
    auto_batch_size: bool = False
    auto_batch_max: int = 128
    auto_batch_target_vram_gib: float = 8.0
    log_every: int = 2000
    save_every: int = 6000
    seed: int = 24823612


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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_token_data(token_dir: Path) -> tuple[list[int], dict, dict]:
    token_ids = json.loads((token_dir / "all_works_token_ids.json").read_text(encoding="utf-8"))
    vocab = json.loads((token_dir / "all_works_vocab.json").read_text(encoding="utf-8"))
    meta = json.loads((token_dir / "all_works_meta.json").read_text(encoding="utf-8"))
    return token_ids, vocab, meta


def prune_token_ids_in_place(token_ids: list[int], vocab: dict, max_vocab_size: int) -> tuple[dict, dict]:
    source_vocab_size = len(vocab["itos"])
    unk_id = int(vocab["stoi"]["<UNK>"])
    max_vocab_size = min(max_vocab_size, source_vocab_size)

    if max_vocab_size <= unk_id:
        raise ValueError("max_vocab_size must keep the special tokens, including <UNK>")

    effective_vocab = {
        "itos": vocab["itos"][:max_vocab_size],
        "stoi": {token: idx for idx, token in enumerate(vocab["itos"][:max_vocab_size])},
    }

    replaced_tokens = 0
    if max_vocab_size < source_vocab_size:
        for idx, token_id in enumerate(token_ids):
            if token_id >= max_vocab_size:
                token_ids[idx] = unk_id
                replaced_tokens += 1

    prune_meta = {
        "source_vocab_size": source_vocab_size,
        "effective_vocab_size": max_vocab_size,
        "unk_id": unk_id,
        "replaced_token_count": replaced_tokens,
        "replaced_token_fraction": (replaced_tokens / len(token_ids)) if token_ids else 0.0,
    }
    return effective_vocab, prune_meta


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


def probe_micro_batch_size(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    batch_start: int,
    batch_max: int,
    target_vram_gib: float,
) -> tuple[int, float]:
    if device.type != "cuda":
        return batch_start, 0.0

    target_bytes = int(target_vram_gib * (1024 ** 3))
    low = max(1, batch_start)
    high = max(low, batch_max)
    best_batch = low
    best_peak_gib = 0.0
    amp_device = "cuda"

    def run_probe(candidate: int) -> tuple[bool, float]:
        try:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            x_list = []
            y_list = []
            for idx in range(candidate):
                x_item, y_item = dataset[idx % len(dataset)]
                x_list.append(x_item)
                y_list.append(y_item)

            x = torch.stack(x_list).to(device, non_blocking=True)
            y = torch.stack(y_list).to(device, non_blocking=True)

            with torch.autocast(device_type=amp_device, enabled=use_amp, dtype=amp_dtype):
                _, loss = model(x, y)

            loss.backward()
            torch.cuda.synchronize(device)
            peak = float(torch.cuda.max_memory_allocated(device))
            optimizer.zero_grad(set_to_none=True)
            del x, y, loss, x_list, y_list
            return True, peak
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False, float("inf")

    while low <= high:
        candidate = (low + high) // 2
        ok, peak = run_probe(candidate)
        if ok and peak <= target_bytes:
            best_batch = candidate
            best_peak_gib = peak / (1024 ** 3)
            low = candidate + 1
        else:
            high = candidate - 1

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return best_batch, best_peak_gib


def compile_model_if_possible(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    requested: bool,
) -> tuple[nn.Module, bool, str | None]:
    if not requested:
        return model, False, None

    if not hasattr(torch, "compile"):
        return model, False, "torch.compile is not available in this PyTorch build"

    compiled_model = torch.compile(model)
    if device.type != "cuda":
        return compiled_model, True, None

    try:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        x_list = []
        y_list = []
        warmup_batch = min(2, len(dataset))
        for idx in range(warmup_batch):
            x_item, y_item = dataset[idx]
            x_list.append(x_item)
            y_list.append(y_item)

        x = torch.stack(x_list).to(device, non_blocking=True)
        y = torch.stack(y_list).to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
            _, loss = compiled_model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del x, y, loss, x_list, y_list
        torch.cuda.empty_cache()
        return compiled_model, True, None
    except Exception as exc:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return model, False, f"{type(exc).__name__}: {exc}"


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


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    meta: dict,
    epoch: int,
    step: int,
    best_val_loss: float,
    epoch_end: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_config": asdict(config),
            "meta": meta,
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
            "epoch_end": epoch_end,
        },
        path,
    )


def validate_resume_compatibility(config: TrainConfig, checkpoint_config: dict) -> None:
    fields_to_match = [
        "context_len",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "dropout",
        "max_vocab_size",
        "tie_embeddings",
    ]
    mismatches = []
    for field in fields_to_match:
        if checkpoint_config.get(field) != getattr(config, field):
            mismatches.append(
                f"{field}: current={getattr(config, field)!r} checkpoint={checkpoint_config.get(field)!r}"
            )

    if mismatches:
        raise ValueError(
            "resume checkpoint is not compatible with the requested model/token setup:\n"
            + "\n".join(mismatches)
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, max_batches: int) -> float:
    model.eval()
    losses = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device, enabled=use_amp, dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small GPT-style model on all_works tokens")
    parser.add_argument("--token-dir", default="learned/all_works_tokenized")
    parser.add_argument("--output-dir", default="learned/gpt_all_works")
    parser.add_argument("--resume-from", default=None, help="Resume training from a checkpoint path")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size that must fit in VRAM")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Number of micro-batches per optimizer step")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--max-vocab-size", type=int, default=32768, help="Keep only the most frequent tokens from step 0 and map the rest to <UNK>")
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-batches", type=int, default=200, help="Validation batches per evaluation pass")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU workers for DataLoader")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per worker when num_workers > 0")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False, help="Use torch.compile for the model")
    parser.add_argument("--auto-batch-size", action=argparse.BooleanOptionalAction, default=False, help="Probe for a larger safe micro-batch size on CUDA before training")
    parser.add_argument("--auto-batch-max", type=int, default=128, help="Upper bound for auto batch size probing")
    parser.add_argument("--auto-batch-target-vram-gib", type=float, default=8.0, help="Target PyTorch peak VRAM for auto batch size probing")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> int:
    config = parse_args()
    if config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if config.eval_batches < 1:
        raise ValueError("eval_batches must be at least 1")
    if config.max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2")
    if config.num_workers < 0:
        raise ValueError("num_workers must be at least 0")
    if config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    if config.auto_batch_max < 1:
        raise ValueError("auto_batch_max must be at least 1")
    if config.auto_batch_target_vram_gib <= 0:
        raise ValueError("auto_batch_target_vram_gib must be greater than 0")

    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    token_dir = Path(config.token_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_ids, vocab, meta = load_token_data(token_dir)
    effective_vocab, prune_meta = prune_token_ids_in_place(token_ids, vocab, config.max_vocab_size)
    train_ids, val_ids = split_token_ids(token_ids, config.train_fraction)

    train_ds = TokenSequenceDataset(train_ids, config.context_len)
    val_ds = TokenSequenceDataset(val_ids, config.context_len)

    vocab_size = len(effective_vocab["itos"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler_enabled = use_amp and amp_dtype == torch.float16

    model = GPTSmall(
        vocab_size=vocab_size,
        context_len=config.context_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        tie_embeddings=config.tie_embeddings,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    resume_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if config.resume_from is not None:
        resume_path = Path(config.resume_from)
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_config = checkpoint.get("train_config", {})
        validate_resume_compatibility(config, checkpoint_config)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        scaler_state = checkpoint.get("scaler_state")
        if scaler_state:
            scaler.load_state_dict(scaler_state)

        global_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_epoch_end = bool(checkpoint.get("epoch_end", False))
        resume_epoch = checkpoint_epoch if checkpoint_epoch_end else max(0, checkpoint_epoch - 1)
        print(f"resuming from:    {resume_path}")
        print(f"resume step:      {global_step}")
        print(f"resume epoch:     {resume_epoch + 1}")

    auto_batch_peak_gib = 0.0
    if config.auto_batch_size:
        selected_batch_size, auto_batch_peak_gib = probe_micro_batch_size(
            model=model,
            optimizer=optimizer,
            dataset=train_ds,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            batch_start=config.batch_size,
            batch_max=config.auto_batch_max,
            target_vram_gib=config.auto_batch_target_vram_gib,
        )
        config.batch_size = selected_batch_size

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "pin_memory": pin_memory,
        "num_workers": config.num_workers,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    train_model, compile_enabled, compile_error = compile_model_if_possible(
        model=model,
        optimizer=optimizer,
        dataset=train_ds,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        requested=config.compile_model,
    )

    source_vocab_size = prune_meta["source_vocab_size"]
    bytes_per_logit = 2 if use_amp else 4
    approx_source_logits_gib = (
        config.batch_size * config.context_len * source_vocab_size * bytes_per_logit / (1024 ** 3)
    )
    approx_effective_logits_gib = (
        config.batch_size * config.context_len * vocab_size * bytes_per_logit / (1024 ** 3)
    )

    model_config = {
        "train_config": asdict(config),
        "vocab_size": vocab_size,
        "token_meta": meta,
        "vocab_prune": prune_meta,
        "device": str(device),
    }
    save_json(output_dir / "config.json", model_config)
    save_json(output_dir / "effective_vocab.json", effective_vocab)

    print(f"device:           {device}")
    print(f"cuda enabled:     {torch.cuda.is_available()}")
    print(f"token count:      {len(token_ids)}")
    print(f"train tokens:     {len(train_ids)}")
    print(f"val tokens:       {len(val_ids)}")
    print(f"source vocab:     {source_vocab_size}")
    print(f"vocab size:       {vocab_size}")
    print(f"context len:      {config.context_len}")
    print(f"d_model:          {config.d_model}")
    print(f"heads:            {config.n_heads}")
    print(f"layers:           {config.n_layers}")
    print(f"batch size:       {config.batch_size}")
    print(f"grad accum:       {config.grad_accum_steps}")
    print(f"effective batch:  {config.batch_size * config.grad_accum_steps}")
    print(f"num workers:      {config.num_workers}")
    print(f"persistent work:  {config.persistent_workers if config.num_workers > 0 else False}")
    print(f"prefetch factor:  {config.prefetch_factor if config.num_workers > 0 else 0}")
    print(f"compile model:    {compile_enabled}")
    print(f"auto batch size:  {config.auto_batch_size}")
    print(f"auto batch max:   {config.auto_batch_max}")
    print(f"auto batch tgt:   {config.auto_batch_target_vram_gib}")
    print(f"auto batch peak:  {auto_batch_peak_gib:.2f}")
    if compile_error is not None:
        print(f"compile fallback: {compile_error}")
    print(f"tie embeddings:   {config.tie_embeddings}")
    print(f"amp dtype:        {amp_dtype}")
    print(f"replaced tokens:  {prune_meta['replaced_token_count']} ({prune_meta['replaced_token_fraction']:.2%})")
    print(f"logits GiB raw:   {approx_source_logits_gib:.2f}")
    print(f"logits GiB used:  {approx_effective_logits_gib:.2f}")
    print(f"epochs:           {config.epochs}")
    print(f"learning rate:    {config.learning_rate}")
    start_time = time.time()

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    target_epoch = resume_epoch + config.epochs

    for epoch in range(resume_epoch, target_epoch):
        train_model.train()
        running_loss = 0.0
        running_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=amp_device, enabled=use_amp, dtype=amp_dtype):
                _, loss = train_model(x, y)

            running_loss += loss.item()
            running_steps += 1

            scaled_loss = loss / config.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            should_step = (
                batch_idx % config.grad_accum_steps == 0
                or batch_idx == len(train_loader)
            )
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % config.log_every == 0:
                avg_train_loss = running_loss / max(1, running_steps)
                val_loss = evaluate(train_model, val_loader, device, use_amp, amp_dtype, config.eval_batches)
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
                    f"epoch {epoch + 1}/{target_epoch} step {global_step} "
                    f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} elapsed={elapsed:.1f}s"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_training_checkpoint(
                        output_dir / "checkpoint.pt",
                        model,
                        optimizer,
                        scaler,
                        config,
                        meta,
                        epoch + 1,
                        global_step,
                        best_val_loss,
                        False,
                    )

            if global_step % config.save_every == 0:
                save_training_checkpoint(
                    output_dir / f"checkpoint_step_{global_step}.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    meta,
                    epoch + 1,
                    global_step,
                    best_val_loss,
                    False,
                )
                save_training_checkpoint(
                    output_dir / "checkpoint_latest.pt",
                    model,
                    optimizer,
                    scaler,
                    config,
                    meta,
                    epoch + 1,
                    global_step,
                    best_val_loss,
                    False,
                )

        epoch_train_loss = running_loss / max(1, running_steps)
        epoch_val_loss = evaluate(train_model, val_loader, device, use_amp, amp_dtype, config.eval_batches)
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
            f"epoch {epoch + 1}/{target_epoch} complete "
            f"train_loss={epoch_train_loss:.6f} val_loss={epoch_val_loss:.6f} elapsed={elapsed:.1f}s"
        )

        save_training_checkpoint(
            output_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            scaler,
            config,
            meta,
            epoch + 1,
            global_step,
            best_val_loss,
            True,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_training_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                optimizer,
                scaler,
                config,
                meta,
                epoch + 1,
                global_step,
                best_val_loss,
                True,
            )

    save_training_checkpoint(
        output_dir / "all_works_checkpoint_final.pt",
        model,
        optimizer,
        scaler,
        config,
        meta,
        target_epoch,
        global_step,
        best_val_loss,
        True,
    )

    total_time = time.time() - start_time
    print(f"training complete in {total_time:.1f}s")
    print(f"best val loss: {best_val_loss:.6f}")
    print(f"checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
