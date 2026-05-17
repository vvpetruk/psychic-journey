from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


@dataclass
class TrainConfig:
    steps: int = 30
    batch_size: int = 2
    seq_len: int = 96
    lr: float = 3e-4
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    chunk_size: int = 32
    window_size: int = 32
    seed: int = 42
    tokenizer_name: str = 'gpt2'


def load_texts(path: Path, limit: int = 2000) -> list[str]:
    texts: list[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            text = (rec.get('text') or '').strip()
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
    if not texts:
        raise RuntimeError(f'No texts found in {path}')
    return texts


def make_batch(texts: list[str], tokenizer: AutoTokenizer, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_batches = []
    target_batches = []
    for _ in range(batch_size):
        text = random.choice(texts)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=seq_len + 1,
            return_tensors='pt',
        )['input_ids'][0]
        if enc.numel() < 2:
            enc = torch.tensor([tokenizer.eos_token_id, tokenizer.eos_token_id], dtype=torch.long)
        if enc.numel() < seq_len + 1:
            pad_id = tokenizer.eos_token_id
            enc = torch.cat([enc, torch.full((seq_len + 1 - enc.numel(),), pad_id, dtype=torch.long)])
        else:
            enc = enc[: seq_len + 1]
        input_batches.append(enc[:-1])
        target_batches.append(enc[1:])
    return torch.stack(input_batches), torch.stack(target_batches)


def main() -> None:
    cfg = TrainConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_path = Path('data/processed/wikitext-103/train.jsonl')
    texts = load_texts(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = int(tokenizer.vocab_size)

    model = MemoryMAC(
        MemoryMACConfig(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            vocab_size=vocab_size,
            max_context_length=cfg.seq_len,
            chunk_size=cfg.chunk_size,
            window_size=cfg.window_size,
        )
    )
    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend is not ready: {model.titans_status.error}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = model.live_backend.to(device)
    backend.train()

    optimizer = torch.optim.AdamW(backend.parameters(), lr=cfg.lr)

    losses = []
    for step in range(1, cfg.steps + 1):
        x, y = make_batch(texts, tokenizer, cfg.batch_size, cfg.seq_len)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _states = backend(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        print({
            'step': step,
            'loss': loss_value,
            'ppl_est': math.exp(min(loss_value, 20)),
            'device': device,
            'python': sys.executable,
            'tokenizer': cfg.tokenizer_name,
            'vocab_size': vocab_size,
        })

    print({
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'device': device,
        'steps': cfg.steps,
        'python': sys.executable,
        'tokenizer': cfg.tokenizer_name,
        'vocab_size': vocab_size,
    })


if __name__ == '__main__':
    main()
