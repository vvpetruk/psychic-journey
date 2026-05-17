from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_length: int) -> list[int]:
        words = text.split()
        ids = [abs(hash(word)) % (self.vocab_size - 1) + 1 for word in words[:max_length]]
        if len(ids) < 2:
            ids = ids + [1] * (2 - len(ids))
        return ids


@dataclass
class TrainConfig:
    steps: int = 20
    batch_size: int = 2
    seq_len: int = 96
    lr: float = 3e-4
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    vocab_size: int = 32000
    chunk_size: int = 32
    window_size: int = 32
    seed: int = 42


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


def make_batch(texts: list[str], tokenizer: SimpleTokenizer, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_inputs = []
    batch_targets = []
    for _ in range(batch_size):
        text = random.choice(texts)
        ids = tokenizer.encode(text, max_length=seq_len + 1)
        if len(ids) < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - len(ids))
        else:
            ids = ids[: seq_len + 1]
        batch_inputs.append(ids[:-1])
        batch_targets.append(ids[1:])
    return (
        torch.tensor(batch_inputs, dtype=torch.long),
        torch.tensor(batch_targets, dtype=torch.long),
    )


def main() -> None:
    cfg = TrainConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_path = Path('data/processed/wikitext-103/train.jsonl')
    texts = load_texts(dataset_path)
    tokenizer = SimpleTokenizer(vocab_size=cfg.vocab_size)

    model = MemoryMAC(
        MemoryMACConfig(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
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
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        print({
            'step': step,
            'loss': loss_value,
            'ppl_est': math.exp(min(loss_value, 20)),
            'device': device,
        })

    print({
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'device': device,
        'steps': cfg.steps,
    })


if __name__ == '__main__':
    main()
