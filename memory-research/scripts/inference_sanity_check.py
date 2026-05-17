from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[str]]:
        words = text.split()
        tokens = words[:max_length]
        ids = [abs(hash(word)) % (self.vocab_size - 1) + 1 for word in tokens]
        if not ids:
            ids = [1]
            tokens = ['<empty>']
        return ids, tokens


def load_samples(jsonl_path: Path, limit: int = 3) -> list[str]:
    samples: list[str] = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            text = (rec.get('text') or '').strip()
            if text:
                samples.append(text)
            if len(samples) >= limit:
                break
    if not samples:
        raise RuntimeError(f'No samples found in {jsonl_path}')
    return samples


def topk_stats(logits: torch.Tensor, k: int = 5) -> list[dict]:
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=k, dim=-1)
    out = []
    for rank in range(k):
        out.append({
            'token_id': int(indices[rank].item()),
            'prob': float(values[rank].item()),
        })
    return out


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return float((-(probs * log_probs).sum()).item())


def main() -> None:
    dataset_path = Path('data/processed/wikitext-103/validation.jsonl')
    samples = load_samples(dataset_path, limit=3)

    config = MemoryMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=32000,
        max_context_length=96,
        chunk_size=32,
        window_size=32,
    )
    model = MemoryMAC(config)

    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend is not ready: {model.titans_status.error}')

    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    model.live_backend.eval()

    report = []
    with torch.no_grad():
        for idx, text in enumerate(samples, start=1):
            input_ids, tokens = tokenizer.encode(text, max_length=config.max_context_length)
            x = torch.tensor([input_ids], dtype=torch.long)
            logits, states = model.live_backend(x)
            last_logits = logits[0, -1]
            report.append({
                'sample_index': idx,
                'preview': text[:180],
                'input_tokens': len(input_ids),
                'last_input_token': tokens[-1],
                'logits_shape': tuple(logits.shape),
                'state_count': len(states),
                'entropy_last_step': entropy_from_logits(last_logits),
                'top5_next_token_ids': topk_stats(last_logits, k=5),
            })

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
