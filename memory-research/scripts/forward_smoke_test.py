from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_length: int) -> list[int]:
        words = text.split()
        ids = [abs(hash(word)) % (self.vocab_size - 1) + 1 for word in words[:max_length]]
        if not ids:
            ids = [1]
        return ids


def load_first_text(jsonl_path: Path) -> str:
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            text = (rec.get('text') or '').strip()
            if text:
                return text
    raise RuntimeError(f'No text found in {jsonl_path}')


def main() -> None:
    dataset_path = Path('data/processed/wikitext-103/train.jsonl')
    text = load_first_text(dataset_path)

    config = MemoryMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=32000,
        max_context_length=128,
        chunk_size=32,
        window_size=32,
    )
    model = MemoryMAC(config)

    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend is not ready: {model.titans_status.error}')

    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    input_ids = tokenizer.encode(text, max_length=config.max_context_length)
    x = torch.tensor([input_ids], dtype=torch.long)

    model.live_backend.eval()
    with torch.no_grad():
        logits, states = model.live_backend(x)

    print({
        'input_tokens': len(input_ids),
        'input_shape': tuple(x.shape),
        'logits_shape': tuple(logits.shape),
        'num_states': len(states),
        'backend': type(model.live_backend).__name__,
        'sample_text_preview': text[:160],
    })


if __name__ == '__main__':
    main()
