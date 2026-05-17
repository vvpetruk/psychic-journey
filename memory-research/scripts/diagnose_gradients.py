from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


def load_texts(path: Path, limit: int) -> list[str]:
    texts = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            text = (rec.get('text') or '').strip()
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
    return texts


def build_token_stream(texts: list[str], tokenizer: AutoTokenizer) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    token_ids = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        token_ids.append(eos_id)
    return torch.tensor(token_ids, dtype=torch.long)


def sample_batch(token_stream: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_stream.numel() - (seq_len + 1)
    starts = torch.randint(0, max_start + 1, (batch_size,))
    inputs = [token_stream[s : s + seq_len] for s in starts.tolist()]
    targets = [token_stream[s + 1 : s + seq_len + 1] for s in starts.tolist()]
    return torch.stack(inputs), torch.stack(targets)


def tensor_norm(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    return float(torch.sqrt(torch.sum(t.detach() * t.detach())).item())


def classify_name(name: str) -> str:
    if name.startswith('embed'):
        return 'embedding'
    if '.attention.' in name:
        return 'attention'
    if '.ffn.' in name:
        return 'ffn'
    if '.memory.memory.' in name:
        return 'memory_mlp'
    if '.memory.' in name:
        return 'memory_other'
    if '.persistent.' in name:
        return 'persistent'
    if name.startswith('norm') or '.norm' in name:
        return 'norm'
    if name.startswith('head'):
        return 'head'
    return 'other'


def collect_grad_report(model: torch.nn.Module) -> tuple[dict, list[dict]]:
    by_group = defaultdict(lambda: {'params': 0, 'grad_norm_sum': 0.0, 'param_norm_sum': 0.0, 'requires_grad_true': 0})
    examples = []
    for name, param in model.named_parameters():
        group = classify_name(name)
        gnorm = tensor_norm(param.grad)
        pnorm = tensor_norm(param)
        by_group[group]['params'] += param.numel()
        by_group[group]['grad_norm_sum'] += gnorm
        by_group[group]['param_norm_sum'] += pnorm
        if param.requires_grad:
            by_group[group]['requires_grad_true'] += 1
        if any(key in name for key in ['embed.weight', 'head.weight', 'blocks.0.attention', 'blocks.0.memory.memory', 'blocks.0.ffn']):
            examples.append({
                'name': name,
                'requires_grad': bool(param.requires_grad),
                'grad_norm': gnorm,
                'param_norm': pnorm,
            })
    return dict(by_group), examples[:40]


def main() -> None:
    train_path = Path('data/processed/wikitext-103/train.jsonl')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_texts(train_path, limit=2000)
    token_stream = build_token_stream(texts, tokenizer)

    model = MemoryMAC(MemoryMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=int(tokenizer.vocab_size),
        max_context_length=128,
        chunk_size=32,
        window_size=32,
    ))
    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend unavailable: {model.titans_status.error}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = model.live_backend.to(device)
    backend.train()
    optimizer = torch.optim.AdamW(backend.parameters(), lr=3e-4)

    reports = []
    for step in [1, 2]:
        x, y = sample_batch(token_stream, batch_size=4, seq_len=128)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = backend(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        groups, examples = collect_grad_report(backend)
        reports.append({
            'step': step,
            'loss': float(loss.item()),
            'group_report': groups,
            'example_params': examples,
        })
        optimizer.step()

    print(json.dumps({
        'device': device,
        'python': sys.executable,
        'reports': reports,
    }, indent=2))


if __name__ == '__main__':
    main()
