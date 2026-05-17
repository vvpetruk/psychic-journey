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


def collect_group_grad_report(model: torch.nn.Module) -> dict:
    by_group = defaultdict(lambda: {'grad_norm_sum': 0.0, 'requires_grad_true': 0})
    for name, param in model.named_parameters():
        group = classify_name(name)
        by_group[group]['grad_norm_sum'] += tensor_norm(param.grad)
        if param.requires_grad:
            by_group[group]['requires_grad_true'] += 1
    return dict(by_group)


def disable_memory_updates(backend: torch.nn.Module) -> None:
    for block in backend.blocks:
        original_forward = block.memory.forward
        original_compute_gradients = block.memory._compute_gradients

        def no_update_forward(x, state=None, return_state=True, _mem=block.memory):
            batch_size, _seq_len, _ = x.shape
            device = x.device
            if state is None:
                state = _mem.init_state(batch_size, device)
            retrieved = _mem.retrieve(x, state)
            if return_state:
                return retrieved, state.detach()
            return retrieved, None

        block.memory.forward = no_update_forward
        block.memory._compute_gradients = lambda *args, **kwargs: original_compute_gradients(*args, **kwargs)
        block.memory._original_forward = original_forward


def run_case(ablate_memory: bool) -> dict:
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
    backend = model.live_backend
    if ablate_memory:
        disable_memory_updates(backend)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = backend.to(device)
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
        reports.append({
            'step': step,
            'loss': float(loss.item()),
            'groups': collect_group_grad_report(backend),
        })
        optimizer.step()

    return {
        'ablate_memory': ablate_memory,
        'reports': reports,
    }


def main() -> None:
    print(json.dumps({
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'python': sys.executable,
        'cases': [run_case(False), run_case(True)],
    }, indent=2))


if __name__ == '__main__':
    main()
