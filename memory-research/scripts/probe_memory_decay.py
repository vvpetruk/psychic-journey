from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    td = t.detach().float()
    return float(torch.sqrt(torch.sum(td * td)).item())


def tensor_rms(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    td = t.detach().float()
    return float(torch.sqrt(torch.mean(td * td)).item())


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


def collect_group_grad_report(model: torch.nn.Module) -> dict[str, dict[str, float | int]]:
    by_group: dict[str, dict[str, float | int]] = defaultdict(lambda: {'grad_norm_sum': 0.0, 'requires_grad_true': 0})
    for name, param in model.named_parameters():
        group = classify_name(name)
        by_group[group]['grad_norm_sum'] = float(by_group[group]['grad_norm_sum']) + tensor_norm(param.grad)
        if param.requires_grad:
            by_group[group]['requires_grad_true'] = int(by_group[group]['requires_grad_true']) + 1
    return dict(by_group)


def install_decay_probe(backend: torch.nn.Module, decay_mode: str) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    for layer_idx, block in enumerate(backend.blocks):
        mem = block.memory
        original_forward = mem.forward

        def wrapped_forward(x, state=None, return_state=True, _mem=mem, _layer_idx=layer_idx, _orig=original_forward, _mode=decay_mode):
            batch_size, _seq_len, _ = x.shape
            device = x.device
            if state is None:
                state = _mem.init_state(batch_size, device)

            pre_weight_norm = sum(tensor_norm(w) for w in state.weights)
            pre_momentum_norm = sum(tensor_norm(m) for m in state.momentum)

            k = _mem.proj_k(x)
            v = _mem.proj_v(x)
            q = _mem.proj_q(x)
            k, v, q = _mem._apply_conv(k, v, q)
            k = F.silu(k)
            v = F.silu(v)
            q = F.silu(q)
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            retrieved = _mem.memory.forward_with_weights(q, state.weights)

            x_mean = x.mean(dim=1, keepdim=True)
            raw_alpha = _mem.gate_decay(x_mean).mean()
            theta = _mem.gate_lr(x_mean).mean() * _mem.config.memory_lr
            eta = _mem.gate_momentum(x_mean).mean() * _mem.config.memory_momentum
            grads = _mem._compute_gradients(k, v, state.weights)

            if _mode == 'force_zero':
                alpha = raw_alpha * 0.0
            elif _mode == 'clamp_small':
                alpha = torch.clamp(raw_alpha, min=0.0, max=0.01)
            elif _mode == 'scale_1pct':
                alpha = raw_alpha * 0.01
            else:
                alpha = raw_alpha

            if _mem.memory.config.use_conv:
                pass

            if _mem.memory.config is None:
                pass

            if _mem.config is None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.memory is None:
                pass

            if _mem.memory is not None:
                pass

            if _mem.config is not None:
                pass

            new_weights, new_momentum = _mem._standard_memory_update(
                state.weights, state.momentum, grads, alpha, eta, theta
            )
            output = _mem.proj_out(retrieved)
            new_state = type(state)(weights=new_weights, momentum=new_momentum)
            final_state = new_state.detach() if return_state else None

            post_weight_norm = sum(tensor_norm(w) for w in new_state.weights)
            post_momentum_norm = sum(tensor_norm(m) for m in new_state.momentum)

            traces.append({
                'layer': _layer_idx,
                'raw_alpha': float(raw_alpha.detach().item()),
                'effective_alpha': float(alpha.detach().item()),
                'theta': float(theta.detach().item()),
                'eta': float(eta.detach().item()),
                'retrieved_rms': tensor_rms(retrieved),
                'input_rms': tensor_rms(x),
                'memory_grad_norm_sum': sum(tensor_norm(g) for g in grads),
                'pre_weight_norm_sum': pre_weight_norm,
                'post_weight_norm_sum': post_weight_norm,
                'pre_momentum_norm_sum': pre_momentum_norm,
                'post_momentum_norm_sum': post_momentum_norm,
            })
            return output, final_state

        mem.forward = wrapped_forward

    return traces


def run_case(case: dict[str, Any], token_stream: torch.Tensor, vocab_size: int) -> dict[str, Any]:
    cfg = MemoryMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=vocab_size,
        max_context_length=128,
        chunk_size=32,
        window_size=32,
    )
    model = MemoryMAC(cfg)
    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend unavailable: {model.titans_status.error}')
    backend = model.live_backend
    backend.config.memory_lr = case['memory_lr']
    backend.config.memory_momentum = case['memory_momentum']
    backend.config.memory_decay = case['memory_decay']
    for block in backend.blocks:
        block.memory.config.memory_lr = case['memory_lr']
        block.memory.config.memory_momentum = case['memory_momentum']
        block.memory.config.memory_decay = case['memory_decay']

    traces = install_decay_probe(backend, case['decay_mode'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = backend.to(device)
    backend.train()
    optimizer = torch.optim.AdamW(backend.parameters(), lr=3e-4)

    steps = case.get('steps', 8)
    results = []
    for step in range(1, steps + 1):
        trace_start = len(traces)
        x, y = sample_batch(token_stream, batch_size=4, seq_len=128)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = backend(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        group_report = collect_group_grad_report(backend)
        optimizer.step()
        results.append({
            'step': step,
            'loss': float(loss.item()),
            'ppl_estimate': float(math.exp(min(20.0, float(loss.item())))),
            'grad_groups': group_report,
            'memory_trace': traces[trace_start:],
        })

    return {
        'name': case['name'],
        'decay_mode': case['decay_mode'],
        'memory_lr': case['memory_lr'],
        'memory_momentum': case['memory_momentum'],
        'memory_decay': case['memory_decay'],
        'steps': steps,
        'results': results,
    }


def main() -> None:
    train_path = Path('data/processed/wikitext-103/train.jsonl')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = load_texts(train_path, limit=2000)
    token_stream = build_token_stream(texts, tokenizer)

    cases = [
        {'name': 'baseline_decay', 'decay_mode': 'baseline', 'memory_lr': 0.1, 'memory_momentum': 0.9, 'memory_decay': 0.01, 'steps': 8},
        {'name': 'alpha_zero', 'decay_mode': 'force_zero', 'memory_lr': 0.1, 'memory_momentum': 0.9, 'memory_decay': 0.01, 'steps': 8},
        {'name': 'alpha_clamp_small', 'decay_mode': 'clamp_small', 'memory_lr': 0.1, 'memory_momentum': 0.9, 'memory_decay': 0.01, 'steps': 8},
        {'name': 'alpha_scale_1pct', 'decay_mode': 'scale_1pct', 'memory_lr': 0.1, 'memory_momentum': 0.9, 'memory_decay': 0.01, 'steps': 8},
    ]

    report = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'python': sys.executable,
        'cases': [run_case(case, token_stream, int(tokenizer.vocab_size)) for case in cases],
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
