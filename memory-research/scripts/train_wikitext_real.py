from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from typing import Iterable

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 4
    grad_accum_steps: int = 1
    seq_len: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    chunk_size: int = 32
    window_size: int = 32
    memory_decay: float = 0.001
    memory_lr: float = 0.1
    memory_momentum: float = 0.9
    memory_ablation: str = 'none'
    stateful_segments: bool = False
    state_reset_tokens: int = 0
    persistent_memory: bool = False
    persistent_eval_memory: bool = False
    stream_cursor_mode: str = 'random'
    seed: int = 42
    tokenizer_name: str = 'gpt2'
    train_limit: int = 20000
    val_limit: int = 2000
    eval_every: int = 20
    eval_batches: int = 8
    save_every: int = 50
    log_every: int = 1
    memory_diagnostics_every: int = 1
    out_dir: str | None = None
    run_name: str | None = None
    append_metrics: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description='Train Memory Workspace on WikiText with real tokenizer and Titans backend.')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument(
        '--grad-accum-steps',
        type=int,
        default=1,
        help=(
            'Accumulate this many passes before each optimizer step. In stateful mode, '
            'each pass walks every independent stream once.'
        ),
    )
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--memory-decay', type=float, default=0.001)
    parser.add_argument('--memory-lr', type=float, default=0.1)
    parser.add_argument('--memory-momentum', type=float, default=0.9)
    parser.add_argument('--memory-ablation', choices=('none', 'no-update'), default='none')
    parser.add_argument(
        '--stateful-segments',
        action='store_true',
        help='Train on contiguous segments and carry MAC memory state between optimizer steps.',
    )
    parser.add_argument(
        '--state-reset-tokens',
        type=int,
        default=0,
        help=(
            'Reset carried state after this many streamed tokens. 0 disables periodic '
            'resets. Stream wrap still resets unless --persistent-memory is set.'
        ),
    )
    parser.add_argument(
        '--persistent-memory',
        action='store_true',
        help=(
            'Carry the fixed-size MAC memory state indefinitely during training. This '
            'disables both periodic state resets and wrap resets; the memory tensor '
            'shape stays constant and the autograd graph is detached between segments.'
        ),
    )
    parser.add_argument(
        '--persistent-eval-memory',
        action='store_true',
        help=(
            'Carry fixed-size MAC memory state across validation batches and eval calls '
            'instead of evaluating every batch with fresh memory.'
        ),
    )
    parser.add_argument(
        '--stream-cursor-mode',
        choices=('random', 'sharded'),
        default='random',
        help=(
            'How stateful streams start. random matches older runs; sharded starts '
            'streams evenly through the token stream for deterministic full-corpus coverage.'
        ),
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tokenizer-name', type=str, default='gpt2')
    parser.add_argument('--train-limit', type=int, default=20000)
    parser.add_argument('--val-limit', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--eval-batches', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument(
        '--log-every',
        type=int,
        default=1,
        help='Write/print a metrics record every N steps, always including eval/save steps.',
    )
    parser.add_argument(
        '--memory-diagnostics-every',
        type=int,
        default=1,
        help='Collect verbose memory update/gradient diagnostics every N logged steps.',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Explicit artifact directory. Defaults to a timestamped run directory.',
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Human-readable run label used when --out-dir is omitted.',
    )
    parser.add_argument(
        '--append-metrics',
        action='store_true',
        help='Append to an existing metrics.jsonl instead of requiring a fresh run file.',
    )
    args = parser.parse_args()
    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip_norm=args.grad_clip_norm,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        memory_decay=args.memory_decay,
        memory_lr=args.memory_lr,
        memory_momentum=args.memory_momentum,
        memory_ablation=args.memory_ablation,
        stateful_segments=args.stateful_segments,
        state_reset_tokens=args.state_reset_tokens,
        persistent_memory=args.persistent_memory,
        persistent_eval_memory=args.persistent_eval_memory,
        stream_cursor_mode=args.stream_cursor_mode,
        seed=args.seed,
        tokenizer_name=args.tokenizer_name,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        memory_diagnostics_every=args.memory_diagnostics_every,
        out_dir=args.out_dir,
        run_name=args.run_name,
        append_metrics=args.append_metrics,
    )


def load_texts(path: Path, limit: int) -> list[str]:
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


def build_token_stream(texts: Iterable[str], tokenizer: AutoTokenizer) -> torch.Tensor:
    token_stream, _boundaries = build_token_stream_with_boundaries(texts, tokenizer)
    return token_stream


def build_token_stream_with_boundaries(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
) -> tuple[torch.Tensor, list[int]]:
    eos_id = tokenizer.eos_token_id
    token_ids: list[int] = []
    boundaries: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        token_ids.append(eos_id)
        boundaries.append(len(token_ids) - 1)
    if len(token_ids) < 2:
        raise RuntimeError('Token stream is too small for language modeling')
    return torch.tensor(token_ids, dtype=torch.long), boundaries


def valid_document_starts(boundaries: list[int], seq_len: int, max_start: int) -> list[int]:
    starts = []
    prev_boundary = -1
    for boundary in boundaries:
        start = prev_boundary + 1
        if start <= max_start and boundary >= start + seq_len:
            starts.append(start)
        prev_boundary = boundary
    if not starts:
        raise RuntimeError(
            'No WikiText document is long enough for seq_len + 1 tokens; '
            'reduce --seq-len or increase the dataset limit.'
        )
    return starts


def next_boundary_at_or_after(boundaries: list[int], cursor: int) -> int | None:
    boundary_idx = bisect_left(boundaries, cursor)
    if boundary_idx >= len(boundaries):
        return None
    return boundaries[boundary_idx]


def align_cursor_to_document_segment(
    token_stream: torch.Tensor,
    boundaries: list[int],
    cursor: int,
    seq_len: int,
) -> tuple[int, bool, bool, int]:
    max_start = token_stream.numel() - (seq_len + 1)
    if max_start < 0:
        raise RuntimeError('Token stream is shorter than seq_len + 1')

    starts = valid_document_starts(boundaries, seq_len, max_start)
    wrapped = False
    boundary_reset = False
    skipped_documents = 0
    attempts = 0
    max_attempts = len(boundaries) + 2

    while True:
        if cursor > max_start:
            cursor = starts[0]
            wrapped = True
            boundary_reset = True

        boundary = next_boundary_at_or_after(boundaries, cursor)
        if boundary is not None and boundary >= cursor + seq_len:
            return cursor, wrapped, boundary_reset, skipped_documents

        boundary_reset = True
        skipped_documents += 1
        if boundary is None:
            cursor = starts[0]
            wrapped = True
        else:
            cursor = boundary + 1

        attempts += 1
        if attempts > max_attempts:
            cursor = starts[0]
            return cursor, True, True, skipped_documents


def sample_batch(token_stream: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_stream.numel() - (seq_len + 1)
    if max_start < 0:
        raise RuntimeError('Token stream is shorter than seq_len + 1')
    starts = torch.randint(0, max_start + 1, (batch_size,))
    inputs = [token_stream[start : start + seq_len] for start in starts.tolist()]
    targets = [token_stream[start + 1 : start + seq_len + 1] for start in starts.tolist()]
    return torch.stack(inputs), torch.stack(targets)


def next_stateful_segment(
    token_stream: torch.Tensor,
    cursor: int,
    seq_len: int,
    boundaries: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, bool, int]:
    max_start = token_stream.numel() - (seq_len + 1)
    if max_start < 0:
        raise RuntimeError('Token stream is shorter than seq_len + 1')

    wrapped = False
    boundary_reset = False
    skipped_documents = 0
    if cursor > max_start:
        cursor = 0
        wrapped = True
    if boundaries is not None:
        cursor, wrapped, boundary_reset, skipped_documents = align_cursor_to_document_segment(
            token_stream,
            boundaries,
            cursor,
            seq_len,
        )

    x = token_stream[cursor : cursor + seq_len].unsqueeze(0)
    y = token_stream[cursor + 1 : cursor + seq_len + 1].unsqueeze(0)
    return x, y, cursor + seq_len, wrapped, boundary_reset, skipped_documents


def next_stateful_batch(
    token_stream: torch.Tensor,
    cursors: list[int],
    seq_len: int,
    boundaries: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[bool], list[bool], list[int]]:
    xs = []
    ys = []
    next_cursors = []
    wrapped_flags = []
    boundary_reset_flags = []
    skipped_documents = []
    for cursor in cursors:
        x, y, next_cursor, wrapped, boundary_reset, skipped = next_stateful_segment(
            token_stream,
            cursor,
            seq_len,
            boundaries,
        )
        xs.append(x.squeeze(0))
        ys.append(y.squeeze(0))
        next_cursors.append(next_cursor)
        wrapped_flags.append(wrapped)
        boundary_reset_flags.append(boundary_reset)
        skipped_documents.append(skipped)
    return torch.stack(xs), torch.stack(ys), next_cursors, wrapped_flags, boundary_reset_flags, skipped_documents


def summarize_values(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {}
    float_values = [float(value) for value in values]
    return {
        'min': min(float_values),
        'max': max(float_values),
        'mean': sum(float_values) / len(float_values),
    }


def initial_stream_cursors(
    token_stream: torch.Tensor,
    batch_size: int,
    seq_len: int,
    mode: str,
    boundaries: list[int] | None = None,
) -> list[int]:
    max_start = max(int(token_stream.numel()) - (seq_len + 1), 0)
    if boundaries is not None:
        starts = valid_document_starts(boundaries, seq_len, max_start)
        if mode == 'sharded':
            if batch_size == 1:
                return [starts[0]]
            return [
                starts[(stream_idx * (len(starts) - 1)) // (batch_size - 1)]
                for stream_idx in range(batch_size)
            ]
        return [random.choice(starts) for _ in range(batch_size)]

    if mode == 'sharded':
        span = max_start + 1
        cursors = []
        for stream_idx in range(batch_size):
            cursor = (stream_idx * span) // batch_size
            cursor = (cursor // seq_len) * seq_len
            cursors.append(min(cursor, max_start))
        return cursors
    return [random.randint(0, max_start) for _ in range(batch_size)]


def build_eval_batches(token_stream: torch.Tensor, batch_size: int, seq_len: int, batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    span = seq_len + 1
    max_start = token_stream.numel() - span
    if max_start < 0:
        raise RuntimeError('Validation token stream is shorter than seq_len + 1')

    total_windows = batch_size * batches
    if total_windows == 1:
        starts = [max_start // 2]
    else:
        starts = [(idx * max_start) // (total_windows - 1) for idx in range(total_windows)]

    eval_batches = []
    for batch_idx in range(batches):
        batch_starts = starts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        inputs = [token_stream[start : start + seq_len] for start in batch_starts]
        targets = [token_stream[start + 1 : start + seq_len + 1] for start in batch_starts]
        eval_batches.append((torch.stack(inputs), torch.stack(targets)))
    return eval_batches


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            total += float((grad * grad).sum().item())
    return total ** 0.5


def collect_memory_update_stats(backend: torch.nn.Module) -> dict:
    layer_stats = []
    for layer_idx, block in enumerate(getattr(backend, 'blocks', [])):
        memory = getattr(block, 'memory', None)
        stats = getattr(memory, 'last_update_stats', None)
        if stats:
            layer_stats.append({'layer': layer_idx, **stats})

    if not layer_stats:
        return {}

    summary_keys = [
        'effective_decay',
        'effective_lr',
        'effective_momentum',
        'memory_grad_norm_sum',
        'weight_delta_norm_sum',
        'momentum_delta_norm_sum',
        'post_weight_norm_sum',
        'post_momentum_norm_sum',
    ]
    summary = {}
    for key in summary_keys:
        values = [float(stats[key]) for stats in layer_stats if key in stats]
        if values:
            summary[f'{key}_mean'] = sum(values) / len(values)
            summary[f'{key}_max'] = max(values)

    return {
        'layers': layer_stats,
        'summary': summary,
    }


def collect_memory_grad_stats(backend: torch.nn.Module) -> dict:
    groups: dict[str, list[torch.nn.Parameter]] = {
        'gate_decay': [],
        'gate_lr': [],
        'gate_momentum': [],
        'proj_k': [],
        'proj_v': [],
        'proj_q': [],
        'proj_out': [],
        'memory_mlp': [],
    }

    for block in getattr(backend, 'blocks', []):
        memory = getattr(block, 'memory', None)
        if memory is None:
            continue
        groups['gate_decay'].extend(memory.gate_decay.parameters())
        groups['gate_lr'].extend(memory.gate_lr.parameters())
        groups['gate_momentum'].extend(memory.gate_momentum.parameters())
        groups['proj_k'].extend(memory.proj_k.parameters())
        groups['proj_v'].extend(memory.proj_v.parameters())
        groups['proj_q'].extend(memory.proj_q.parameters())
        groups['proj_out'].extend(memory.proj_out.parameters())
        groups['memory_mlp'].extend(memory.memory.parameters())

    summary = {}
    for name, params in groups.items():
        with_grad = 0
        nonzero_grad = 0
        grad_norm_sq = 0.0
        for param in params:
            if param.grad is None:
                continue
            with_grad += 1
            grad = param.grad.detach().float()
            grad_norm_sq += float((grad * grad).sum().item())
            if bool((grad != 0).any().item()):
                nonzero_grad += 1
        summary[name] = {
            'params': len(params),
            'with_grad': with_grad,
            'nonzero_grad': nonzero_grad,
            'grad_norm': grad_norm_sq ** 0.5,
        }
    return summary


def detach_states(states: list | None) -> list | None:
    if states is None:
        return None
    return [state.detach() if state is not None else None for state in states]


def reset_batched_state_rows(
    model: torch.nn.Module,
    states: list | None,
    row_indices: list[int],
    batch_size: int,
    device: str,
) -> list | None:
    if states is None or not row_indices:
        return states

    rows = torch.tensor(sorted(set(row_indices)), dtype=torch.long, device=device)
    reset_states = [
        block.memory.init_state(batch_size, torch.device(device))
        for block in model.blocks
    ]
    next_states = detach_states(states)

    def replace_rows(value: torch.Tensor, reset_value: torch.Tensor) -> torch.Tensor:
        mask_shape = (batch_size,) + (1,) * (value.ndim - 1)
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=value.device)
        mask.index_fill_(0, rows.to(value.device), True)
        return torch.where(mask, reset_value.to(value.device, value.dtype), value)

    for state, reset_state in zip(next_states, reset_states, strict=True):
        for idx, (weight, reset_weight) in enumerate(zip(state.weights, reset_state.weights, strict=True)):
            state.weights[idx] = replace_rows(weight, reset_weight)
        for idx, (momentum, reset_momentum) in enumerate(zip(state.momentum, reset_state.momentum, strict=True)):
            state.momentum[idx] = replace_rows(momentum, reset_momentum)
    return next_states


def serialize_states(states: list | None) -> list[dict[str, list[torch.Tensor]]] | None:
    if states is None:
        return None

    serialized = []
    for state in states:
        if state is None:
            serialized.append({'weights': [], 'momentum': []})
            continue
        serialized.append({
            'weights': [weight.detach().cpu() for weight in state.weights],
            'momentum': [momentum.detach().cpu() for momentum in state.momentum],
        })
    return serialized


def state_norm_summary(states: list | None) -> dict:
    if states is None:
        return {'layers': 0, 'weight_norm_sum': 0.0, 'momentum_norm_sum': 0.0}

    weight_norm = 0.0
    momentum_norm = 0.0
    layers = 0
    for state in states:
        if state is None:
            continue
        layers += 1
        weight_norm += sum(float(weight.detach().float().norm().item()) for weight in state.weights)
        momentum_norm += sum(float(momentum.detach().float().norm().item()) for momentum in state.momentum)
    return {
        'layers': layers,
        'weight_norm_sum': weight_norm,
        'momentum_norm_sum': momentum_norm,
    }


def runtime_memory_payload(
    cfg: TrainConfig,
    train_batched_states: list | None,
    train_states_by_stream: list[list | None],
    eval_states_by_stream: list | None,
    train_cursors: list[int],
    eval_cursors: list[int],
    tokens_since_state_reset_by_stream: list[int],
) -> dict:
    if cfg.stateful_segments and cfg.persistent_memory:
        train_states = train_batched_states
        train_mode = 'batched'
    elif cfg.stateful_segments:
        train_states = train_states_by_stream
        train_mode = 'per_stream'
    else:
        train_states = None
        train_mode = 'none'

    return {
        'format': 'memory_mac_runtime_memory_v1',
        'fixed_size': True,
        'train_mode': train_mode,
        'train_states': serialize_states(train_states),
        'eval_states': serialize_states(eval_states_by_stream),
        'train_cursors': [int(value) for value in train_cursors],
        'eval_cursors': [int(value) for value in eval_cursors],
        'tokens_since_state_reset': [int(value) for value in tokens_since_state_reset_by_stream],
        'state_reset_tokens': int(cfg.state_reset_tokens),
        'persistent_memory': bool(cfg.persistent_memory),
        'persistent_eval_memory': bool(cfg.persistent_eval_memory),
        'train_state_norms': state_norm_summary(train_states),
        'eval_state_norms': state_norm_summary(eval_states_by_stream),
    }


def evaluate(model: torch.nn.Module, eval_batches: list[tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x_cpu, y_cpu in eval_batches:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def evaluate_stateful(
    model: torch.nn.Module,
    token_stream: torch.Tensor,
    boundaries: list[int],
    batch_size: int,
    seq_len: int,
    batches: int,
    device: str,
    states_by_stream: list | None,
    cursors: list[int],
    persistent_memory: bool,
) -> tuple[float, list | None, list[int], int, int, int]:
    model.eval()
    losses = []
    wrap_count = 0
    reset_count = 0
    skipped_document_count = 0
    with torch.no_grad():
        if persistent_memory:
            batched_states = states_by_stream
            for _batch_idx in range(batches):
                x_cpu, y_cpu, cursors, wrapped_flags, boundary_reset_flags, skipped_documents = next_stateful_batch(
                    token_stream,
                    cursors,
                    seq_len,
                    boundaries,
                )
                reset_rows = [
                    idx
                    for idx, (wrapped, boundary_reset) in enumerate(zip(wrapped_flags, boundary_reset_flags, strict=True))
                    if wrapped or boundary_reset
                ]
                if reset_rows:
                    batched_states = reset_batched_state_rows(
                        model,
                        batched_states,
                        reset_rows,
                        batch_size,
                        device,
                    )
                reset_count += len(reset_rows)
                skipped_document_count += sum(skipped_documents)
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                logits, new_states = model(x, states=batched_states)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(float(loss.item()))
                batched_states = detach_states(new_states)
                wrap_count += sum(1 for wrapped in wrapped_flags if wrapped)
            states_by_stream = batched_states
        else:
            if states_by_stream is None:
                states_by_stream = [None] * batch_size
            for _batch_idx in range(batches):
                for stream_idx in range(batch_size):
                    x_cpu, y_cpu, cursors[stream_idx], wrapped, boundary_reset, skipped_documents = next_stateful_segment(
                        token_stream,
                        cursors[stream_idx],
                        seq_len,
                        boundaries,
                    )
                    skipped_document_count += skipped_documents
                    if wrapped or boundary_reset:
                        wrap_count += 1
                        reset_count += 1
                        states_by_stream[stream_idx] = None

                    x = x_cpu.to(device)
                    y = y_cpu.to(device)
                    logits, new_states = model(x, states=states_by_stream[stream_idx])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    losses.append(float(loss.item()))
                    states_by_stream[stream_idx] = detach_states(new_states)
    model.train()
    return sum(losses) / len(losses), states_by_stream, cursors, wrap_count, reset_count, skipped_document_count


def disable_memory_updates(backend: torch.nn.Module) -> None:
    """Keep memory retrieval active but prevent test-time memory weight updates."""
    for block in backend.blocks:
        memory = block.memory

        def no_update_forward(x, state=None, return_state=True, _memory=memory):
            batch_size, _seq_len, _dim = x.shape
            if state is None:
                state = _memory.init_state(batch_size, x.device)
            retrieved = _memory.retrieve(x, state)
            if return_state:
                return retrieved, state.detach()
            return retrieved, None

        memory.forward = no_update_forward


def slugify(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9._-]+', '-', value.strip()).strip('-')
    return slug or 'run'


def resolve_output_dir(root: Path, cfg: TrainConfig) -> Path:
    if cfg.out_dir:
        return Path(cfg.out_dir).expanduser()

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    prefix = slugify(cfg.run_name) if cfg.run_name else 'wikitext-real'
    return root / 'artifacts' / 'runs' / f'{prefix}-{timestamp}'


def prepare_run_files(out_dir: Path, cfg: TrainConfig) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / 'metrics.jsonl'
    config_path = out_dir / 'config.json'
    if metrics_path.exists() and not cfg.append_metrics:
        raise FileExistsError(
            f'{metrics_path} already exists. Choose a new --run-name/--out-dir '
            'or pass --append-metrics intentionally.'
        )
    return metrics_path, config_path


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> LambdaLR:
    """Linear warmup followed by cosine decay to min_lr_ratio of base LR."""
    warmup_steps = max(0, int(cfg.steps * cfg.warmup_ratio))
    min_ratio = float(cfg.min_lr_ratio)

    def lr_lambda(step_idx: int) -> float:
        # LambdaLR calls this with 0 before the first optimizer step.
        step = step_idx + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return max(step / warmup_steps, 1e-8)
        decay_steps = max(cfg.steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    cfg = parse_args()
    if cfg.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if cfg.grad_accum_steps < 1:
        raise ValueError('--grad-accum-steps must be >= 1')
    if cfg.state_reset_tokens < 0:
        raise ValueError('--state-reset-tokens must be >= 0')
    if cfg.persistent_memory and not cfg.stateful_segments:
        raise ValueError('--persistent-memory requires --stateful-segments')
    if cfg.persistent_memory and cfg.state_reset_tokens != 0:
        raise ValueError('--persistent-memory requires --state-reset-tokens 0')
    if cfg.log_every < 1:
        raise ValueError('--log-every must be >= 1')
    if cfg.memory_diagnostics_every < 1:
        raise ValueError('--memory-diagnostics-every must be >= 1')

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    root = Path(__file__).resolve().parents[1]
    out_dir = resolve_output_dir(root, cfg)

    train_path = Path('data/processed/wikitext-103/train.jsonl')
    val_path = Path('data/processed/wikitext-103/validation.jsonl')

    train_texts = load_texts(train_path, cfg.train_limit)
    val_texts = load_texts(val_path, cfg.val_limit)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f'Tokenizer {cfg.tokenizer_name} must define eos_token_id')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tokens, train_boundaries = build_token_stream_with_boundaries(train_texts, tokenizer)
    val_tokens, val_boundaries = build_token_stream_with_boundaries(val_texts, tokenizer)
    eval_batches = build_eval_batches(val_tokens, cfg.batch_size, cfg.seq_len, cfg.eval_batches)

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
            memory_decay=cfg.memory_decay,
            memory_lr=cfg.memory_lr,
            memory_momentum=cfg.memory_momentum,
        )
    )
    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend is not ready: {model.titans_status.error}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = model.live_backend.to(device)
    if cfg.memory_ablation == 'no-update':
        disable_memory_updates(backend)
    backend.train()

    optimizer = torch.optim.AdamW(
        backend.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    metrics_path, config_path = prepare_run_files(out_dir, cfg)
    config_path.write_text(json.dumps({
        **asdict(cfg),
        'train_tokens': int(train_tokens.numel()),
        'val_tokens': int(val_tokens.numel()),
        'train_documents': len(train_boundaries),
        'val_documents': len(val_boundaries),
        'document_boundary_resets': bool(cfg.stateful_segments),
        'document_boundary_policy': 'stateful segments never cross EOS; affected stream rows reset to learned initial memory state',
        'effective_tokens_per_step': cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len,
        'stateful_independent_streams': cfg.batch_size if cfg.stateful_segments else 0,
        'memory_state_shape': 'fixed-size per layer/stream; no KV/cache growth',
        'autograd_through_memory_state': 'truncated by detach between streamed segments',
        'stateful_forward_mode': 'batched' if cfg.stateful_segments and cfg.persistent_memory else 'serial',
    }, indent=2), encoding='utf-8')

    print({'status': 'starting', 'artifacts': str(out_dir)}, flush=True)

    best_val = float('inf')
    train_states_by_stream: list[list | None] = [None] * cfg.batch_size
    train_batched_states: list | None = None
    train_cursors = initial_stream_cursors(
        train_tokens,
        cfg.batch_size,
        cfg.seq_len,
        cfg.stream_cursor_mode,
        train_boundaries if cfg.stateful_segments else None,
    )
    eval_states_by_stream: list | None = None if cfg.persistent_memory else [None] * cfg.batch_size
    eval_cursors = initial_stream_cursors(
        val_tokens,
        cfg.batch_size,
        cfg.seq_len,
        'sharded' if cfg.persistent_eval_memory else cfg.stream_cursor_mode,
        val_boundaries if cfg.persistent_eval_memory else None,
    )
    if cfg.state_reset_tokens > 0:
        tokens_since_state_reset_by_stream = [
            random.randrange(0, cfg.state_reset_tokens)
            for _ in range(cfg.batch_size)
        ]
    else:
        tokens_since_state_reset_by_stream = [0] * cfg.batch_size
    tokens_seen = 0
    for step in range(1, cfg.steps + 1):
        state_was_reset = False
        stream_reset_count = 0
        stream_wrap_count = 0
        stream_skipped_document_count = 0
        micro_losses: list[float] = []
        if cfg.stateful_segments and cfg.persistent_memory:
            micro_batches = cfg.grad_accum_steps
            stateful_forward_mode = 'batched'
        elif cfg.stateful_segments:
            micro_batches = cfg.batch_size * cfg.grad_accum_steps
            stateful_forward_mode = 'serial'
        else:
            micro_batches = cfg.grad_accum_steps
            stateful_forward_mode = 'random'

        optimizer.zero_grad(set_to_none=True)
        if cfg.stateful_segments and cfg.persistent_memory:
            for _accum_idx in range(cfg.grad_accum_steps):
                x_cpu, y_cpu, train_cursors, wrapped_flags, boundary_reset_flags, skipped_documents = next_stateful_batch(
                    train_tokens,
                    train_cursors,
                    cfg.seq_len,
                    train_boundaries,
                )
                stream_wrap_count += sum(1 for wrapped in wrapped_flags if wrapped)
                stream_skipped_document_count += sum(skipped_documents)
                reset_rows = [
                    idx
                    for idx, (wrapped, boundary_reset) in enumerate(zip(wrapped_flags, boundary_reset_flags, strict=True))
                    if wrapped or boundary_reset
                ]
                if reset_rows:
                    train_batched_states = reset_batched_state_rows(
                        backend,
                        train_batched_states,
                        reset_rows,
                        cfg.batch_size,
                        device,
                    )
                    for stream_idx in reset_rows:
                        tokens_since_state_reset_by_stream[stream_idx] = 0
                    state_was_reset = True
                    stream_reset_count += len(reset_rows)

                x = x_cpu.to(device)
                y = y_cpu.to(device)
                logits, new_states = backend(x, states=train_batched_states)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                (loss / cfg.grad_accum_steps).backward()
                micro_losses.append(float(loss.item()))
                train_batched_states = detach_states(new_states)
                tokens_since_state_reset_by_stream = [
                    value + cfg.seq_len for value in tokens_since_state_reset_by_stream
                ]
                tokens_seen += cfg.batch_size * cfg.seq_len
        elif cfg.stateful_segments:
            for _accum_idx in range(cfg.grad_accum_steps):
                for stream_idx in range(cfg.batch_size):
                    should_reset = (
                        not cfg.persistent_memory
                        and
                        cfg.state_reset_tokens > 0
                        and tokens_since_state_reset_by_stream[stream_idx] >= cfg.state_reset_tokens
                    )
                    if should_reset:
                        train_states_by_stream[stream_idx] = None
                        tokens_since_state_reset_by_stream[stream_idx] = 0
                        state_was_reset = True
                        stream_reset_count += 1

                    x_cpu, y_cpu, train_cursors[stream_idx], wrapped, boundary_reset, skipped_documents = next_stateful_segment(
                        train_tokens,
                        train_cursors[stream_idx],
                        cfg.seq_len,
                        train_boundaries,
                    )
                    stream_skipped_document_count += skipped_documents
                    if wrapped:
                        stream_wrap_count += 1
                    if wrapped or boundary_reset:
                        train_states_by_stream[stream_idx] = None
                        tokens_since_state_reset_by_stream[stream_idx] = 0
                        state_was_reset = True
                        stream_reset_count += 1

                    x = x_cpu.to(device)
                    y = y_cpu.to(device)
                    logits, new_states = backend(x, states=train_states_by_stream[stream_idx])
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                    (loss / micro_batches).backward()
                    micro_losses.append(float(loss.item()))
                    train_states_by_stream[stream_idx] = detach_states(new_states)
                    tokens_since_state_reset_by_stream[stream_idx] += cfg.seq_len
                    tokens_seen += cfg.seq_len
        else:
            for _accum_idx in range(cfg.grad_accum_steps):
                x_cpu, y_cpu = sample_batch(train_tokens, cfg.batch_size, cfg.seq_len)
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                logits, _ = backend(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                (loss / cfg.grad_accum_steps).backward()
                micro_losses.append(float(loss.item()))
                tokens_seen += cfg.batch_size * cfg.seq_len

        should_eval = step % cfg.eval_every == 0 or step == 1
        should_save = step % cfg.save_every == 0
        should_log = step % cfg.log_every == 0 or should_eval or should_save
        should_collect_memory_diagnostics = (
            should_log
            and (step == 1 or step % cfg.memory_diagnostics_every == 0 or should_eval or should_save)
        )

        memory_grad_stats = (
            collect_memory_grad_stats(backend)
            if should_collect_memory_diagnostics
            else {}
        )
        unclipped_grad_norm = grad_norm(backend)
        clipped_grad_norm = float(torch.nn.utils.clip_grad_norm_(backend.parameters(), cfg.grad_clip_norm).item())
        optimizer.step()
        scheduler.step()

        train_loss = sum(micro_losses) / len(micro_losses)
        loss_variance = sum((value - train_loss) ** 2 for value in micro_losses) / len(micro_losses)
        record = {
            'step': step,
            'train_loss': train_loss,
            'micro_loss_min': min(micro_losses),
            'micro_loss_max': max(micro_losses),
            'micro_loss_std': loss_variance ** 0.5,
            'train_ppl_est': math.exp(min(train_loss, 20)),
            'grad_norm': unclipped_grad_norm,
            'grad_norm_after_clip': min(clipped_grad_norm, cfg.grad_clip_norm),
            'grad_was_clipped': unclipped_grad_norm > cfg.grad_clip_norm,
            'tokens_seen': tokens_seen,
            'effective_tokens_per_step': cfg.batch_size * cfg.grad_accum_steps * cfg.seq_len,
            'micro_batches': micro_batches,
            'stateful_forward_mode': stateful_forward_mode,
            'device': device,
            'python': sys.executable,
            'stateful_segments': cfg.stateful_segments,
            'stateful_independent_streams': cfg.batch_size if cfg.stateful_segments else 0,
            'persistent_memory': cfg.persistent_memory,
            'persistent_eval_memory': cfg.persistent_eval_memory,
            'stream_cursor_mode': cfg.stream_cursor_mode,
            'document_boundary_resets': cfg.stateful_segments,
            'state_was_reset': state_was_reset,
            'stream_reset_count': stream_reset_count,
            'stream_wrap_count': stream_wrap_count,
            'stream_skipped_document_count': stream_skipped_document_count,
            'tokens_since_state_reset': (
                summarize_values(tokens_since_state_reset_by_stream)
                if cfg.stateful_segments
                else {}
            ),
            'memory_update': (
                collect_memory_update_stats(backend)
                if should_collect_memory_diagnostics
                else {}
            ),
            'memory_grads': memory_grad_stats,
        }

        if should_eval:
            eval_wrap_count = 0
            eval_reset_count = 0
            eval_skipped_document_count = 0
            if cfg.persistent_eval_memory:
                (
                    val_loss,
                    eval_states_by_stream,
                    eval_cursors,
                    eval_wrap_count,
                    eval_reset_count,
                    eval_skipped_document_count,
                ) = evaluate_stateful(
                    backend,
                    val_tokens,
                    val_boundaries,
                    cfg.batch_size,
                    cfg.seq_len,
                    cfg.eval_batches,
                    device,
                    eval_states_by_stream,
                    eval_cursors,
                    cfg.persistent_memory,
                )
            else:
                val_loss = evaluate(backend, eval_batches, device)
            record['val_loss'] = val_loss
            record['val_ppl_est'] = math.exp(min(val_loss, 20))
            record['eval_wrap_count'] = eval_wrap_count
            record['eval_reset_count'] = eval_reset_count
            record['eval_skipped_document_count'] = eval_skipped_document_count
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'model_state': backend.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'step': step,
                    'best_val': best_val,
                    'config': asdict(cfg),
                    'runtime_memory_state': runtime_memory_payload(
                        cfg,
                        train_batched_states,
                        train_states_by_stream,
                        eval_states_by_stream,
                        train_cursors,
                        eval_cursors,
                        tokens_since_state_reset_by_stream,
                    ),
                    'rng_state': torch.random.get_rng_state(),
                    'python': sys.executable,
                }, out_dir / 'best.pt')

        if should_save:
            torch.save({
                'model_state': backend.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'step': step,
                'best_val': best_val,
                'config': asdict(cfg),
                'runtime_memory_state': runtime_memory_payload(
                    cfg,
                    train_batched_states,
                    train_states_by_stream,
                    eval_states_by_stream,
                    train_cursors,
                    eval_cursors,
                    tokens_since_state_reset_by_stream,
                ),
                'rng_state': torch.random.get_rng_state(),
                'python': sys.executable,
            }, out_dir / f'checkpoint-step-{step}.pt')

        if should_log:
            with metrics_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')

            print(record, flush=True)

    print({'status': 'done', 'best_val': best_val, 'artifacts': str(out_dir)}, flush=True)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
