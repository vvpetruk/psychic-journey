from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_article_memory import apply_memory_controls, clone_states  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint  # noqa: E402


MODEL_CONFIG_KEYS = (
    "dim",
    "num_heads",
    "num_layers",
    "seq_len",
    "chunk_size",
    "window_size",
    "memory_decay",
    "memory_lr",
    "memory_momentum",
    "tokenizer_name",
)


def merge_model_config(checkpoint: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = dict(checkpoint.get("config") or {})
    has_model_config = any(key in cfg for key in MODEL_CONFIG_KEYS)
    base_path = checkpoint.get("base_checkpoint") or cfg.get("checkpoint")
    if not has_model_config and base_path:
        base = load_checkpoint(Path(base_path), device)
        merged = dict(base.get("config") or {})
        merged.update(cfg)
        checkpoint["config"] = merged
        return merged
    return cfg


@dataclass
class TrainConfig:
    checkpoint: Path
    dataset_name: str = "GAIR/lima"
    dataset_path: Path | None = None
    train_split: str = "train"
    eval_split: str | None = None
    out_dir: Path | None = None
    run_name: str = "codex-lima-memory-task"
    steps: int = 500
    eval_every: int = 50
    save_every: int = 250
    max_train_examples: int = 1000
    max_eval_examples: int = 64
    batch_size: int = 1
    item_boundary_batch: bool = False
    max_prefill_tokens: int = 192
    max_answer_tokens: int = 128
    prefill_batch_tokens: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 123
    device: str = "auto"
    query_prompt: str = "Assistant:"
    hf_token: str | None = None
    generate_samples: int = 3
    max_new_tokens: int = 80
    persistent_memory: bool = False
    load_runtime_memory_state: bool = False
    runtime_memory_batch_index: int = 0
    write_target_to_memory: bool = False
    instruction_persistent_init: str = "none"
    answer_start_tokens: int = 16
    answer_start_weight: float = 1.0
    repetition_unlikelihood_weight: float = 0.0
    repetition_unlikelihood_window: int = 32


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune MAC on LIMA-style prompt/answer episodes with a memory "
            "task structure: prefill prompt, freeze memory updates, train answer loss."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="GAIR/lima")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="codex-lima-memory-task")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-eval-examples", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of independent LIMA item streams to train in parallel.",
    )
    parser.add_argument(
        "--item-boundary-batch",
        action="store_true",
        help=(
            "Batch LIMA episodes as independent item/document streams. Each row "
            "is reset to learned initial memory at the item boundary, then "
            "prefilled and trained on answer tokens without crossing examples."
        ),
    )
    parser.add_argument("--max-prefill-tokens", type=int, default=192)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--prefill-batch-tokens", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--query-prompt", type=str, default="Assistant:")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--generate-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--persistent-memory",
        action="store_true",
        help="Carry one fixed-size MAC memory state through all LIMA training and eval examples.",
    )
    parser.add_argument(
        "--load-runtime-memory-state",
        action="store_true",
        help="Initialize the LIMA memory bank from runtime_memory_state in the base checkpoint when present.",
    )
    parser.add_argument(
        "--runtime-memory-batch-index",
        type=int,
        default=0,
        help="When loading a batched runtime memory state, select this stream index for batch-1 LIMA training.",
    )
    parser.add_argument(
        "--write-target-to-memory",
        action="store_true",
        help="After each optimizer step, write the gold assistant response into the persistent memory bank.",
    )
    parser.add_argument(
        "--instruction-persistent-init",
        choices=("none", "copy", "fresh"),
        default="none",
        help=(
            "Use a LIMA-specific persistent-token bank. 'copy' clones the pretrained "
            "tokens into new trainable parameters; 'fresh' reinitializes them."
        ),
    )
    parser.add_argument(
        "--answer-start-tokens",
        type=int,
        default=16,
        help="Number of initial answer tokens to upweight in the teacher-forced loss.",
    )
    parser.add_argument(
        "--answer-start-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for the first --answer-start-tokens answer tokens.",
    )
    parser.add_argument(
        "--repetition-unlikelihood-weight",
        type=float,
        default=0.0,
        help="Weight for token-level unlikelihood loss against repeating recent answer tokens.",
    )
    parser.add_argument(
        "--repetition-unlikelihood-window",
        type=int,
        default=32,
        help="How many previous answer tokens to consider for repetition unlikelihood.",
    )
    return TrainConfig(**vars(parser.parse_args()))


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("train", "data", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find records in {path}")


def load_records(cfg: TrainConfig, split: str) -> list[dict[str, Any]]:
    if cfg.dataset_path is not None:
        return read_json_records(cfg.dataset_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for --dataset-name loading") from exc

    token = cfg.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ds = load_dataset(cfg.dataset_name, split=split, token=token)
    return [dict(item) for item in ds]


def message_content(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, dict):
        value = message.get("content") or message.get("value") or message.get("text") or ""
        return str(value).strip()
    return str(message).strip()


def message_role(message: Any, index: int) -> str:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"human", "user"}:
            return "user"
        if role in {"gpt", "assistant", "bot"}:
            return "assistant"
    return "user" if index % 2 == 0 else "assistant"


def render_turn(role: str, content: str) -> str:
    label = "User" if role == "user" else "Assistant"
    return f"{label}: {content.strip()}\n"


def conversation_episodes(record: dict[str, Any]) -> list[dict[str, str]]:
    conversation = (
        record.get("conversations")
        or record.get("conversation")
        or record.get("messages")
        or record.get("dialog")
    )
    if isinstance(conversation, list):
        turns: list[tuple[str, str]] = []
        for idx, message in enumerate(conversation):
            content = message_content(message)
            if content:
                turns.append((message_role(message, idx), content))

        episodes = []
        for idx, (role, content) in enumerate(turns):
            if role != "assistant":
                continue
            prefill_turns = turns[:idx]
            if not prefill_turns:
                continue
            prefill = "".join(render_turn(turn_role, turn_content) for turn_role, turn_content in prefill_turns)
            episodes.append({
                "prefill": prefill,
                "query": "Assistant:",
                "target": " " + content,
            })
        if episodes:
            return episodes

    instruction = (
        record.get("instruction")
        or record.get("prompt")
        or record.get("question")
        or record.get("input")
    )
    answer = (
        record.get("output")
        or record.get("response")
        or record.get("answer")
        or record.get("completion")
    )
    if instruction and answer:
        extra_input = ""
        if record.get("input") and record.get("input") != instruction:
            extra_input = "\n" + str(record["input"]).strip()
        return [{
            "prefill": f"User: {str(instruction).strip()}{extra_input}\n",
            "query": "Assistant:",
            "target": " " + str(answer).strip(),
        }]

    return []


def build_episodes(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    episodes: list[dict[str, str]] = []
    for record in records:
        episodes.extend(conversation_episodes(record))
        if len(episodes) >= limit:
            break
    return episodes[:limit]


def encode_limited(tokenizer: AutoTokenizer, text: str, limit: int, keep: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if limit > 0 and len(ids) > limit:
        if keep == "right":
            ids = ids[-limit:]
        else:
            ids = ids[:limit]
    return ids


@contextmanager
def frozen_memory_updates(model: torch.nn.Module):
    originals = []
    for block in model.blocks:
        memory = block.memory
        originals.append((memory, memory.forward))

        def no_update_forward(x, state=None, return_state=True, _memory=memory):
            batch_size, _seq_len, _dim = x.shape
            if state is None:
                state = _memory.init_state(batch_size, x.device)
            retrieved = _memory.retrieve(x, state)
            if return_state:
                return retrieved, state
            return retrieved, None

        memory.forward = no_update_forward

    try:
        yield
    finally:
        for memory, original in originals:
            memory.forward = original


def detach_states(states):
    if states is None:
        return None
    return [state.detach() if state is not None else None for state in states]


def serialize_states(states) -> list[dict[str, list[torch.Tensor]]] | None:
    if states is None:
        return None

    serialized = []
    for state in states:
        if state is None:
            serialized.append({"weights": [], "momentum": []})
            continue
        serialized.append({
            "weights": [weight.detach().cpu() for weight in state.weights],
            "momentum": [momentum.detach().cpu() for momentum in state.momentum],
        })
    return serialized


def _resize_state_tensor(tensor: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    value = tensor.detach().to(device)
    if value.ndim == 3 and value.shape[0] != batch_size:
        if value.shape[0] > batch_size:
            value = value[:batch_size]
        elif value.shape[0] == 1:
            value = value.expand(batch_size, -1, -1)
        else:
            value = value[:1].expand(batch_size, -1, -1)
    return value.clone()


def deserialize_states(payload, model: torch.nn.Module, device: torch.device, batch_size: int = 1):
    if not payload:
        return None

    states = []
    for block, state_payload in zip(model.blocks, payload, strict=False):
        state = block.memory.init_state(batch_size, device)
        weights = state_payload.get("weights") or []
        momentum = state_payload.get("momentum") or []
        if weights:
            state.weights = [
                _resize_state_tensor(weight, batch_size, device)
                for weight in weights
            ]
        if momentum:
            state.momentum = [
                _resize_state_tensor(value, batch_size, device)
                for value in momentum
            ]
        states.append(state)
    return states


def select_runtime_train_states(checkpoint: dict[str, Any], batch_index: int = 0):
    runtime = checkpoint.get("runtime_memory_state") or {}
    states = runtime.get("train_states")
    if not states:
        return None

    selected = []
    for state_payload in states:
        next_payload = {"weights": [], "momentum": []}
        for key in ("weights", "momentum"):
            for tensor in state_payload.get(key) or []:
                value = tensor
                if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] > 1:
                    index = min(max(0, batch_index), value.shape[0] - 1)
                    value = value[index : index + 1].clone()
                next_payload[key].append(value)
        selected.append(next_payload)
    return selected


def state_norm_summary(states) -> dict[str, Any]:
    if states is None:
        return {"layers": 0, "weight_norm_sum": 0.0, "momentum_norm_sum": 0.0}

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
        "layers": layers,
        "weight_norm_sum": weight_norm,
        "momentum_norm_sum": momentum_norm,
    }


def summarize_ints(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def init_batched_memory_states(model: torch.nn.Module, batch_size: int, device: torch.device):
    return [
        block.memory.init_state(batch_size, device)
        for block in model.blocks
    ]


def gather_state_rows(states, row_indices: list[int], device: torch.device):
    if states is None:
        return None
    rows = torch.tensor(row_indices, dtype=torch.long, device=device)
    gathered = []
    for state in states:
        gathered.append(type(state)(
            weights=[weight.index_select(0, rows.to(weight.device)) for weight in state.weights],
            momentum=[momentum.index_select(0, rows.to(momentum.device)) for momentum in state.momentum],
        ))
    return gathered


def scatter_state_rows(states, row_states, row_indices: list[int], device: torch.device):
    if states is None or row_states is None:
        return states
    rows = torch.tensor(row_indices, dtype=torch.long, device=device)
    scattered = []
    for state, row_state in zip(states, row_states, strict=True):
        weights = []
        momentum = []
        for base, update in zip(state.weights, row_state.weights, strict=True):
            next_value = base.clone()
            next_value.index_copy_(0, rows.to(base.device), update.to(base.device, base.dtype))
            weights.append(next_value)
        for base, update in zip(state.momentum, row_state.momentum, strict=True):
            next_value = base.clone()
            next_value.index_copy_(0, rows.to(base.device), update.to(base.device, base.dtype))
            momentum.append(next_value)
        scattered.append(type(state)(weights=weights, momentum=momentum))
    return scattered


def runtime_memory_payload(cfg: TrainConfig, states) -> dict[str, Any]:
    return {
        "format": "memory_mac_runtime_memory_v1",
        "fixed_size": True,
        "train_mode": (
            "single_persistent_lima_bank"
            if cfg.persistent_memory
            else "item_boundary_batch_reset"
            if cfg.item_boundary_batch
            else "none"
        ),
        "train_states": serialize_states(states),
        "eval_states": None,
        "persistent_memory": bool(cfg.persistent_memory),
        "item_boundary_batch": bool(cfg.item_boundary_batch),
        "write_target_to_memory": bool(cfg.write_target_to_memory),
        "state_norms": state_norm_summary(states),
    }


def configure_instruction_persistent_tokens(
    model: torch.nn.Module,
    init_mode: str,
    init_std: float,
) -> dict[str, Any]:
    if init_mode == "none":
        return {"mode": init_mode, "layers": 0, "tokens": 0}

    layers = 0
    tokens = 0
    for block in model.blocks:
        current = block.persistent.tokens
        if current is None:
            continue
        if init_mode == "fresh":
            next_tokens = torch.empty_like(current)
            nn.init.normal_(next_tokens, std=init_std)
        elif init_mode == "copy":
            next_tokens = current.detach().clone()
        else:
            raise ValueError(f"Unknown instruction persistent init mode: {init_mode}")
        block.persistent.tokens = nn.Parameter(next_tokens)
        layers += 1
        tokens += int(next_tokens.shape[0])

    return {
        "mode": init_mode,
        "layers": layers,
        "tokens": tokens,
        "init_std": init_std,
    }


def prefill_memory_graph(
    model: torch.nn.Module,
    token_ids: list[int],
    batch_tokens: int,
    device: torch.device,
    states=None,
):
    if not token_ids:
        return states
    chunk_size = int(getattr(model.config, "chunk_size", batch_tokens))
    step_tokens = max(1, min(batch_tokens, chunk_size))
    for start in range(0, len(token_ids), step_tokens):
        chunk = token_ids[start : start + step_tokens]
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        _, states = model(x, states=states)
    return states


def batched_prefill_memory_graph(
    model: torch.nn.Module,
    batch_token_ids: list[list[int]],
    batch_tokens: int,
    device: torch.device,
    states=None,
):
    """Prefill independent batch rows without crossing item boundaries."""
    batch_size = len(batch_token_ids)
    if batch_size == 0:
        return states
    if states is None:
        states = init_batched_memory_states(model, batch_size, device)

    max_tokens = max((len(ids) for ids in batch_token_ids), default=0)
    if max_tokens == 0:
        return states

    chunk_size = int(getattr(model.config, "chunk_size", batch_tokens))
    step_tokens = max(1, min(batch_tokens, chunk_size))
    for start in range(0, max_tokens, step_tokens):
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for row_idx, token_ids in enumerate(batch_token_ids):
            chunk = token_ids[start : start + step_tokens]
            if chunk:
                groups.setdefault(len(chunk), []).append((row_idx, chunk))

        for _length, items in sorted(groups.items()):
            rows = [row_idx for row_idx, _chunk in items]
            x = torch.tensor(
                [chunk for _row_idx, chunk in items],
                dtype=torch.long,
                device=device,
            )
            row_states = gather_state_rows(states, rows, device)
            _, new_row_states = model(x, states=row_states)
            states = scatter_state_rows(states, new_row_states, rows, device)
    return states


def weighted_answer_ce(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        target_logits.reshape(-1, target_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    if cfg.answer_start_weight == 1.0 or cfg.answer_start_tokens <= 0:
        return token_losses.mean()

    weights = torch.ones_like(token_losses)
    start_tokens = min(cfg.answer_start_tokens, token_losses.numel())
    weights[:start_tokens] = float(cfg.answer_start_weight)
    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def repetition_unlikelihood_loss(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.repetition_unlikelihood_weight <= 0.0 or cfg.repetition_unlikelihood_window <= 0:
        return target_logits.new_zeros(())

    probs = F.softmax(target_logits.float(), dim=-1)
    losses = []
    for pos in range(1, labels.numel()):
        start = max(0, pos - cfg.repetition_unlikelihood_window)
        candidates = labels[start:pos]
        candidates = candidates[candidates != labels[pos]]
        if candidates.numel() == 0:
            continue
        candidates = torch.unique(candidates)
        candidate_probs = probs[pos, candidates].clamp(max=1.0 - 1e-6)
        losses.append(-torch.log1p(-candidate_probs).mean())

    if not losses:
        return target_logits.new_zeros(())
    return torch.stack(losses).mean()


def weighted_answer_ce_segments(
    logit_segments: list[torch.Tensor],
    label_segments: list[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    flat_logits = torch.cat(logit_segments, dim=0)
    flat_labels = torch.cat(label_segments, dim=0)
    token_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if cfg.answer_start_weight == 1.0 or cfg.answer_start_tokens <= 0:
        return token_losses.mean()

    weights = []
    for labels in label_segments:
        row_weights = torch.ones(labels.numel(), dtype=token_losses.dtype, device=token_losses.device)
        row_weights[: min(cfg.answer_start_tokens, labels.numel())] = float(cfg.answer_start_weight)
        weights.append(row_weights)
    flat_weights = torch.cat(weights, dim=0)
    return (token_losses * flat_weights).sum() / flat_weights.sum().clamp_min(1.0)


def repetition_unlikelihood_loss_segments(
    logit_segments: list[torch.Tensor],
    label_segments: list[torch.Tensor],
    cfg: TrainConfig,
) -> torch.Tensor:
    if cfg.repetition_unlikelihood_weight <= 0.0:
        return logit_segments[0].new_zeros(())
    losses = [
        repetition_unlikelihood_loss(logits, labels, cfg)
        for logits, labels in zip(logit_segments, label_segments, strict=True)
    ]
    valid = [loss for loss in losses if float(loss.detach().item()) != 0.0]
    if not valid:
        return logit_segments[0].new_zeros(())
    return torch.stack(valid).mean()


def batched_episode_loss_with_item_boundaries(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episodes: list[dict[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Train a batch of independent LIMA items with per-row memory cutoffs."""
    batch_size = len(episodes)
    if batch_size < 1:
        raise RuntimeError("Empty LIMA batch")

    prefill_ids_batch = [
        encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
        for episode in episodes
    ]
    query_ids_batch = [
        encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
        for episode in episodes
    ]
    query_ids_batch = [ids if ids else [eos_id] for ids in query_ids_batch]
    target_ids_batch = []
    for episode in episodes:
        target_ids = encode_limited(tokenizer, episode["target"], cfg.max_answer_tokens - 1, keep="left")
        target_ids.append(eos_id)
        target_ids_batch.append(target_ids)

    states = init_batched_memory_states(model, batch_size, device)
    states = batched_prefill_memory_graph(
        model,
        prefill_ids_batch,
        cfg.prefill_batch_tokens,
        device,
        states=states,
    )

    answer_inputs = [
        query_ids + target_ids[:-1]
        for query_ids, target_ids in zip(query_ids_batch, target_ids_batch, strict=True)
    ]
    max_input_len = max(len(ids) for ids in answer_inputs)
    x = torch.full(
        (batch_size, max_input_len),
        fill_value=eos_id,
        dtype=torch.long,
        device=device,
    )
    for row_idx, ids in enumerate(answer_inputs):
        x[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    with frozen_memory_updates(model):
        logits, _ = model(x, states=states)

    logit_segments = []
    label_segments = []
    for row_idx, (query_ids, target_ids) in enumerate(zip(query_ids_batch, target_ids_batch, strict=True)):
        labels = torch.tensor(target_ids, dtype=torch.long, device=device)
        start = len(query_ids) - 1
        logit_segments.append(logits[row_idx, start : start + labels.numel()])
        label_segments.append(labels)

    ce_loss = weighted_answer_ce_segments(logit_segments, label_segments, cfg)
    unlikelihood_loss = repetition_unlikelihood_loss_segments(logit_segments, label_segments, cfg)
    loss = ce_loss + float(cfg.repetition_unlikelihood_weight) * unlikelihood_loss

    prefill_lengths = [len(ids) for ids in prefill_ids_batch]
    query_lengths = [len(ids) for ids in query_ids_batch]
    target_lengths = [len(ids) for ids in target_ids_batch]
    return loss, {
        "batch_size": batch_size,
        "item_boundary_batch": True,
        "item_boundary_resets": batch_size,
        "prefill_tokens": summarize_ints(prefill_lengths),
        "query_tokens": summarize_ints(query_lengths),
        "target_tokens": summarize_ints(target_lengths),
        "ce_loss": float(ce_loss.detach().item()),
        "repetition_unlikelihood_loss": float(unlikelihood_loss.detach().item()),
        "answer_start_tokens": cfg.answer_start_tokens,
        "answer_start_weight": cfg.answer_start_weight,
        "repetition_unlikelihood_weight": cfg.repetition_unlikelihood_weight,
        "prefill_preview": episodes[0]["prefill"][:160],
        "target_preview": episodes[0]["target"][:160],
    }


def episode_loss_with_state(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states=None,
) -> tuple[torch.Tensor, dict[str, Any], Any]:
    prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
    query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    target_ids = encode_limited(tokenizer, episode["target"], cfg.max_answer_tokens - 1, keep="left")
    target_ids.append(eos_id)
    if not target_ids:
        raise RuntimeError("Empty target")

    states = prefill_memory_graph(model, prefill_ids, cfg.prefill_batch_tokens, device, states=states)
    answer_input = query_ids + target_ids[:-1]
    x = torch.tensor([answer_input], dtype=torch.long, device=device)
    labels = torch.tensor(target_ids, dtype=torch.long, device=device)
    with frozen_memory_updates(model):
        logits, _ = model(x, states=states)

    start = len(query_ids) - 1
    target_logits = logits[0, start : start + len(target_ids)]
    ce_loss = weighted_answer_ce(target_logits, labels, cfg)
    unlikelihood_loss = repetition_unlikelihood_loss(target_logits, labels, cfg)
    loss = ce_loss + float(cfg.repetition_unlikelihood_weight) * unlikelihood_loss
    info = {
        "prefill_tokens": len(prefill_ids),
        "query_tokens": len(query_ids),
        "target_tokens": len(target_ids),
        "ce_loss": float(ce_loss.detach().item()),
        "repetition_unlikelihood_loss": float(unlikelihood_loss.detach().item()),
        "answer_start_tokens": cfg.answer_start_tokens,
        "answer_start_weight": cfg.answer_start_weight,
        "repetition_unlikelihood_weight": cfg.repetition_unlikelihood_weight,
        "prefill_preview": episode["prefill"][:160],
        "target_preview": episode["target"][:160],
    }
    return loss, info, states


def episode_loss(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss, info, _states = episode_loss_with_state(
        model,
        tokenizer,
        episode,
        cfg,
        device,
        eos_id,
        states=None,
    )
    return loss, info


def write_episode_target_to_memory(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
):
    query_text = episode.get("query") or cfg.query_prompt
    target_text = episode["target"]
    text = f"{query_text}{target_text}"
    token_ids = encode_limited(
        tokenizer,
        text,
        max(1, 32 + cfg.max_answer_tokens - 1),
        keep="left",
    )
    token_ids.append(eos_id)
    with torch.no_grad():
        states = prefill_memory_graph(
            model,
            token_ids,
            cfg.prefill_batch_tokens,
            device,
            states=states,
        )
    return detach_states(states)


def score_episode(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    wrong_episode: dict[str, str] | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
        wrong_prefill_ids = (
            encode_limited(tokenizer, wrong_episode["prefill"], cfg.max_prefill_tokens, keep="right")
            if wrong_episode is not None
            else []
        )
        query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
        if not query_ids:
            query_ids = [eos_id]
        target_ids = encode_limited(tokenizer, episode["target"], cfg.max_answer_tokens - 1, keep="left")
        target_ids.append(eos_id)

        def score_with_states(states) -> dict[str, float]:
            answer_input = query_ids + target_ids[:-1]
            x = torch.tensor([answer_input], dtype=torch.long, device=device)
            labels = torch.tensor(target_ids, dtype=torch.long, device=device)
            with frozen_memory_updates(model):
                logits, _ = model(x, states=clone_states(states))
            start = len(query_ids) - 1
            target_logits = logits[0, start : start + len(target_ids)].float()
            nll = float(F.cross_entropy(target_logits, labels, reduction="sum").item())
            avg_nll = nll / len(target_ids)
            return {
                "nll": nll,
                "avg_nll": avg_nll,
                "ppl": math.exp(min(avg_nll, 20.0)),
                "tokens": len(target_ids),
            }

        same_states = prefill_memory_graph(model, prefill_ids, cfg.prefill_batch_tokens, device)
        wrong_states = (
            prefill_memory_graph(model, wrong_prefill_ids, cfg.prefill_batch_tokens, device)
            if wrong_prefill_ids
            else None
        )
        reset = score_with_states(None)
        same = score_with_states(same_states)
        wrong = score_with_states(wrong_states) if wrong_states is not None else None

    model.train()
    return {
        "reset": reset,
        "same_prefill": same,
        "wrong_prefill": wrong,
        "deltas": {
            "same_minus_reset_avg_nll": same["avg_nll"] - reset["avg_nll"],
            "wrong_minus_reset_avg_nll": (
                wrong["avg_nll"] - reset["avg_nll"] if wrong is not None else None
            ),
            "same_minus_wrong_avg_nll": (
                same["avg_nll"] - wrong["avg_nll"] if wrong is not None else None
            ),
        },
    }


def generate_answer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
) -> dict[str, str]:
    model.eval()
    with torch.no_grad():
        prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
        states = prefill_memory_graph(model, prefill_ids, cfg.prefill_batch_tokens, device)
        query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
        if not query_ids:
            query_ids = [eos_id]
        generated = list(query_ids)
        x = torch.tensor([query_ids], dtype=torch.long, device=device)
        with frozen_memory_updates(model):
            logits, states = model(x, states=states)
            new_ids = []
            for _ in range(cfg.max_new_tokens):
                next_id = int(torch.argmax(logits[0, -1].float()).item())
                if next_id == eos_id:
                    break
                new_ids.append(next_id)
                generated.append(next_id)
                x = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits, states = model(x, states=states)
    model.train()
    return {
        "prefill": episode["prefill"],
        "target": episode["target"],
        "generated": tokenizer.decode(new_ids, skip_special_tokens=True),
    }


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def score_episode_from_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> dict[str, Any]:
    query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    target_ids = encode_limited(tokenizer, episode["target"], cfg.max_answer_tokens - 1, keep="left")
    target_ids.append(eos_id)
    answer_input = query_ids + target_ids[:-1]

    x = torch.tensor([answer_input], dtype=torch.long, device=device)
    labels = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad(), frozen_memory_updates(model):
        logits, _ = model(x, states=states)

    start = len(query_ids) - 1
    target_logits = logits[0, start : start + len(target_ids)].float()
    nll = float(F.cross_entropy(target_logits, labels, reduction="sum").item())
    avg_nll = nll / len(target_ids)
    first_logits = target_logits[0]
    first_id = int(target_ids[0])
    first_probs = F.softmax(first_logits, dim=-1)
    return {
        "nll": nll,
        "avg_nll": avg_nll,
        "ppl": math.exp(min(avg_nll, 20.0)),
        "tokens": len(target_ids),
        "first_token": {
            "id": first_id,
            "text": tokenizer.decode([first_id]),
            "prob": float(first_probs[first_id].item()),
            "rank": rank_of(first_logits, first_id),
        },
    }


def prefill_then_score_episode_from_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> tuple[dict[str, Any], Any]:
    prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
    with torch.no_grad():
        states = prefill_memory_graph(
            model,
            prefill_ids,
            cfg.prefill_batch_tokens,
            device,
            states=states,
        )
    states = detach_states(states)
    return score_episode_from_bank(
        model,
        tokenizer,
        episode,
        cfg,
        device,
        eos_id,
        states,
    ), states


def generate_answer_from_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> dict[str, str]:
    model.eval()
    with torch.no_grad(), frozen_memory_updates(model):
        query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
        if not query_ids:
            query_ids = [eos_id]
        x = torch.tensor([query_ids], dtype=torch.long, device=device)
        logits, states = model(x, states=states)
        new_ids = []
        for _ in range(cfg.max_new_tokens):
            next_id = int(torch.argmax(logits[0, -1].float()).item())
            if next_id == eos_id:
                break
            new_ids.append(next_id)
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, states = model(x, states=states)
    model.train()
    return {
        "prefill": episode["prefill"],
        "target": episode["target"],
        "generated": tokenizer.decode(new_ids, skip_special_tokens=True),
    }


def generate_prefill_then_answer_from_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> tuple[dict[str, str], Any]:
    prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
    with torch.no_grad():
        states = prefill_memory_graph(
            model,
            prefill_ids,
            cfg.prefill_batch_tokens,
            device,
            states=states,
        )
    states = detach_states(states)
    return generate_answer_from_bank(
        model,
        tokenizer,
        episode,
        cfg,
        device,
        eos_id,
        states,
    ), states


def evaluate_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episodes: list[dict[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> dict[str, Any]:
    model.eval()
    rows = [
        score_episode_from_bank(model, tokenizer, episode, cfg, device, eos_id, states)
        for episode in episodes
    ]
    model.train()
    if not rows:
        return {"examples": 0}

    mean_nll = sum(row["avg_nll"] for row in rows) / len(rows)
    ranks = [row["first_token"]["rank"] for row in rows]
    return {
        "examples": len(rows),
        "bank": {
            "mean_avg_nll": mean_nll,
            "ppl_from_mean_nll": math.exp(min(mean_nll, 20.0)),
            "mean_first_token_rank": sum(ranks) / len(ranks),
            "median_first_token_rank": sorted(ranks)[len(ranks) // 2],
        },
        "runtime_memory_state": state_norm_summary(states),
    }


def evaluate_prefill_then_answer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episodes: list[dict[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
) -> dict[str, Any]:
    model.eval()
    eval_states = clone_states(states)
    rows = []
    for episode in episodes:
        row, eval_states = prefill_then_score_episode_from_bank(
            model,
            tokenizer,
            episode,
            cfg,
            device,
            eos_id,
            eval_states,
        )
        rows.append(row)
    model.train()
    if not rows:
        return {"examples": 0}

    mean_nll = sum(row["avg_nll"] for row in rows) / len(rows)
    ranks = [row["first_token"]["rank"] for row in rows]
    return {
        "examples": len(rows),
        "prefill_then_answer": {
            "mean_avg_nll": mean_nll,
            "ppl_from_mean_nll": math.exp(min(mean_nll, 20.0)),
            "mean_first_token_rank": sum(ranks) / len(ranks),
            "median_first_token_rank": sorted(ranks)[len(ranks) // 2],
        },
        "runtime_memory_state_before_eval": state_norm_summary(states),
        "runtime_memory_state_after_eval": state_norm_summary(eval_states),
    }


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> LambdaLR:
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, cfg.steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episodes: list[dict[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
) -> dict[str, Any]:
    rows = []
    for idx, episode in enumerate(episodes):
        wrong = episodes[(idx + 1) % len(episodes)] if len(episodes) > 1 else None
        rows.append(score_episode(model, tokenizer, episode, cfg, device, eos_id, wrong))

    summary: dict[str, Any] = {"examples": len(rows)}
    for key in ("reset", "same_prefill", "wrong_prefill"):
        valid = [row[key] for row in rows if row.get(key) is not None]
        if valid:
            mean_nll = sum(item["avg_nll"] for item in valid) / len(valid)
            summary[key] = {
                "mean_avg_nll": mean_nll,
                "ppl_from_mean_nll": math.exp(min(mean_nll, 20.0)),
            }

    for key in ("same_minus_reset_avg_nll", "wrong_minus_reset_avg_nll", "same_minus_wrong_avg_nll"):
        values = [row["deltas"][key] for row in rows if row["deltas"].get(key) is not None]
        if values:
            summary[key] = {
                "mean": sum(values) / len(values),
                "wins_negative": sum(1 for value in values if value < 0.0),
                "total": len(values),
            }
    return summary


def output_dir(cfg: TrainConfig) -> Path:
    if cfg.out_dir is not None:
        return cfg.out_dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("artifacts/runs") / f"{cfg.run_name}-{stamp}"


def main() -> None:
    cfg = parse_args()
    if cfg.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if cfg.item_boundary_batch and cfg.persistent_memory:
        raise ValueError("--item-boundary-batch is intentionally separate from --persistent-memory")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    checkpoint = load_checkpoint(cfg.checkpoint, torch.device("cpu"))
    ckpt_cfg = merge_model_config(checkpoint, torch.device("cpu"))
    tokenizer_name = ckpt_cfg.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name} must have eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_records(cfg, cfg.train_split)
    eval_records = (
        load_records(cfg, cfg.eval_split)
        if cfg.eval_split and cfg.dataset_path is None
        else []
    )
    train_limit = max(cfg.max_train_examples + cfg.max_eval_examples, cfg.max_train_examples)
    episodes = build_episodes(train_records, train_limit)
    if len(episodes) < 2:
        raise RuntimeError("Need at least two LIMA-style prompt/answer episodes")

    explicit_eval_episodes = build_episodes(eval_records, cfg.max_eval_examples) if eval_records else []
    eval_count = min(cfg.max_eval_examples, max(1, len(episodes) // 10))
    if cfg.persistent_memory:
        train_episodes = episodes[: cfg.max_train_examples]
        eval_episodes = train_episodes[:eval_count]
    else:
        eval_episodes = explicit_eval_episodes or episodes[:eval_count]
        train_episodes = episodes[0: cfg.max_train_examples] if explicit_eval_episodes else episodes[eval_count: cfg.max_train_examples + eval_count]
        if not train_episodes:
            train_episodes = episodes[eval_count:]

    model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    apply_memory_controls(
        model,
        float(ckpt_cfg.get("memory_decay", 0.0)),
        float(ckpt_cfg.get("memory_lr", 0.1)),
        float(ckpt_cfg.get("memory_momentum", 0.9)),
    )
    instruction_persistent_tokens = configure_instruction_persistent_tokens(
        model,
        cfg.instruction_persistent_init,
        float(ckpt_cfg.get("persistent_init_std", 0.02)),
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    cfg_dict = asdict(cfg)
    cfg_dict["hf_token"] = "<redacted>" if cfg.hf_token else None
    model_config_payload = {
        key: ckpt_cfg[key]
        for key in MODEL_CONFIG_KEYS
        if key in ckpt_cfg
    }
    config_payload = {
        **model_config_payload,
        **cfg_dict,
        "checkpoint": str(cfg.checkpoint),
        "dataset_path": str(cfg.dataset_path) if cfg.dataset_path else None,
        "out_dir": str(out),
        "device": str(device),
        "tokenizer": tokenizer_name,
        "base_model_config": model_config_payload,
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "base_checkpoint_step": checkpoint.get("step"),
        "base_checkpoint_best_val": checkpoint.get("best_val"),
        "base_checkpoint_has_runtime_memory_state": bool(checkpoint.get("runtime_memory_state")),
        "instruction_persistent_tokens": instruction_persistent_tokens,
        "runtime_memory_state_shape": "fixed-size per layer; no KV/cache growth",
        "autograd_through_memory_state": "truncated by detach between LIMA episodes",
        "item_boundary_batch": bool(cfg.item_boundary_batch),
        "item_boundary_policy": (
            "each batch row is one LIMA item; memory state is reset to learned initial state at item boundary; segments never cross items"
            if cfg.item_boundary_batch
            else None
        ),
        "lima_eval_protocol": "persistent_prefill_then_readonly_answer" if cfg.persistent_memory else "same_vs_reset_controls",
    }
    (out / "config.json").write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    print({"status": "starting", "out_dir": str(out), "train_episodes": len(train_episodes), "eval_episodes": len(eval_episodes)}, flush=True)

    best_eval = float("inf")
    eos_id = int(tokenizer.eos_token_id)
    train_states = None
    loaded_runtime_state = False
    if cfg.persistent_memory and cfg.load_runtime_memory_state:
        runtime_states = select_runtime_train_states(
            checkpoint,
            cfg.runtime_memory_batch_index,
        )
        train_states = deserialize_states(runtime_states, model, device, batch_size=1)
        loaded_runtime_state = train_states is not None

    train_order = list(range(len(train_episodes)))
    if cfg.persistent_memory or cfg.item_boundary_batch:
        random.shuffle(train_order)
    train_cursor = 0
    for step in range(1, cfg.steps + 1):
        if cfg.item_boundary_batch:
            batch_indices = []
            for _ in range(cfg.batch_size):
                if train_cursor >= len(train_order):
                    train_cursor = 0
                    random.shuffle(train_order)
                batch_indices.append(train_order[train_cursor])
                train_cursor += 1
            batch_episodes = [train_episodes[idx] for idx in batch_indices]
            episode_index = batch_indices[0]
            episode = batch_episodes[0]
        elif cfg.persistent_memory:
            if train_cursor >= len(train_order):
                train_cursor = 0
                random.shuffle(train_order)
            episode_index = train_order[train_cursor]
            train_cursor += 1
            episode = train_episodes[episode_index]
            batch_indices = [episode_index]
            batch_episodes = [episode]
        else:
            episode_index = -1
            episode = random.choice(train_episodes)
            batch_indices = [episode_index]
            batch_episodes = [episode]
        optimizer.zero_grad(set_to_none=True)
        if cfg.item_boundary_batch:
            loss, info = batched_episode_loss_with_item_boundaries(
                model,
                tokenizer,
                batch_episodes,
                cfg,
                device,
                eos_id,
            )
        elif cfg.persistent_memory:
            loss, info, train_states = episode_loss_with_state(
                model,
                tokenizer,
                episode,
                cfg,
                device,
                eos_id,
                states=train_states,
            )
        else:
            loss, info = episode_loss(model, tokenizer, episode, cfg, device, eos_id)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item())
        optimizer.step()
        scheduler.step()
        if cfg.persistent_memory:
            train_states = detach_states(train_states)
            if cfg.write_target_to_memory:
                train_states = write_episode_target_to_memory(
                    model,
                    tokenizer,
                    episode,
                    cfg,
                    device,
                    eos_id,
                    train_states,
                )

        record = {
            "step": step,
            "train_loss": float(loss.detach().item()),
            "train_ppl": math.exp(min(float(loss.detach().item()), 20.0)),
            "grad_norm": grad_norm,
            "lr": float(scheduler.get_last_lr()[0]),
            "episode_index": episode_index,
            "batch_episode_indices": batch_indices,
            "effective_examples_per_step": cfg.batch_size if cfg.item_boundary_batch else 1,
            "item_boundary_batch": cfg.item_boundary_batch,
            "persistent_memory": cfg.persistent_memory,
            "loaded_runtime_memory_state": loaded_runtime_state,
            "write_target_to_memory": cfg.write_target_to_memory,
            "instruction_persistent_tokens": instruction_persistent_tokens,
            "runtime_memory_state": state_norm_summary(train_states) if cfg.persistent_memory else {},
            **info,
        }

        if step % cfg.eval_every == 0 or step == 1:
            if cfg.persistent_memory:
                eval_summary = evaluate_prefill_then_answer(
                    model,
                    tokenizer,
                    eval_episodes,
                    cfg,
                    device,
                    eos_id,
                    train_states,
                )
                current_eval = eval_summary["prefill_then_answer"]["mean_avg_nll"]
            else:
                eval_summary = evaluate(model, tokenizer, eval_episodes, cfg, device, eos_id)
                current_eval = eval_summary["same_prefill"]["mean_avg_nll"]
            record["eval"] = eval_summary
            if current_eval < best_eval:
                best_eval = current_eval
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "best_eval": best_eval,
                    "config": config_payload,
                    "base_checkpoint": str(cfg.checkpoint),
                    "runtime_memory_state": runtime_memory_payload(cfg, train_states),
                }, out / "best.pt")

        if step % cfg.save_every == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "best_eval": best_eval,
                "config": config_payload,
                "base_checkpoint": str(cfg.checkpoint),
                "runtime_memory_state": runtime_memory_payload(cfg, train_states),
            }, out / f"checkpoint-step-{step}.pt")

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        print(record, flush=True)

    if cfg.persistent_memory:
        final_eval = evaluate_prefill_then_answer(model, tokenizer, eval_episodes, cfg, device, eos_id, train_states)
        sample_states = clone_states(train_states)
        samples = []
        for episode in eval_episodes[: max(0, cfg.generate_samples)]:
            sample, sample_states = generate_prefill_then_answer_from_bank(
                model,
                tokenizer,
                episode,
                cfg,
                device,
                eos_id,
                sample_states,
            )
            samples.append(sample)
    else:
        final_eval = evaluate(model, tokenizer, eval_episodes, cfg, device, eos_id)
        samples = [
            generate_answer(model, tokenizer, episode, cfg, device, eos_id)
            for episode in eval_episodes[: max(0, cfg.generate_samples)]
        ]
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": cfg.steps,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "config": config_payload,
        "base_checkpoint": str(cfg.checkpoint),
        "runtime_memory_state": runtime_memory_payload(cfg, train_states),
    }, out / "final.pt")
    (out / "eval_samples.json").write_text(json.dumps({
        "final_eval": final_eval,
        "samples": samples,
    }, indent=2), encoding="utf-8")
    print({"status": "done", "out_dir": str(out), "best_eval": best_eval, "final_eval": final_eval}, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
