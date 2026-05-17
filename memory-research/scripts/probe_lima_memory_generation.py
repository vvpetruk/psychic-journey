from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_article_memory import apply_memory_controls, clone_states  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint, sample_next_id  # noqa: E402
from train_lima_memory_task import (  # noqa: E402
    TrainConfig,
    build_episodes,
    encode_limited,
    frozen_memory_updates,
    load_records,
    prefill_memory_graph,
)


MODEL_CONFIG_KEYS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe LIMA-style frozen-memory generation and first-token probabilities."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-eval-examples", type=int, default=64)
    parser.add_argument("--max-prefill-tokens", type=int, default=192)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--prefill-batch-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def merge_model_config(checkpoint: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = dict(checkpoint.get("config") or {})
    missing_model_keys = not MODEL_CONFIG_KEYS.intersection(cfg)
    base_path = checkpoint.get("base_checkpoint") or cfg.get("checkpoint")
    if missing_model_keys and base_path:
        base = load_checkpoint(Path(base_path), device)
        merged = dict(base.get("config") or {})
        merged.update(cfg)
        return merged
    return cfg


def build_cfg(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset_path,
        max_eval_examples=args.max_eval_examples,
        max_prefill_tokens=args.max_prefill_tokens,
        max_answer_tokens=args.max_answer_tokens,
        prefill_batch_tokens=args.prefill_batch_tokens,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def select_eval_episodes(args: argparse.Namespace, cfg: TrainConfig) -> list[dict[str, str]]:
    records = load_records(cfg, cfg.train_split)
    limit = max(args.max_eval_examples, args.sample_count)
    return build_episodes(records, limit)


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def top_tokens(tokenizer: AutoTokenizer, logits: torch.Tensor, k: int = 8) -> list[dict[str, Any]]:
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        rows.append({
            "id": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "prob": float(prob),
        })
    return rows


def first_token_probe(
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
    target_id = target_ids[0]
    x = torch.tensor([query_ids], dtype=torch.long, device=device)
    with torch.no_grad(), frozen_memory_updates(model):
        logits, _ = model(x, states=clone_states(states))
    next_logits = logits[0, -1].float()
    probs = F.softmax(next_logits, dim=-1)
    return {
        "target_id": int(target_id),
        "target_text": tokenizer.decode([target_id]),
        "target_prob": float(probs[target_id].item()),
        "target_rank": rank_of(next_logits, target_id),
        "eos_prob": float(probs[eos_id].item()),
        "eos_rank": rank_of(next_logits, eos_id),
        "top": top_tokens(tokenizer, next_logits),
    }


def sample_answer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
    *,
    force_min_tokens: bool,
    greedy: bool,
    args: argparse.Namespace,
) -> str:
    query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    x = torch.tensor([query_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    with torch.no_grad(), frozen_memory_updates(model):
        logits, states = model(x, states=clone_states(states))
        for _ in range(args.max_new_tokens):
            next_logits = logits[0, -1].float().clone()
            if force_min_tokens and len(new_ids) < args.min_new_tokens:
                next_logits[eos_id] = float("-inf")
            next_id = sample_next_id(
                next_logits,
                temperature=0.0 if greedy else args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            if next_id == eos_id:
                break
            new_ids.append(next_id)
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, states = model(x, states=states)
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(rows)}
    for memory_key in ("reset", "same_prefill", "wrong_prefill"):
        probes = [row["first_token"][memory_key] for row in rows]
        summary[memory_key] = {
            "eos_top1": sum(1 for item in probes if item["eos_rank"] == 1),
            "mean_eos_prob": sum(item["eos_prob"] for item in probes) / len(probes),
            "mean_target_prob": sum(item["target_prob"] for item in probes) / len(probes),
            "mean_target_rank": sum(item["target_rank"] for item in probes) / len(probes),
            "median_target_rank": sorted(item["target_rank"] for item in probes)[len(probes) // 2],
        }
    same_ranks = [row["first_token"]["same_prefill"]["target_rank"] for row in rows]
    reset_ranks = [row["first_token"]["reset"]["target_rank"] for row in rows]
    wrong_ranks = [row["first_token"]["wrong_prefill"]["target_rank"] for row in rows]
    summary["same_rank_beats_reset"] = sum(1 for same, reset in zip(same_ranks, reset_ranks) if same < reset)
    summary["same_rank_beats_wrong"] = sum(1 for same, wrong in zip(same_ranks, wrong_ranks) if same < wrong)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    merged_cfg = merge_model_config(checkpoint, torch.device("cpu"))
    checkpoint_for_build = deepcopy(checkpoint)
    checkpoint_for_build["config"] = merged_cfg
    tokenizer_name = merged_cfg.get("tokenizer_name") or merged_cfg.get("tokenizer") or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name} must have eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(checkpoint_for_build, int(tokenizer.vocab_size), device)
    apply_memory_controls(
        model,
        float(merged_cfg.get("memory_decay", 0.0)),
        float(merged_cfg.get("memory_lr", 0.1)),
        float(merged_cfg.get("memory_momentum", 0.9)),
    )
    model.eval()

    cfg = build_cfg(args)
    episodes = select_eval_episodes(args, cfg)[: args.sample_count]
    eos_id = int(tokenizer.eos_token_id)
    rows = []

    for idx, episode in enumerate(episodes):
        wrong_episode = episodes[(idx + 1) % len(episodes)] if len(episodes) > 1 else episode
        prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
        wrong_prefill_ids = encode_limited(tokenizer, wrong_episode["prefill"], cfg.max_prefill_tokens, keep="right")
        same_states = prefill_memory_graph(model, prefill_ids, cfg.prefill_batch_tokens, device)
        wrong_states = prefill_memory_graph(model, wrong_prefill_ids, cfg.prefill_batch_tokens, device)

        first_token = {
            "reset": first_token_probe(model, tokenizer, episode, cfg, device, eos_id, None),
            "same_prefill": first_token_probe(model, tokenizer, episode, cfg, device, eos_id, same_states),
            "wrong_prefill": first_token_probe(model, tokenizer, episode, cfg, device, eos_id, wrong_states),
        }
        rows.append({
            "index": idx,
            "prefill": episode["prefill"],
            "target": episode["target"],
            "first_token": first_token,
            "greedy_same_prefill": sample_answer(
                model, tokenizer, episode, cfg, device, eos_id, same_states,
                force_min_tokens=False, greedy=True, args=args,
            ),
            "sampled_same_prefill_min_tokens": sample_answer(
                model, tokenizer, episode, cfg, device, eos_id, same_states,
                force_min_tokens=True, greedy=False, args=args,
            ),
        })

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(args.dataset_path),
        "device": str(device),
        "max_prefill_tokens": args.max_prefill_tokens,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "summary": summarize(rows),
        "samples": rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
