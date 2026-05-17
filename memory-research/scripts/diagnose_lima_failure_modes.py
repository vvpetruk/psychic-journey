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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from probe_article_memory import clone_states  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint, sample_next_id  # noqa: E402
from train_lima_memory_task import (  # noqa: E402
    TrainConfig,
    build_episodes,
    deserialize_states,
    detach_states,
    encode_limited,
    frozen_memory_updates,
    load_records,
    merge_model_config,
    prefill_memory_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose LIMA memory/use failure modes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--max-prefill-tokens", type=int, default=192)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--prefill-batch-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def score_logits(
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
        logits, _ = model(x, states=clone_states(states))

    start = len(query_ids) - 1
    target_logits = logits[0, start : start + len(target_ids)].float()
    token_losses = F.cross_entropy(target_logits, labels, reduction="none")
    ranks = [rank_of(target_logits[pos], int(labels[pos])) for pos in range(labels.numel())]
    probs = F.softmax(target_logits, dim=-1)
    gold_probs = [float(probs[pos, int(labels[pos])].item()) for pos in range(labels.numel())]
    top_ids = [int(torch.argmax(target_logits[pos]).item()) for pos in range(labels.numel())]
    return {
        "avg_nll": float(token_losses.mean().item()),
        "ppl": math.exp(min(float(token_losses.mean().item()), 20.0)),
        "token_losses": [float(v.item()) for v in token_losses],
        "ranks": ranks,
        "gold_probs": gold_probs,
        "top_tokens": [tokenizer.decode([idx]) for idx in top_ids[:16]],
    }


def prefill_episode(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    states,
):
    prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
    with torch.no_grad():
        states = prefill_memory_graph(
            model,
            prefill_ids,
            cfg.prefill_batch_tokens,
            device,
            states=states,
        )
    return detach_states(states)


def generate_after_gold_prefix(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
    gold_prefix_tokens: int,
    args: argparse.Namespace,
) -> str:
    query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    target_ids = encode_limited(tokenizer, episode["target"], cfg.max_answer_tokens - 1, keep="left")
    forced = target_ids[:gold_prefix_tokens]
    prompt_ids = query_ids + forced
    if not prompt_ids:
        prompt_ids = [eos_id]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    with torch.no_grad(), frozen_memory_updates(model):
        logits, states = model(x, states=clone_states(states))
        for _ in range(args.max_new_tokens):
            next_id = sample_next_id(
                logits[0, -1].float(),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            if next_id == eos_id:
                break
            new_ids.append(next_id)
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, states = model(x, states=states)
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bucket_rank_summary(rows: list[dict[str, Any]], positions: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pos in positions:
        ranks = [row["same_score"]["ranks"][pos] for row in rows if len(row["same_score"]["ranks"]) > pos]
        losses = [row["same_score"]["token_losses"][pos] for row in rows if len(row["same_score"]["token_losses"]) > pos]
        if ranks:
            out[str(pos)] = {
                "mean_rank": mean([float(v) for v in ranks]),
                "median_rank": sorted(ranks)[len(ranks) // 2],
                "top1": sum(1 for v in ranks if v == 1),
                "mean_nll": mean(losses),
            }
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    merged_cfg = merge_model_config(checkpoint, torch.device("cpu"))
    checkpoint_for_build = deepcopy(checkpoint)
    checkpoint_for_build["config"] = merged_cfg

    tokenizer_name = merged_cfg.get("tokenizer_name") or merged_cfg.get("tokenizer") or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = int(tokenizer.eos_token_id)

    model = build_model(checkpoint_for_build, int(tokenizer.vocab_size), device)
    model.eval()

    runtime = checkpoint.get("runtime_memory_state") or {}
    base_states = deserialize_states(runtime.get("train_states"), model, device, batch_size=1)
    if base_states is None:
        raise RuntimeError("Checkpoint lacks runtime_memory_state.train_states")

    cfg = TrainConfig(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset_path,
        max_prefill_tokens=args.max_prefill_tokens,
        max_answer_tokens=args.max_answer_tokens,
        prefill_batch_tokens=args.prefill_batch_tokens,
        persistent_memory=True,
    )
    records = load_records(cfg, cfg.train_split)
    episodes = build_episodes(records, max(args.sample_count + 1, 2))[: args.sample_count]

    rows = []
    for idx, episode in enumerate(episodes):
        wrong = episodes[(idx + 1) % len(episodes)]
        no_prefill_states = clone_states(base_states)
        same_states = prefill_episode(model, tokenizer, episode, cfg, device, clone_states(base_states))
        wrong_states = prefill_episode(model, tokenizer, wrong, cfg, device, clone_states(base_states))
        same_score = score_logits(model, tokenizer, episode, cfg, device, eos_id, same_states)
        no_prefill_score = score_logits(model, tokenizer, episode, cfg, device, eos_id, no_prefill_states)
        wrong_score = score_logits(model, tokenizer, episode, cfg, device, eos_id, wrong_states)
        rows.append({
            "index": idx,
            "prefill_preview": episode["prefill"][:180],
            "target_preview": episode["target"][:180],
            "same_score": same_score,
            "no_prefill_score": no_prefill_score,
            "wrong_score": wrong_score,
            "same_minus_no_prefill_nll": same_score["avg_nll"] - no_prefill_score["avg_nll"],
            "same_minus_wrong_nll": same_score["avg_nll"] - wrong_score["avg_nll"],
            "greedy_after_gold0": generate_after_gold_prefix(model, tokenizer, episode, cfg, device, eos_id, same_states, 0, args),
            "greedy_after_gold8": generate_after_gold_prefix(model, tokenizer, episode, cfg, device, eos_id, same_states, 8, args),
            "greedy_after_gold24": generate_after_gold_prefix(model, tokenizer, episode, cfg, device, eos_id, same_states, 24, args),
        })

    payload = {
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "summary": {
            "same_mean_nll": mean([row["same_score"]["avg_nll"] for row in rows]),
            "no_prefill_mean_nll": mean([row["no_prefill_score"]["avg_nll"] for row in rows]),
            "wrong_mean_nll": mean([row["wrong_score"]["avg_nll"] for row in rows]),
            "same_minus_no_prefill_mean_nll": mean([row["same_minus_no_prefill_nll"] for row in rows]),
            "same_minus_wrong_mean_nll": mean([row["same_minus_wrong_nll"] for row in rows]),
            "rank_by_position": bucket_rank_summary(rows, [0, 1, 2, 4, 8, 16, 32, 64]),
        },
        "rows": rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "out_file": str(args.out_file)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
