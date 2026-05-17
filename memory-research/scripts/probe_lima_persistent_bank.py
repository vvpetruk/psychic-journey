from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_article_memory import apply_memory_controls  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint, sample_next_id  # noqa: E402
from train_lima_memory_task import (  # noqa: E402
    TrainConfig,
    build_episodes,
    detach_states,
    deserialize_states,
    encode_limited,
    frozen_memory_updates,
    load_records,
    merge_model_config,
    prefill_memory_graph,
    state_norm_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a persisted LIMA MAC memory bank without resets. Each instruction "
            "prefills memory by surprise, then the answer is scored/generated with "
            "memory updates frozen."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def lexical_overlap(generated: str, target: str) -> dict[str, Any]:
    generated_words = set(re.findall(r"[a-z0-9]+", generated.lower()))
    target_words = set(re.findall(r"[a-z0-9]+", target.lower()))
    if not target_words:
        return {"target_word_recall": 0.0, "jaccard": 0.0}
    intersection = generated_words & target_words
    union = generated_words | target_words
    return {
        "target_word_recall": len(intersection) / len(target_words),
        "jaccard": len(intersection) / len(union) if union else 0.0,
        "overlap_words": sorted(intersection)[:25],
    }


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def score_from_bank(
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
    first_id = int(target_ids[0])
    first_logits = target_logits[0]
    probs = F.softmax(first_logits, dim=-1)
    eos_prob = float(probs[eos_id].item())
    eos_rank = rank_of(first_logits, eos_id)
    return {
        "tokens": len(target_ids),
        "nll": nll,
        "avg_nll": avg_nll,
        "ppl": math.exp(min(avg_nll, 20.0)),
        "first_token": {
            "id": first_id,
            "text": tokenizer.decode([first_id]),
            "prob": float(probs[first_id].item()),
            "rank": rank_of(first_logits, first_id),
            "eos_prob": eos_prob,
            "eos_rank": eos_rank,
        },
    }


def generate_from_bank(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int,
    states,
    args: argparse.Namespace,
    *,
    greedy: bool,
) -> str:
    query_ids = encode_limited(tokenizer, episode.get("query") or cfg.query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    x = torch.tensor([query_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    with torch.no_grad(), frozen_memory_updates(model):
        logits, states = model(x, states=states)
        for _ in range(args.max_new_tokens):
            next_logits = logits[0, -1].float().clone()
            if len(new_ids) < args.min_new_tokens:
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
    if not rows:
        return {"samples": 0}
    nll = [row["score"]["avg_nll"] for row in rows]
    rank = [row["score"]["first_token"]["rank"] for row in rows]
    eos_rank = [row["score"]["first_token"]["eos_rank"] for row in rows]
    recall = [row["sampled_overlap"]["target_word_recall"] for row in rows]
    return {
        "samples": len(rows),
        "mean_avg_nll": sum(nll) / len(nll),
        "ppl_from_mean_nll": math.exp(min(sum(nll) / len(nll), 20.0)),
        "mean_first_token_rank": sum(rank) / len(rank),
        "median_first_token_rank": sorted(rank)[len(rank) // 2],
        "eos_top1": sum(1 for value in eos_rank if value == 1),
        "nonempty_greedy": sum(1 for row in rows if row["greedy"].strip()),
        "nonempty_sampled": sum(1 for row in rows if row["sampled"].strip()),
        "mean_sampled_target_word_recall": sum(recall) / len(recall),
    }


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
    eos_id = int(tokenizer.eos_token_id)

    model = build_model(checkpoint_for_build, int(tokenizer.vocab_size), device)
    apply_memory_controls(
        model,
        float(merged_cfg.get("memory_decay", 0.0)),
        float(merged_cfg.get("memory_lr", 0.1)),
        float(merged_cfg.get("memory_momentum", 0.9)),
    )
    model.eval()

    runtime_state = checkpoint.get("runtime_memory_state") or {}
    state_payload = runtime_state.get("train_states")
    if not state_payload:
        raise RuntimeError(f"{args.checkpoint} does not contain runtime_memory_state.train_states")
    states = deserialize_states(state_payload, model, device, batch_size=1)

    cfg = TrainConfig(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset_path,
        max_eval_examples=args.max_eval_examples,
        max_prefill_tokens=args.max_prefill_tokens,
        max_answer_tokens=args.max_answer_tokens,
        prefill_batch_tokens=args.prefill_batch_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        persistent_memory=True,
    )
    records = load_records(cfg, cfg.train_split)
    episodes = build_episodes(records, max(args.max_eval_examples, args.sample_count))
    episodes = episodes[: args.sample_count]

    rows = []
    for index, episode in enumerate(episodes):
        prefill_ids = encode_limited(tokenizer, episode["prefill"], cfg.max_prefill_tokens, keep="right")
        state_before = state_norm_summary(states)
        with torch.no_grad():
            states = prefill_memory_graph(
                model,
                prefill_ids,
                cfg.prefill_batch_tokens,
                device,
                states=states,
            )
        states = detach_states(states)
        state_after_prefill = state_norm_summary(states)
        score = score_from_bank(model, tokenizer, episode, cfg, device, eos_id, states)
        greedy = generate_from_bank(model, tokenizer, episode, cfg, device, eos_id, states, args, greedy=True)
        sampled = generate_from_bank(model, tokenizer, episode, cfg, device, eos_id, states, args, greedy=False)
        rows.append(
            {
                "index": index,
                "prefill": episode["prefill"],
                "target": episode["target"],
                "prefill_tokens": len(prefill_ids),
                "state_before_prefill": state_before,
                "state_after_prefill": state_after_prefill,
                "score": score,
                "greedy": greedy,
                "sampled": sampled,
                "greedy_overlap": lexical_overlap(greedy, episode["target"]),
                "sampled_overlap": lexical_overlap(sampled, episode["target"]),
                "exact_normalized_match": normalize_text(greedy) == normalize_text(episode["target"]),
            }
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(args.dataset_path),
        "device": str(device),
        "protocol": {
            "uses_persisted_runtime_memory_state": True,
            "memory_reset_during_probe": False,
            "memory_update_during_instruction_prefill": True,
            "memory_update_during_query_and_generation": False,
            "state_norms": state_norm_summary(states),
        },
        "summary": summarize(rows),
        "samples": rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "out_file": str(args.out_file)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
