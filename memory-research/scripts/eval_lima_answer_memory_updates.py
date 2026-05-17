from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_article_memory import apply_memory_controls, clone_states  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint  # noqa: E402
from train_lima_memory_task import (  # noqa: E402
    MODEL_CONFIG_KEYS,
    build_episodes,
    encode_limited,
    frozen_memory_updates,
    merge_model_config,
    prefill_memory_graph,
    read_json_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LIMA answer scoring with frozen vs live test-time memory updates."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=64)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--prefill-batch-tokens", type=int, default=None)
    parser.add_argument("--query-prompt", type=str, default=None)
    parser.add_argument("--wrong-offset", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize_scores(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"examples": len(rows)}
    for key in ("reset", "same_prefill", "wrong_prefill"):
        values = [row[mode][key]["avg_nll"] for row in rows]
        summary[key] = {
            "mean_avg_nll": mean(values),
            "ppl_from_mean_nll": math.exp(min(mean(values), 20.0)),
        }

    delta_specs = {
        "same_minus_reset_avg_nll": ("same_prefill", "reset"),
        "wrong_minus_reset_avg_nll": ("wrong_prefill", "reset"),
        "same_minus_wrong_avg_nll": ("same_prefill", "wrong_prefill"),
    }
    for name, (left, right) in delta_specs.items():
        values = [
            row[mode][left]["avg_nll"] - row[mode][right]["avg_nll"]
            for row in rows
        ]
        summary[name] = {
            "mean": mean(values),
            "wins_negative": sum(1 for value in values if value < 0.0),
            "total": len(values),
        }
    return summary


def score_answer(
    model: torch.nn.Module,
    query_ids: list[int],
    target_ids: list[int],
    states,
    device: torch.device,
    freeze_answer_memory: bool,
) -> dict[str, Any]:
    answer_input = query_ids + target_ids[:-1]
    x = torch.tensor([answer_input], dtype=torch.long, device=device)
    labels = torch.tensor(target_ids, dtype=torch.long, device=device)

    context = frozen_memory_updates(model) if freeze_answer_memory else nullcontext()
    with torch.no_grad(), context:
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


def prefill_states(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    max_prefill_tokens: int,
    prefill_batch_tokens: int,
    device: torch.device,
):
    token_ids = encode_limited(tokenizer, text, max_prefill_tokens, keep="right")
    with torch.no_grad():
        return prefill_memory_graph(model, token_ids, prefill_batch_tokens, device)


def score_episode_pair(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    episode: dict[str, str],
    wrong_episode: dict[str, str],
    max_prefill_tokens: int,
    max_answer_tokens: int,
    prefill_batch_tokens: int,
    query_prompt: str,
    device: torch.device,
    eos_id: int,
) -> dict[str, Any]:
    query_ids = encode_limited(tokenizer, episode.get("query") or query_prompt, 32, keep="left")
    if not query_ids:
        query_ids = [eos_id]
    target_ids = encode_limited(tokenizer, episode["target"], max_answer_tokens - 1, keep="left")
    target_ids.append(eos_id)

    same_states = prefill_states(
        model,
        tokenizer,
        episode["prefill"],
        max_prefill_tokens,
        prefill_batch_tokens,
        device,
    )
    wrong_states = prefill_states(
        model,
        tokenizer,
        wrong_episode["prefill"],
        max_prefill_tokens,
        prefill_batch_tokens,
        device,
    )

    return {
        "target_tokens": len(target_ids),
        "frozen_answer_memory": {
            "reset": score_answer(model, query_ids, target_ids, None, device, True),
            "same_prefill": score_answer(model, query_ids, target_ids, same_states, device, True),
            "wrong_prefill": score_answer(model, query_ids, target_ids, wrong_states, device, True),
        },
        "live_answer_memory": {
            "reset": score_answer(model, query_ids, target_ids, None, device, False),
            "same_prefill": score_answer(model, query_ids, target_ids, same_states, device, False),
            "wrong_prefill": score_answer(model, query_ids, target_ids, wrong_states, device, False),
        },
    }


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    ckpt_cfg = merge_model_config(checkpoint, torch.device("cpu"))
    train_cfg = checkpoint.get("config") or {}
    tokenizer_name = ckpt_cfg.get("tokenizer_name") or train_cfg.get("tokenizer") or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name} must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = args.dataset_path or Path(train_cfg.get("dataset_path") or "")
    if not dataset_path:
        raise RuntimeError("--dataset-path is required when checkpoint config does not include it")

    max_eval_examples = int(args.max_eval_examples)
    max_prefill_tokens = int(args.max_prefill_tokens or train_cfg.get("max_prefill_tokens", 192))
    max_answer_tokens = int(args.max_answer_tokens or train_cfg.get("max_answer_tokens", 128))
    prefill_batch_tokens = int(args.prefill_batch_tokens or train_cfg.get("prefill_batch_tokens", 32))
    query_prompt = args.query_prompt or train_cfg.get("query_prompt") or "Assistant:"

    records = read_json_records(dataset_path)
    episodes = build_episodes(records, max_eval_examples)
    if len(episodes) < 2:
        raise RuntimeError("Need at least two eval episodes")

    model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    apply_memory_controls(
        model,
        float(ckpt_cfg.get("memory_decay", 0.0)),
        float(ckpt_cfg.get("memory_lr", 0.1)),
        float(ckpt_cfg.get("memory_momentum", 0.9)),
    )
    model.eval()

    rows = []
    wrong_offset = max(1, int(args.wrong_offset))
    eos_id = int(tokenizer.eos_token_id)
    for idx, episode in enumerate(episodes):
        wrong_episode = episodes[(idx + wrong_offset) % len(episodes)]
        rows.append(
            score_episode_pair(
                model,
                tokenizer,
                episode,
                wrong_episode,
                max_prefill_tokens,
                max_answer_tokens,
                prefill_batch_tokens,
                query_prompt,
                device,
                eos_id,
            )
        )

    frozen = summarize_scores(rows, "frozen_answer_memory")
    live = summarize_scores(rows, "live_answer_memory")
    live_minus_frozen_same = [
        row["live_answer_memory"]["same_prefill"]["avg_nll"]
        - row["frozen_answer_memory"]["same_prefill"]["avg_nll"]
        for row in rows
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_eval": checkpoint.get("best_eval"),
        "device": str(device),
        "dataset_path": str(dataset_path),
        "examples": len(rows),
        "max_prefill_tokens": max_prefill_tokens,
        "max_answer_tokens": max_answer_tokens,
        "prefill_batch_tokens": prefill_batch_tokens,
        "model_config": {
            key: ckpt_cfg[key]
            for key in MODEL_CONFIG_KEYS
            if key in ckpt_cfg
        },
        "frozen_answer_memory": frozen,
        "live_answer_memory": live,
        "live_minus_frozen_same_prefill_avg_nll": {
            "mean": mean(live_minus_frozen_same),
            "wins_negative": sum(1 for value in live_minus_frozen_same if value < 0.0),
            "total": len(live_minus_frozen_same),
        },
    }

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
