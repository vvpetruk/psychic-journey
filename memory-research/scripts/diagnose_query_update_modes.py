from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from probe_article_memory import (  # noqa: E402
    build_readonly_prefilled_model,
    build_readonly_reset_model,
    clone_states,
    encode_article,
    load_article_span,
    score_target,
)
from sample_wikitext_checkpoint import build_model, load_checkpoint, prefill_memory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen vs dynamic query memory updates.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wikitext-103/validation.jsonl"),
    )
    parser.add_argument("--article-index", type=int, default=0)
    parser.add_argument("--wrong-article-index", type=int, default=1000)
    parser.add_argument("--prefill-batch-tokens", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_probe(article: str) -> tuple[str, str]:
    words = " ".join(article.split()).split()
    return " ".join(words[:18]), " " + " ".join(words[18:24])


def build_dynamic_prefilled(
    checkpoint: dict,
    tokenizer: AutoTokenizer,
    device: torch.device,
    article_ids: list[int] | None,
    batch_tokens: int,
) -> tuple[torch.nn.Module, list | None]:
    model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    states = None
    if article_ids:
        states = prefill_memory(model, article_ids, batch_tokens, device)
    return model, states


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    cfg = checkpoint.get("config", {})
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer_name", "gpt2"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    article = load_article_span(args.dataset, args.article_index, 1)
    wrong_article = load_article_span(args.dataset, args.wrong_article_index, 1)
    article_ids, _ = encode_article(tokenizer, article, token_limit=0)
    wrong_ids, _ = encode_article(tokenizer, wrong_article, token_limit=0)
    prompt, target = choose_probe(article)

    memory_decay = float(cfg.get("memory_decay", 0.0))
    memory_lr = float(cfg.get("memory_lr", 0.1))
    memory_momentum = float(cfg.get("memory_momentum", 0.9))
    vocab_size = int(tokenizer.vocab_size)

    frozen_reset = build_readonly_reset_model(
        deepcopy(checkpoint), vocab_size, device, memory_decay, memory_lr, memory_momentum
    )
    frozen_correct, frozen_correct_states, _ = build_readonly_prefilled_model(
        deepcopy(checkpoint), vocab_size, device, article_ids, args.prefill_batch_tokens,
        memory_decay, memory_lr, memory_momentum
    )
    frozen_wrong, frozen_wrong_states, _ = build_readonly_prefilled_model(
        deepcopy(checkpoint), vocab_size, device, wrong_ids, args.prefill_batch_tokens,
        memory_decay, memory_lr, memory_momentum
    )

    dynamic_reset, dynamic_reset_states = build_dynamic_prefilled(
        checkpoint, tokenizer, device, None, args.prefill_batch_tokens
    )
    dynamic_correct, dynamic_correct_states = build_dynamic_prefilled(
        checkpoint, tokenizer, device, article_ids, args.prefill_batch_tokens
    )
    dynamic_wrong, dynamic_wrong_states = build_dynamic_prefilled(
        checkpoint, tokenizer, device, wrong_ids, args.prefill_batch_tokens
    )

    cases = [
        ("frozen_reset", frozen_reset, None),
        ("frozen_correct", frozen_correct, frozen_correct_states),
        ("frozen_wrong", frozen_wrong, frozen_wrong_states),
        ("dynamic_reset", dynamic_reset, dynamic_reset_states),
        ("dynamic_correct", dynamic_correct, dynamic_correct_states),
        ("dynamic_wrong", dynamic_wrong, dynamic_wrong_states),
    ]
    results = []
    for name, model, states in cases:
        score = score_target(model, tokenizer, prompt, target, clone_states(states), device)
        results.append({
            "case": name,
            "avg_nll": score["avg_nll"],
            "ppl": score["ppl"],
            "tokens": [
                {"token": item["token"], "nll": item["nll"], "rank": item["rank"]}
                for item in score["tokens"]
            ],
        })

    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "prompt": prompt,
        "target": target,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
