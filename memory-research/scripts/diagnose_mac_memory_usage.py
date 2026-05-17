from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
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
from sample_wikitext_checkpoint import build_model, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether MAC memory state affects causal scoring.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wikitext-103/validation.jsonl"),
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--article-index", type=int, default=0)
    parser.add_argument("--wrong-article-index", type=int, default=1000)
    parser.add_argument("--prefill-batch-tokens", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def iter_texts(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line) if path.suffix == ".jsonl" else {"text": line}
            text = (rec.get("text") or "").strip()
            if text:
                yield text


def build_token_stream(path: Path, tokenizer: AutoTokenizer, max_tokens: int) -> list[int]:
    ids: list[int] = []
    for text in iter_texts(path):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.extend(encoded)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
        if len(ids) >= max_tokens:
            break
    return ids[:max_tokens]


def chunked_loss(
    model: torch.nn.Module,
    token_ids: list[int],
    seq_len: int,
    segments: int,
    device: torch.device,
    chunk_size: int | None = None,
) -> dict:
    old_chunk_size = int(model.config.chunk_size)
    if chunk_size is not None:
        model.config.chunk_size = int(chunk_size)

    losses = []
    with torch.no_grad():
        for idx in range(segments):
            start = idx * seq_len
            segment = token_ids[start : start + seq_len + 1]
            if len(segment) < seq_len + 1:
                break
            x = torch.tensor([segment[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([segment[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))

    model.config.chunk_size = old_chunk_size
    avg = sum(losses) / max(1, len(losses))
    return {
        "segments": len(losses),
        "chunk_size": int(chunk_size if chunk_size is not None else old_chunk_size),
        "avg_nll": avg,
        "ppl": math.exp(min(avg, 20.0)),
    }


def sequential_loss(
    model: torch.nn.Module,
    token_ids: list[int],
    seq_len: int,
    segments: int,
    device: torch.device,
) -> dict:
    losses = []
    with torch.no_grad():
        for idx in range(segments):
            start = idx * seq_len
            segment = token_ids[start : start + seq_len + 1]
            if len(segment) < seq_len + 1:
                break
            states = None
            nll = 0.0
            count = 0
            for pos in range(seq_len):
                x = torch.tensor([[segment[pos]]], dtype=torch.long, device=device)
                logits, states = model(x, states=states)
                target = int(segment[pos + 1])
                log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
                nll += float(-log_probs[target].item())
                count += 1
            losses.append(nll / count)

    avg = sum(losses) / max(1, len(losses))
    return {
        "segments": len(losses),
        "avg_nll": avg,
        "ppl": math.exp(min(avg, 20.0)),
    }


def make_probe(article: str, prompt_words: int = 18, target_words: int = 6) -> tuple[str, str]:
    words = " ".join(article.split()).split()
    return " ".join(words[:prompt_words]), " " + " ".join(words[prompt_words : prompt_words + target_words])


def logits_for_prompt(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    states: list | None,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x, states=clone_states(states))
    return logits[0, -1].float().detach().cpu()


def logit_delta_report(
    checkpoint: dict,
    tokenizer: AutoTokenizer,
    device: torch.device,
    article: str,
    wrong_article: str,
    prompt: str,
    target: str,
    prefill_batch_tokens: int,
) -> dict:
    cfg = checkpoint.get("config", {})
    memory_decay = float(cfg.get("memory_decay", 0.0))
    memory_lr = float(cfg.get("memory_lr", 0.1))
    memory_momentum = float(cfg.get("memory_momentum", 0.9))
    vocab_size = int(tokenizer.vocab_size)

    article_ids, _ = encode_article(tokenizer, article, token_limit=0)
    wrong_ids, _ = encode_article(tokenizer, wrong_article, token_limit=0)

    reset_model = build_readonly_reset_model(
        deepcopy(checkpoint), vocab_size, device, memory_decay, memory_lr, memory_momentum
    )
    correct_model, correct_states, _ = build_readonly_prefilled_model(
        deepcopy(checkpoint), vocab_size, device, article_ids, prefill_batch_tokens, memory_decay, memory_lr, memory_momentum
    )
    wrong_model, wrong_states, _ = build_readonly_prefilled_model(
        deepcopy(checkpoint), vocab_size, device, wrong_ids, prefill_batch_tokens, memory_decay, memory_lr, memory_momentum
    )

    reset_logits = logits_for_prompt(reset_model, tokenizer, prompt, None, device)
    correct_logits = logits_for_prompt(correct_model, tokenizer, prompt, correct_states, device)
    wrong_logits = logits_for_prompt(wrong_model, tokenizer, prompt, wrong_states, device)

    correct_delta = correct_logits - reset_logits
    wrong_delta = wrong_logits - reset_logits

    reset_score = score_target(reset_model, tokenizer, prompt, target, None, device)
    correct_score = score_target(correct_model, tokenizer, prompt, target, clone_states(correct_states), device)
    wrong_score = score_target(wrong_model, tokenizer, prompt, target, clone_states(wrong_states), device)

    return {
        "prompt": prompt,
        "target": target,
        "prefill_tokens": len(article_ids),
        "wrong_prefill_tokens": len(wrong_ids),
        "reset_avg_nll": reset_score["avg_nll"],
        "correct_avg_nll": correct_score["avg_nll"],
        "wrong_avg_nll": wrong_score["avg_nll"],
        "correct_minus_reset_avg_nll": correct_score["avg_nll"] - reset_score["avg_nll"],
        "correct_minus_wrong_avg_nll": correct_score["avg_nll"] - wrong_score["avg_nll"],
        "correct_logit_delta_rms": float(torch.sqrt(torch.mean(correct_delta * correct_delta)).item()),
        "wrong_logit_delta_rms": float(torch.sqrt(torch.mean(wrong_delta * wrong_delta)).item()),
        "correct_logit_delta_max_abs": float(correct_delta.abs().max().item()),
        "wrong_logit_delta_max_abs": float(wrong_delta.abs().max().item()),
        "correct_wrong_logit_delta_rms": float(
            torch.sqrt(torch.mean((correct_logits - wrong_logits) ** 2)).item()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    cfg = checkpoint.get("config", {})
    tokenizer_name = cfg.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = build_token_stream(args.dataset, tokenizer, args.segments * args.seq_len + 1)
    model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    model.eval()

    article = load_article_span(args.dataset, args.article_index, 1)
    wrong_article = load_article_span(args.dataset, args.wrong_article_index, 1)
    prompt, target = make_probe(article)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "device": str(device),
        "configured_chunk_size": int(model.config.chunk_size),
        "seq_len": args.seq_len,
        "segments": args.segments,
        "loss_modes": {
            "chunked_configured": chunked_loss(model, token_ids, args.seq_len, args.segments, device),
            "chunked_force_chunk1": chunked_loss(model, token_ids, args.seq_len, args.segments, device, chunk_size=1),
            "sequential_token_by_token": sequential_loss(model, token_ids, args.seq_len, args.segments, device),
        },
        "article_probe": logit_delta_report(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            device=device,
            article=article,
            wrong_article=wrong_article,
            prompt=prompt,
            target=target,
            prefill_batch_tokens=args.prefill_batch_tokens,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
