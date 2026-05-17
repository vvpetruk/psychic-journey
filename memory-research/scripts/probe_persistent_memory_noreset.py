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
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_article_memory import apply_memory_controls, collect_layer_update_stats  # noqa: E402
from sample_wikitext_checkpoint import build_model, load_checkpoint, sample_next_id  # noqa: E402
from train_lima_memory_task import frozen_memory_updates  # noqa: E402


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
        description=(
            "Probe MAC memory without reset baselines. A single fixed-size memory "
            "state is carried through warmup, article prefill, scoring, and generation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wikitext-103/validation.jsonl"),
    )
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--start-article-index", type=int, default=0)
    parser.add_argument("--warmup-articles", type=int, default=8)
    parser.add_argument("--probe-count", type=int, default=12)
    parser.add_argument("--article-span", type=int, default=1)
    parser.add_argument("--prefill-token-limit", type=int, default=1024)
    parser.add_argument("--prefill-batch-tokens", type=int, default=32)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--target-words", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def iter_texts(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                record = json.loads(line)
                text = (record.get("text") or "").strip()
            else:
                text = line
            if text:
                yield text


def load_article_span(texts: list[str], index: int, span: int) -> str:
    if span < 1:
        raise ValueError(f"article span must be >= 1, got {span}")
    stop = index + span
    if index < 0 or stop > len(texts):
        raise IndexError(f"requested article span {index}:{stop}, but dataset has {len(texts)} records")
    return "\n\n".join(texts[index:stop])


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def choose_auto_cloze(article: str, prompt_words: int, target_words: int) -> tuple[str, str]:
    article = normalize_space(article)
    sentences = re.split(r"(?<=[.!?])\s+", article)
    needed = prompt_words + target_words
    for sentence in sentences:
        words = sentence.split()
        if len(words) >= needed and any(ch.isalpha() for ch in sentence):
            prompt = " ".join(words[:prompt_words])
            target = " " + " ".join(words[prompt_words:needed])
            return prompt, target

    words = article.split()
    if len(words) < needed:
        raise RuntimeError(f"article is too short for cloze: need {needed} words, got {len(words)}")
    return " ".join(words[:prompt_words]), " " + " ".join(words[prompt_words:needed])


def encode_article(tokenizer: AutoTokenizer, text: str, token_limit: int) -> tuple[list[int], int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    raw_count = len(token_ids)
    if token_limit > 0:
        token_ids = token_ids[:token_limit]
    return token_ids, raw_count


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


def detach_states(states):
    if states is None:
        return None
    return [state.detach() if state is not None else None for state in states]


def state_norm_summary(states) -> dict[str, Any]:
    if states is None:
        return {"layers": 0, "weight_norm_sum": 0.0, "momentum_norm_sum": 0.0, "shapes": []}

    weight_norm = 0.0
    momentum_norm = 0.0
    shapes = []
    for layer_index, state in enumerate(states):
        if state is None:
            continue
        layer_weight_norm = sum(float(weight.detach().float().norm().item()) for weight in state.weights)
        layer_momentum_norm = sum(float(momentum.detach().float().norm().item()) for momentum in state.momentum)
        weight_norm += layer_weight_norm
        momentum_norm += layer_momentum_norm
        shapes.append(
            {
                "layer": layer_index,
                "weight_shapes": [list(weight.shape) for weight in state.weights],
                "momentum_shapes": [list(momentum.shape) for momentum in state.momentum],
                "weight_norm_sum": layer_weight_norm,
                "momentum_norm_sum": layer_momentum_norm,
            }
        )

    return {
        "layers": len(shapes),
        "weight_norm_sum": weight_norm,
        "momentum_norm_sum": momentum_norm,
        "shapes": shapes,
    }


def summarize_prefill_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {
            "chunks": 0,
            "tokens": 0,
            "memory_grad_norm_sum": 0.0,
            "weight_delta_norm_sum": 0.0,
            "momentum_delta_norm_sum": 0.0,
            "nonzero_update_chunks": 0,
        }

    totals = {
        "memory_grad_norm_sum": 0.0,
        "weight_delta_norm_sum": 0.0,
        "momentum_delta_norm_sum": 0.0,
    }
    nonzero_chunks = 0
    layer_totals: dict[int, dict[str, float]] = {}
    layer_counts: dict[int, int] = {}
    for chunk in trace:
        chunk_weight_delta = 0.0
        chunk_momentum_delta = 0.0
        for layer in chunk["layers"]:
            layer_index = int(layer["layer"])
            layer_total = layer_totals.setdefault(
                layer_index,
                {
                    "memory_grad_norm_sum": 0.0,
                    "weight_delta_norm_sum": 0.0,
                    "momentum_delta_norm_sum": 0.0,
                    "effective_decay": 0.0,
                    "effective_lr": 0.0,
                    "effective_momentum": 0.0,
                },
            )
            layer_counts[layer_index] = layer_counts.get(layer_index, 0) + 1
            for key in ("memory_grad_norm_sum", "weight_delta_norm_sum", "momentum_delta_norm_sum"):
                value = float(layer.get(key, 0.0))
                totals[key] += value
                layer_total[key] += value
            layer_total["effective_decay"] += float(layer.get("effective_decay", 0.0))
            layer_total["effective_lr"] += float(layer.get("effective_lr", 0.0))
            layer_total["effective_momentum"] += float(layer.get("effective_momentum", 0.0))
            chunk_weight_delta += float(layer.get("weight_delta_norm_sum", 0.0))
            chunk_momentum_delta += float(layer.get("momentum_delta_norm_sum", 0.0))
        if chunk_weight_delta > 0.0 or chunk_momentum_delta > 0.0:
            nonzero_chunks += 1

    layers = []
    for layer_index in sorted(layer_totals):
        count = layer_counts[layer_index]
        total = layer_totals[layer_index]
        layers.append(
            {
                "layer": layer_index,
                "chunks": count,
                "memory_grad_norm_sum": total["memory_grad_norm_sum"],
                "weight_delta_norm_sum": total["weight_delta_norm_sum"],
                "momentum_delta_norm_sum": total["momentum_delta_norm_sum"],
                "mean_effective_decay": total["effective_decay"] / count,
                "mean_effective_lr": total["effective_lr"] / count,
                "mean_effective_momentum": total["effective_momentum"] / count,
            }
        )

    return {
        "chunks": len(trace),
        "tokens": sum(int(chunk["tokens"]) for chunk in trace),
        "nonzero_update_chunks": nonzero_chunks,
        **totals,
        "layers": layers,
        "chunk_update_sample": {
            "first": trace[:2],
            "last": trace[-2:] if len(trace) > 2 else [],
        },
    }


def stream_tokens_into_memory(
    model: torch.nn.Module,
    states,
    token_ids: list[int],
    batch_tokens: int,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]]]:
    if not token_ids:
        return states, []
    chunk_size = int(getattr(model.config, "chunk_size", batch_tokens))
    step_tokens = max(1, min(batch_tokens, chunk_size))
    trace = []
    for chunk_index, start in enumerate(range(0, len(token_ids), step_tokens)):
        chunk = token_ids[start : start + step_tokens]
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            _, states = model(x, states=states)
        states = detach_states(states)
        trace.append(
            {
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": start + len(chunk),
                "tokens": len(chunk),
                "layers": collect_layer_update_stats(model),
            }
        )
    return states, trace


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    token_logit = logits[token_id]
    return int((logits > token_logit).sum().item()) + 1


def top_tokens(tokenizer: AutoTokenizer, logits: torch.Tensor, k: int = 8) -> list[dict[str, Any]]:
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "id": int(token_id),
                "text": tokenizer.decode([int(token_id)]),
                "prob": float(prob),
            }
        )
    return rows


def score_target_readonly(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    target: str,
    states,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [int(tokenizer.eos_token_id)]
    if not target_ids:
        raise RuntimeError(f"empty target after tokenization: {target!r}")

    answer_input = prompt_ids + target_ids[:-1]
    x = torch.tensor([answer_input], dtype=torch.long, device=device)
    labels = torch.tensor(target_ids, dtype=torch.long, device=device)
    with torch.no_grad(), frozen_memory_updates(model):
        logits, _ = model(x, states=states)

    start = len(prompt_ids) - 1
    target_logits = logits[0, start : start + len(target_ids)].float()
    nll = float(F.cross_entropy(target_logits, labels, reduction="sum").item())
    avg_nll = nll / len(target_ids)
    probs = F.softmax(target_logits[0], dim=-1)
    first_id = int(target_ids[0])
    return {
        "target_tokens": len(target_ids),
        "nll": nll,
        "avg_nll": avg_nll,
        "ppl": math.exp(min(avg_nll, 20.0)),
        "first_token": {
            "id": first_id,
            "text": tokenizer.decode([first_id]),
            "prob": float(probs[first_id].item()),
            "rank": rank_of(target_logits[0], first_id),
            "top": top_tokens(tokenizer, target_logits[0]),
        },
    }


def sample_answer_readonly(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    states,
    device: torch.device,
    args: argparse.Namespace,
    *,
    greedy: bool,
    force_min_tokens: bool,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [int(tokenizer.eos_token_id)]
    eos_id = int(tokenizer.eos_token_id)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    with torch.no_grad(), frozen_memory_updates(model):
        logits, states = model(x, states=states)
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}

    before_nll = [row["before"]["avg_nll"] for row in rows]
    after_nll = [row["after"]["avg_nll"] for row in rows]
    rank_before = [row["before"]["first_token"]["rank"] for row in rows]
    rank_after = [row["after"]["first_token"]["rank"] for row in rows]
    deltas = [after - before for before, after in zip(before_nll, after_nll, strict=True)]
    generation = [row["sampled_after_min_tokens"] for row in rows]

    return {
        "samples": len(rows),
        "after_beats_before_avg_nll": sum(1 for delta in deltas if delta < 0.0),
        "mean_before_avg_nll": sum(before_nll) / len(before_nll),
        "mean_after_avg_nll": sum(after_nll) / len(after_nll),
        "mean_after_minus_before_avg_nll": sum(deltas) / len(deltas),
        "median_after_minus_before_avg_nll": sorted(deltas)[len(deltas) // 2],
        "mean_before_ppl": sum(row["before"]["ppl"] for row in rows) / len(rows),
        "mean_after_ppl": sum(row["after"]["ppl"] for row in rows) / len(rows),
        "mean_first_rank_before": sum(rank_before) / len(rank_before),
        "mean_first_rank_after": sum(rank_after) / len(rank_after),
        "first_rank_after_beats_before": sum(
            1 for before, after in zip(rank_before, rank_after, strict=True) if after < before
        ),
        "nonempty_sampled_generations": sum(1 for text in generation if text.strip()),
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
        raise RuntimeError(f"tokenizer {tokenizer_name} must have eos_token_id")
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

    texts = list(iter_texts(args.dataset))
    if not texts:
        raise RuntimeError(f"no texts found in {args.dataset}")

    states = None
    cursor = args.start_article_index
    warmup_trace = []
    for offset in range(args.warmup_articles):
        article = load_article_span(texts, cursor + offset, args.article_span)
        token_ids, _raw_count = encode_article(tokenizer, article, args.prefill_token_limit)
        states, trace = stream_tokens_into_memory(model, states, token_ids, args.prefill_batch_tokens, device)
        warmup_trace.extend(trace)

    cursor += args.warmup_articles
    rows = []
    skipped = []
    article_index = cursor
    while len(rows) < args.probe_count:
        article = load_article_span(texts, article_index, args.article_span)
        try:
            prompt, target = choose_auto_cloze(article, args.prompt_words, args.target_words)
        except RuntimeError as exc:
            skipped.append({"article_index": article_index, "reason": str(exc)})
            article_index += args.article_span
            continue

        article_ids, raw_tokens = encode_article(tokenizer, article, args.prefill_token_limit)

        state_before = state_norm_summary(states)
        before = score_target_readonly(model, tokenizer, prompt, target, states, device)

        states, prefill_trace = stream_tokens_into_memory(
            model,
            states,
            article_ids,
            args.prefill_batch_tokens,
            device,
        )
        state_after = state_norm_summary(states)
        after = score_target_readonly(model, tokenizer, prompt, target, states, device)

        rows.append(
            {
                "article_index": article_index,
                "article_preview": normalize_space(article)[:400],
                "prompt": prompt,
                "target": target,
                "article_tokens": len(article_ids),
                "article_raw_tokens": raw_tokens,
                "article_truncated": len(article_ids) < raw_tokens,
                "state_before": state_before,
                "state_after": state_after,
                "prefill_trace": summarize_prefill_trace(prefill_trace),
                "before": before,
                "after": after,
                "deltas": {
                    "after_minus_before_avg_nll": after["avg_nll"] - before["avg_nll"],
                    "after_minus_before_first_rank": (
                        after["first_token"]["rank"] - before["first_token"]["rank"]
                    ),
                },
                "greedy_after": sample_answer_readonly(
                    model,
                    tokenizer,
                    prompt,
                    states,
                    device,
                    args,
                    greedy=True,
                    force_min_tokens=False,
                ),
                "sampled_after_min_tokens": sample_answer_readonly(
                    model,
                    tokenizer,
                    prompt,
                    states,
                    device,
                    args,
                    greedy=False,
                    force_min_tokens=True,
                ),
            }
        )
        article_index += args.article_span

    runtime_memory_keys = [
        key for key in checkpoint.keys()
        if "memory_state" in str(key).lower() or "runtime_state" in str(key).lower()
    ]
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "checkpoint_runtime_memory_state_keys": runtime_memory_keys,
        "checkpoint_contains_runtime_memory_state": bool(runtime_memory_keys),
        "dataset": str(args.dataset),
        "device": str(device),
        "protocol": {
            "no_reset_baseline": True,
            "single_state_carried_through_all_examples": True,
            "memory_update_during_article_prefill": True,
            "memory_update_during_query_and_generation": False,
            "state_initialization": (
                "The checkpoint supplies model weights. If it does not include runtime memory state, "
                "the first warmup forward creates one fixed-size state, then the probe never clears it."
            ),
        },
        "settings": {
            "start_article_index": args.start_article_index,
            "warmup_articles": args.warmup_articles,
            "probe_count": args.probe_count,
            "article_span": args.article_span,
            "prefill_token_limit": args.prefill_token_limit,
            "prefill_batch_tokens": args.prefill_batch_tokens,
            "prompt_words": args.prompt_words,
            "target_words": args.target_words,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "warmup": summarize_prefill_trace(warmup_trace),
        "summary": summarize_rows(rows),
        "skipped_articles": skipped,
        "samples": rows,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "out_file": str(args.out_file)}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
