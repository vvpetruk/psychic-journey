from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any

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
from sample_wikitext_checkpoint import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MAC retrieval/readout path.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/wikitext-103/validation.jsonl"),
    )
    parser.add_argument("--article-index", type=int, default=0)
    parser.add_argument("--wrong-article-index", type=int, default=1000)
    parser.add_argument("--prefill-batch-tokens", type=int, default=1)
    parser.add_argument("--prompt-words", type=int, default=18)
    parser.add_argument("--target-words", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_probe(article: str, prompt_words: int, target_words: int) -> tuple[str, str]:
    words = " ".join(article.split()).split()
    if len(words) < prompt_words + target_words:
        raise RuntimeError("Article too short for probe")
    prompt = " ".join(words[:prompt_words])
    target = " " + " ".join(words[prompt_words : prompt_words + target_words])
    return prompt, target


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float()
    flat = x.reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(flat * flat)).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "positive_frac": float((flat > 0).float().mean().item()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = af.norm() * bf.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float(torch.dot(af, bf).div(denom).item())


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for item in traces:
        by_layer.setdefault(int(item["layer"]), []).append(item)

    layers = []
    for layer, items in sorted(by_layer.items()):
        summary = {"layer": layer, "calls": len(items)}
        numeric_keys = [
            "pre_mem_rms",
            "pre_mem_mean",
            "pre_mem_std",
            "y_rms",
            "readout_rms",
            "readout_mean",
            "readout_std",
            "readout_positive_frac",
            "output_rms",
            "output_delta_from_y_rms",
            "output_y_cosine",
        ]
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if key in item]
            if vals:
                summary[f"{key}_mean"] = sum(vals) / len(vals)
                summary[f"{key}_last"] = vals[-1]
        layers.append(summary)
    return {"layers": layers}


def install_mac_readout_probe(
    model: torch.nn.Module,
    mode: str,
    affected_layers: set[int] | None = None,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    for layer_idx, block in enumerate(model.blocks):
        if affected_layers is not None and layer_idx not in affected_layers:
            continue

        def patched_forward(self, x, state=None, _layer_idx=layer_idx):
            batch_size, seq_len, _ = x.shape
            if state is None:
                state = self.memory.init_state(batch_size, x.device)

            use_pre_context = mode not in {"no_pre_context", "no_memory_all"}
            use_post_readout = mode not in {"no_post_readout", "no_memory_all"}

            pre_mem = None
            memory_tokens = None
            if use_pre_context:
                pre_mem = self.memory.retrieve(x, state)
                memory_tokens = self.norm_mem(pre_mem)

            persistent = self.persistent(batch_size)
            normed = self.norm1(x)
            attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
            y_t = x + self.dropout(attn_out)

            if mode == "no_memory_all":
                new_state = state.detach()
                readout = None
                output = y_t
            else:
                _, new_state = self.memory(y_t, state=state)
                readout = self.memory.retrieve(y_t, new_state)
                if mode in {"full", "no_pre_context"}:
                    output = y_t * readout
                elif mode == "no_post_readout":
                    output = y_t
                elif mode == "additive":
                    output = y_t + readout
                elif mode == "residual_0p1":
                    output = y_t + 0.1 * readout
                elif mode == "gate_1_plus":
                    output = y_t * (1.0 + readout)
                else:
                    raise ValueError(f"Unknown readout mode: {mode}")

            item: dict[str, Any] = {"layer": _layer_idx, "mode": mode, "seq_len": int(seq_len)}
            if pre_mem is not None:
                stats = tensor_stats(pre_mem[:, -1])
                item.update({
                    "pre_mem_mean": stats["mean"],
                    "pre_mem_std": stats["std"],
                    "pre_mem_rms": stats["rms"],
                })
            y_stats = tensor_stats(y_t[:, -1])
            item.update({"y_rms": y_stats["rms"]})
            if readout is not None:
                r_stats = tensor_stats(readout[:, -1])
                item.update({
                    "readout_mean": r_stats["mean"],
                    "readout_std": r_stats["std"],
                    "readout_rms": r_stats["rms"],
                    "readout_positive_frac": r_stats["positive_frac"],
                })
            out_stats = tensor_stats(output[:, -1])
            item.update({
                "output_rms": out_stats["rms"],
                "output_delta_from_y_rms": tensor_stats(output[:, -1] - y_t[:, -1])["rms"],
                "output_y_cosine": cosine(output[:, -1], y_t[:, -1]),
            })
            traces.append(item)

            normed = self.norm2(output)
            ffn_out = self.ffn(normed)
            output = output + self.dropout(ffn_out)
            return output, new_state

        block.forward = MethodType(patched_forward, block)

    return traces


def build_condition_model(
    checkpoint: dict,
    tokenizer: AutoTokenizer,
    device: torch.device,
    condition: str,
    article_ids: list[int],
    wrong_article_ids: list[int],
    prefill_batch_tokens: int,
) -> tuple[torch.nn.Module, list | None]:
    cfg = checkpoint.get("config", {})
    vocab_size = int(tokenizer.vocab_size)
    memory_decay = float(cfg.get("memory_decay", 0.0))
    memory_lr = float(cfg.get("memory_lr", 0.1))
    memory_momentum = float(cfg.get("memory_momentum", 0.9))

    if condition == "reset":
        return (
            build_readonly_reset_model(
                deepcopy(checkpoint),
                vocab_size,
                device,
                memory_decay,
                memory_lr,
                memory_momentum,
            ),
            None,
        )
    if condition == "correct":
        model, states, _ = build_readonly_prefilled_model(
            deepcopy(checkpoint),
            vocab_size,
            device,
            article_ids,
            prefill_batch_tokens,
            memory_decay,
            memory_lr,
            memory_momentum,
        )
        return model, states
    if condition == "wrong":
        model, states, _ = build_readonly_prefilled_model(
            deepcopy(checkpoint),
            vocab_size,
            device,
            wrong_article_ids,
            prefill_batch_tokens,
            memory_decay,
            memory_lr,
            memory_momentum,
        )
        return model, states
    raise ValueError(f"Unknown condition: {condition}")


def run_case(
    checkpoint: dict,
    tokenizer: AutoTokenizer,
    device: torch.device,
    condition: str,
    mode: str,
    prompt: str,
    target: str,
    article_ids: list[int],
    wrong_article_ids: list[int],
    prefill_batch_tokens: int,
    affected_layers: set[int] | None = None,
) -> dict[str, Any]:
    model, states = build_condition_model(
        checkpoint,
        tokenizer,
        device,
        condition,
        article_ids,
        wrong_article_ids,
        prefill_batch_tokens,
    )
    traces = install_mac_readout_probe(model, mode=mode, affected_layers=affected_layers)
    score = score_target(model, tokenizer, prompt, target, clone_states(states), device)
    return {
        "condition": condition,
        "mode": mode,
        "affected_layers": sorted(affected_layers) if affected_layers is not None else "all",
        "avg_nll": score["avg_nll"],
        "ppl": score["ppl"],
        "tokens": [
            {
                "token": token["token"],
                "nll": token["nll"],
                "rank": token["rank"],
            }
            for token in score["tokens"]
        ],
        "trace_summary": summarize_traces(traces),
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

    article = load_article_span(args.dataset, args.article_index, 1)
    wrong_article = load_article_span(args.dataset, args.wrong_article_index, 1)
    article_ids, _ = encode_article(tokenizer, article, token_limit=0)
    wrong_article_ids, _ = encode_article(tokenizer, wrong_article, token_limit=0)
    prompt, target = choose_probe(article, args.prompt_words, args.target_words)

    cases = [
        ("reset", "full", None),
        ("correct", "full", None),
        ("wrong", "full", None),
        ("correct", "no_pre_context", None),
        ("correct", "no_post_readout", None),
        ("correct", "no_memory_all", None),
        ("correct", "additive", None),
        ("correct", "residual_0p1", None),
        ("correct", "gate_1_plus", None),
        ("correct", "no_post_readout", {0}),
        ("correct", "no_post_readout", {1}),
        ("correct", "no_pre_context", {0}),
        ("correct", "no_pre_context", {1}),
    ]

    results = [
        run_case(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            device=device,
            condition=condition,
            mode=mode,
            prompt=prompt,
            target=target,
            article_ids=article_ids,
            wrong_article_ids=wrong_article_ids,
            prefill_batch_tokens=args.prefill_batch_tokens,
            affected_layers=affected_layers,
        )
        for condition, mode, affected_layers in cases
    ]

    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_val": checkpoint.get("best_val"),
        "device": str(device),
        "prompt": prompt,
        "target": target,
        "prefill_tokens": len(article_ids),
        "wrong_prefill_tokens": len(wrong_article_ids),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
