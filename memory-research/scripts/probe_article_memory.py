from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_wikitext_checkpoint import build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prefill MAC memory with one article and score article-specific recall prompts.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best.pt'),
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/processed/wikitext-103/validation.jsonl'),
    )
    parser.add_argument('--article-index', type=int, default=0)
    parser.add_argument(
        '--article-span',
        type=int,
        default=1,
        help='Number of consecutive dataset records to concatenate for the article prefill.',
    )
    parser.add_argument('--wrong-article-index', type=int, default=1)
    parser.add_argument(
        '--wrong-article-span',
        type=int,
        default=None,
        help='Number of consecutive dataset records for the wrong-article control. Defaults to --article-span.',
    )
    parser.add_argument('--prompt', action='append', default=None)
    parser.add_argument('--target', action='append', default=None)
    parser.add_argument('--prompt-words', type=int, default=18)
    parser.add_argument('--target-words', type=int, default=6)
    parser.add_argument(
        '--prefill-token-limit',
        type=int,
        default=2048,
        help='Maximum prefill tokens per article. Use 0 to prefill the full article span.',
    )
    parser.add_argument('--prefill-batch-tokens', type=int, default=128)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def iter_texts(path: Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == '.jsonl':
                rec = json.loads(line)
                text = (rec.get('text') or '').strip()
            else:
                text = line
            if text:
                yield text


def load_article_span(path: Path, index: int, span: int) -> str:
    if span < 1:
        raise ValueError(f'Article span must be >= 1, got {span}')

    texts: list[str] = []
    stop_index = index + span
    for item_index, text in enumerate(iter_texts(path)):
        if item_index >= stop_index:
            break
        if item_index >= index:
            texts.append(text)

    if len(texts) != span:
        raise IndexError(
            f'Article span {index}:{stop_index} requested {span} records, found {len(texts)} in {path}'
        )
    return '\n\n'.join(texts)


def normalize_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def choose_auto_cloze(article: str, prompt_words: int, target_words: int) -> tuple[str, str]:
    article = normalize_space(article)
    sentences = re.split(r'(?<=[.!?])\s+', article)
    needed = prompt_words + target_words
    for sentence in sentences:
        words = sentence.split()
        if len(words) >= needed and any(ch.isalpha() for ch in sentence):
            prompt = ' '.join(words[:prompt_words])
            target = ' ' + ' '.join(words[prompt_words:needed])
            return prompt, target

    words = article.split()
    if len(words) < needed:
        raise RuntimeError(
            f'Article is too short for auto cloze: need {needed} words, got {len(words)}'
        )
    return ' '.join(words[:prompt_words]), ' ' + ' '.join(words[prompt_words:needed])


def build_prompts(args: argparse.Namespace, article: str) -> list[dict[str, str]]:
    prompts = args.prompt or []
    targets = args.target or []
    if prompts or targets:
        if len(prompts) != len(targets):
            raise ValueError('--prompt and --target must be provided the same number of times')
        return [
            {'prompt': prompt, 'target': target, 'source': 'explicit'}
            for prompt, target in zip(prompts, targets, strict=True)
        ]

    prompt, target = choose_auto_cloze(article, args.prompt_words, args.target_words)
    return [{'prompt': prompt, 'target': target, 'source': 'auto_cloze'}]


def encode_article(tokenizer: AutoTokenizer, text: str, token_limit: int) -> tuple[list[int], int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)
    raw_token_count = len(token_ids)
    if token_limit > 0:
        token_ids = token_ids[:token_limit]
    return token_ids, raw_token_count


def disable_memory_updates(model: torch.nn.Module) -> None:
    for block in model.blocks:
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


def clone_states(states: list | None) -> list | None:
    if states is None:
        return None
    return [state.clone() if state is not None else None for state in states]


def state_norm_summary(states: list | None) -> dict:
    if states is None:
        return {}
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


def collect_layer_update_stats(model: torch.nn.Module) -> list[dict]:
    layers = []
    for layer_index, block in enumerate(model.blocks):
        stats = getattr(block.memory, 'last_update_stats', None) or {}
        layers.append({
            'layer': layer_index,
            **{key: float(value) for key, value in stats.items()},
        })
    return layers


def prefill_memory_with_trace(
    model: torch.nn.Module,
    token_ids: list[int],
    batch_tokens: int,
    device: torch.device,
) -> tuple[list | None, list[dict]]:
    model_chunk_size = int(getattr(model.config, 'chunk_size', batch_tokens))
    step_tokens = min(batch_tokens, model_chunk_size)
    if step_tokens < 1:
        raise ValueError(f'Prefill batch tokens must be >= 1, got {batch_tokens}')

    states = None
    trace: list[dict] = []
    for chunk_index, start in enumerate(range(0, len(token_ids), step_tokens)):
        chunk = token_ids[start : start + step_tokens]
        if not chunk:
            continue

        x = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            _, states = model(x, states=states)

        trace.append({
            'chunk_index': chunk_index,
            'token_start': start,
            'token_end': start + len(chunk),
            'tokens': len(chunk),
            'layers': collect_layer_update_stats(model),
        })
    return states, trace


def summarize_prefill_trace(trace: list[dict]) -> dict:
    if not trace:
        return {
            'chunks': 0,
            'tokens': 0,
            'layers': [],
            'chunk_update_sample': {'first': [], 'last': []},
        }

    layer_totals: dict[int, dict[str, float]] = {}
    layer_counts: dict[int, int] = {}
    chunk_updates = []

    for chunk in trace:
        chunk_grad_norm = 0.0
        chunk_weight_delta_norm = 0.0
        chunk_momentum_delta_norm = 0.0
        for layer in chunk['layers']:
            layer_index = int(layer['layer'])
            totals = layer_totals.setdefault(layer_index, {
                'memory_grad_norm_sum': 0.0,
                'weight_delta_norm_sum': 0.0,
                'momentum_delta_norm_sum': 0.0,
                'effective_decay': 0.0,
                'effective_lr': 0.0,
                'effective_momentum': 0.0,
                'nonzero_update_steps': 0.0,
            })
            layer_counts[layer_index] = layer_counts.get(layer_index, 0) + 1

            grad_norm = float(layer.get('memory_grad_norm_sum', 0.0))
            weight_delta_norm = float(layer.get('weight_delta_norm_sum', 0.0))
            momentum_delta_norm = float(layer.get('momentum_delta_norm_sum', 0.0))
            totals['memory_grad_norm_sum'] += grad_norm
            totals['weight_delta_norm_sum'] += weight_delta_norm
            totals['momentum_delta_norm_sum'] += momentum_delta_norm
            totals['effective_decay'] += float(layer.get('effective_decay', 0.0))
            totals['effective_lr'] += float(layer.get('effective_lr', 0.0))
            totals['effective_momentum'] += float(layer.get('effective_momentum', 0.0))
            if weight_delta_norm > 0.0 or momentum_delta_norm > 0.0:
                totals['nonzero_update_steps'] += 1.0

            chunk_grad_norm += grad_norm
            chunk_weight_delta_norm += weight_delta_norm
            chunk_momentum_delta_norm += momentum_delta_norm

        chunk_updates.append({
            'chunk_index': chunk['chunk_index'],
            'token_start': chunk['token_start'],
            'token_end': chunk['token_end'],
            'tokens': chunk['tokens'],
            'memory_grad_norm_sum': chunk_grad_norm,
            'weight_delta_norm_sum': chunk_weight_delta_norm,
            'momentum_delta_norm_sum': chunk_momentum_delta_norm,
        })

    layer_summaries = []
    for layer_index in sorted(layer_totals):
        totals = layer_totals[layer_index]
        count = layer_counts[layer_index]
        layer_summaries.append({
            'layer': layer_index,
            'steps': count,
            'nonzero_update_steps': int(totals['nonzero_update_steps']),
            'memory_grad_norm_sum': totals['memory_grad_norm_sum'],
            'weight_delta_norm_sum': totals['weight_delta_norm_sum'],
            'momentum_delta_norm_sum': totals['momentum_delta_norm_sum'],
            'mean_effective_decay': totals['effective_decay'] / count,
            'mean_effective_lr': totals['effective_lr'] / count,
            'mean_effective_momentum': totals['effective_momentum'] / count,
        })

    return {
        'chunks': len(trace),
        'tokens': sum(int(chunk['tokens']) for chunk in trace),
        'layers': layer_summaries,
        'chunk_update_sample': {
            'first': chunk_updates[:4],
            'last': chunk_updates[-4:] if len(chunk_updates) > 4 else [],
        },
    }


def apply_memory_controls(
    model: torch.nn.Module,
    memory_decay: float,
    memory_lr: float,
    memory_momentum: float,
) -> None:
    model.config.memory_decay = memory_decay
    model.config.memory_lr = memory_lr
    model.config.memory_momentum = memory_momentum
    for block in model.blocks:
        block.memory.config.memory_decay = memory_decay
        block.memory.config.memory_lr = memory_lr
        block.memory.config.memory_momentum = memory_momentum


def score_target(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    target: str,
    states: list | None,
    device: torch.device,
) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    if not target_ids:
        raise RuntimeError(f'Empty target after tokenization: {target!r}')

    with torch.no_grad():
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, states = model(x, states=states)

    token_scores = []
    nll = 0.0
    for token_id in target_ids:
        next_logits = logits[0, -1].float()
        log_probs = F.log_softmax(next_logits, dim=-1)
        token_nll = float(-log_probs[token_id].item())
        token_logit = next_logits[token_id]
        rank = int((next_logits > token_logit).sum().item()) + 1
        token_scores.append({
            'token_id': int(token_id),
            'token': tokenizer.decode([token_id]),
            'nll': token_nll,
            'rank': rank,
        })
        nll += token_nll

        with torch.no_grad():
            x = torch.tensor([[token_id]], dtype=torch.long, device=device)
            logits, states = model(x, states=states)

    avg_nll = nll / len(target_ids)
    return {
        'target_tokens': len(target_ids),
        'target_text': tokenizer.decode(target_ids),
        'nll': nll,
        'avg_nll': avg_nll,
        'ppl': float(torch.exp(torch.tensor(avg_nll)).item()),
        'tokens': token_scores,
    }


def build_readonly_prefilled_model(
    checkpoint: dict,
    vocab_size: int,
    device: torch.device,
    article_ids: list[int],
    batch_tokens: int,
    memory_decay: float,
    memory_lr: float,
    memory_momentum: float,
) -> tuple[torch.nn.Module, list | None, list[dict]]:
    model = build_model(deepcopy(checkpoint), vocab_size, device)
    apply_memory_controls(model, memory_decay, memory_lr, memory_momentum)
    states, trace = prefill_memory_with_trace(model, article_ids, batch_tokens, device)
    disable_memory_updates(model)
    return model, states, trace


def build_readonly_reset_model(
    checkpoint: dict,
    vocab_size: int,
    device: torch.device,
    memory_decay: float,
    memory_lr: float,
    memory_momentum: float,
) -> torch.nn.Module:
    model = build_model(deepcopy(checkpoint), vocab_size, device)
    apply_memory_controls(model, memory_decay, memory_lr, memory_momentum)
    disable_memory_updates(model)
    return model


def main() -> None:
    args = parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, torch.device('cpu'))
    cfg = checkpoint.get('config', {})
    tokenizer_name = cfg.get('tokenizer_name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrong_article_span = args.article_span if args.wrong_article_span is None else args.wrong_article_span
    article = load_article_span(args.dataset, args.article_index, args.article_span)
    wrong_article = load_article_span(args.dataset, args.wrong_article_index, wrong_article_span)
    article_ids, article_raw_tokens = encode_article(tokenizer, article, args.prefill_token_limit)
    wrong_article_ids, wrong_article_raw_tokens = encode_article(
        tokenizer,
        wrong_article,
        args.prefill_token_limit,
    )
    probes = build_prompts(args, article)

    memory_decay = float(cfg.get('memory_decay', 0.0))
    memory_lr = float(cfg.get('memory_lr', 0.1))
    memory_momentum = float(cfg.get('memory_momentum', 0.9))
    memory_controls = {
        'memory_decay': memory_decay,
        'memory_lr': memory_lr,
        'memory_momentum': memory_momentum,
    }
    configured_chunk_size = int(cfg.get('chunk_size', args.prefill_batch_tokens))
    prefill_memory_step_tokens = min(args.prefill_batch_tokens, configured_chunk_size)

    header = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_step': checkpoint.get('step'),
        'checkpoint_best_val': checkpoint.get('best_val'),
        'device': str(device),
        'tokenizer': tokenizer_name,
        'dataset': str(args.dataset),
        'article_index': args.article_index,
        'article_span': args.article_span,
        'wrong_article_index': args.wrong_article_index,
        'wrong_article_span': wrong_article_span,
        'article_preview': normalize_space(article)[:500],
        'wrong_article_preview': normalize_space(wrong_article)[:240],
        'prefill_tokens': len(article_ids),
        'article_raw_tokens': article_raw_tokens,
        'article_truncated': len(article_ids) < article_raw_tokens,
        'wrong_prefill_tokens': len(wrong_article_ids),
        'wrong_article_raw_tokens': wrong_article_raw_tokens,
        'wrong_article_truncated': len(wrong_article_ids) < wrong_article_raw_tokens,
        'prefill_batch_tokens': args.prefill_batch_tokens,
        'configured_chunk_size': configured_chunk_size,
        'prefill_memory_step_tokens': prefill_memory_step_tokens,
        'memory_update_granularity': 'model_chunk',
        'memory_controls': memory_controls,
        'probes': probes,
        'readonly_query': True,
    }
    print(json.dumps({'type': 'header', **header}))

    vocab_size = int(tokenizer.vocab_size)
    reset_model = build_readonly_reset_model(
        checkpoint,
        vocab_size,
        device,
        memory_decay,
        memory_lr,
        memory_momentum,
    )
    correct_model, correct_states, correct_trace = build_readonly_prefilled_model(
        checkpoint,
        vocab_size,
        device,
        article_ids,
        args.prefill_batch_tokens,
        memory_decay,
        memory_lr,
        memory_momentum,
    )
    wrong_model, wrong_states, wrong_trace = build_readonly_prefilled_model(
        checkpoint,
        vocab_size,
        device,
        wrong_article_ids,
        args.prefill_batch_tokens,
        memory_decay,
        memory_lr,
        memory_momentum,
    )

    correct_trace_summary = summarize_prefill_trace(correct_trace)
    wrong_trace_summary = summarize_prefill_trace(wrong_trace)
    print(json.dumps({
        'type': 'prefill_trace',
        'condition': 'correct_article',
        'memory_update_granularity': 'model_chunk',
        'memory_step_tokens': prefill_memory_step_tokens,
        'summary': correct_trace_summary,
    }))
    print(json.dumps({
        'type': 'prefill_trace',
        'condition': 'wrong_article',
        'memory_update_granularity': 'model_chunk',
        'memory_step_tokens': prefill_memory_step_tokens,
        'summary': wrong_trace_summary,
    }))

    for probe in probes:
        reset = score_target(
            reset_model,
            tokenizer,
            probe['prompt'],
            probe['target'],
            states=None,
            device=device,
        )
        correct = score_target(
            correct_model,
            tokenizer,
            probe['prompt'],
            probe['target'],
            states=clone_states(correct_states),
            device=device,
        )
        wrong = score_target(
            wrong_model,
            tokenizer,
            probe['prompt'],
            probe['target'],
            states=clone_states(wrong_states),
            device=device,
        )
        print(json.dumps({
            'type': 'result',
            'memory_controls': memory_controls,
            'prompt': probe['prompt'],
            'target': probe['target'],
            'source': probe['source'],
            'reset': reset,
            'correct_article': correct,
            'wrong_article': wrong,
            'deltas': {
                'correct_minus_reset_avg_nll': correct['avg_nll'] - reset['avg_nll'],
                'wrong_minus_reset_avg_nll': wrong['avg_nll'] - reset['avg_nll'],
                'correct_minus_wrong_avg_nll': correct['avg_nll'] - wrong['avg_nll'],
            },
            'state_norms': {
                'correct_article': state_norm_summary(correct_states),
                'wrong_article': state_norm_summary(wrong_states),
            },
        }))


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
