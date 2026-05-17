from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig


DEFAULT_PROMPTS = [
    'The history of artificial intelligence',
    'In mathematics, a group is',
    'The city of London is known for',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sample text from a trained MAC/Titans WikiText checkpoint.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best.pt'),
    )
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--prompt', action='append', default=None)
    parser.add_argument(
        '--prefill-path',
        type=Path,
        default=None,
        help='Optional text or jsonl file used to update neural memory before prompting.',
    )
    parser.add_argument(
        '--prefill-text',
        type=str,
        default=None,
        help='Optional literal text used to update neural memory before prompting.',
    )
    parser.add_argument('--prefill-token-limit', type=int, default=4096)
    parser.add_argument('--prefill-batch-tokens', type=int, default=128)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(checkpoint: dict, tokenizer_vocab_size: int, device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get('config', {})
    model = MemoryMAC(
        MemoryMACConfig(
            dim=int(cfg.get('dim', 128)),
            num_heads=int(cfg.get('num_heads', 4)),
            num_layers=int(cfg.get('num_layers', 2)),
            vocab_size=tokenizer_vocab_size,
            max_context_length=int(cfg.get('seq_len', 128)),
            chunk_size=int(cfg.get('chunk_size', 32)),
            window_size=int(cfg.get('window_size', 32)),
            memory_decay=float(cfg.get('memory_decay', 0.0)),
            memory_lr=float(cfg.get('memory_lr', 0.1)),
            memory_momentum=float(cfg.get('memory_momentum', 0.9)),
        )
    )
    if model.live_backend is None:
        raise RuntimeError(f'Live Titans backend is not ready: {model.titans_status.error}')

    backend = model.live_backend.to(device)
    backend.load_state_dict(checkpoint['model_state'])
    backend.eval()
    for module in backend.modules():
        if hasattr(module, '_use_triton'):
            module._use_triton = False
    return backend


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0 and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[-1], float('-inf'))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(0, sorted_indices, remove_sorted)
        logits = logits.masked_fill(remove, float('-inf'))

    return logits


def sample_next_id(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    next_logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(next_logits).item())

    next_logits = filter_logits(next_logits / temperature, top_k=top_k, top_p=top_p)
    probs = F.softmax(next_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        input_ids = [tokenizer.eos_token_id]

    max_context = int(model.config.max_seq_len)
    generated = list(input_ids)

    for _ in range(max_new_tokens):
        context = generated[-max_context:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)

        next_id = sample_next_id(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p)
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


def iter_texts_from_path(path: Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if path.suffix == '.jsonl':
                rec = json.loads(stripped)
                text = (rec.get('text') or '').strip()
                if text:
                    yield text
            else:
                yield stripped


def encode_prefill(
    tokenizer: AutoTokenizer,
    prefill_path: Path | None,
    prefill_text: str | None,
    token_limit: int,
) -> list[int]:
    ids: list[int] = []

    if prefill_text:
        ids.extend(tokenizer.encode(prefill_text, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)

    if prefill_path is not None:
        for text in iter_texts_from_path(prefill_path):
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            if len(ids) >= token_limit:
                break

    return ids[:token_limit]


def prefill_memory(
    model: torch.nn.Module,
    token_ids: list[int],
    batch_tokens: int,
    device: torch.device,
) -> list | None:
    states = None
    for start in range(0, len(token_ids), batch_tokens):
        chunk = token_ids[start : start + batch_tokens]
        if not chunk:
            continue
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            _, states = model(x, states=states)
    return states


def generate_stateful(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    states: list | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    generated = list(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, states = model(x, states=states)

    for _ in range(max_new_tokens):
        next_id = sample_next_id(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p)
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, states = model(x, states=states)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, torch.device('cpu'))
    cfg = checkpoint.get('config', {})
    tokenizer_name = cfg.get('tokenizer_name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = build_model(checkpoint, int(tokenizer.vocab_size), device)
    prompts = args.prompt or DEFAULT_PROMPTS
    prefill_ids = encode_prefill(
        tokenizer=tokenizer,
        prefill_path=args.prefill_path,
        prefill_text=args.prefill_text,
        token_limit=args.prefill_token_limit,
    )
    prefill_states = None
    if prefill_ids:
        prefill_states = prefill_memory(
            model=model,
            token_ids=prefill_ids,
            batch_tokens=args.prefill_batch_tokens,
            device=device,
        )

    print(json.dumps({
        'checkpoint': str(args.checkpoint),
        'step': checkpoint.get('step'),
        'best_val': checkpoint.get('best_val'),
        'device': str(device),
        'tokenizer': tokenizer_name,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'prefill_tokens': len(prefill_ids),
        'prefill_path': str(args.prefill_path) if args.prefill_path else None,
        'prefill_mode': 'stateful' if prefill_ids else 'context-replay',
    }))

    for idx, prompt in enumerate(prompts, start=1):
        if prefill_states is None:
            text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
        else:
            states = [state.clone() if state is not None else None for state in prefill_states]
            text = generate_stateful(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                states=states,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
        print(f'\n=== sample {idx} ===')
        print(text)


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
