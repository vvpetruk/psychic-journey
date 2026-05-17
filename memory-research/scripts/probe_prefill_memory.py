from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_wikitext_checkpoint import build_model, load_checkpoint, prefill_memory


FACTS = [
    {
        'text': 'The archivist recorded that the orbid flower code is ZANTH-47.',
        'prompt': 'The orbid flower code is',
        'target': ' ZANTH-47',
    },
    {
        'text': 'The archivist recorded that the silver river password is LUMEN-82.',
        'prompt': 'The silver river password is',
        'target': ' LUMEN-82',
    },
    {
        'text': 'The archivist recorded that the hidden comet city is Veyra.',
        'prompt': 'The hidden comet city is',
        'target': ' Veyra',
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Probe whether prefilling Titans neural memory helps retrieve synthetic facts.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/best.pt'),
    )
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--batch-tokens', type=int, default=128)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


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

    return {
        'target_tokens': len(target_ids),
        'target_text': tokenizer.decode(target_ids),
        'nll': nll,
        'avg_nll': nll / len(target_ids),
        'ppl': float(torch.exp(torch.tensor(nll / len(target_ids))).item()),
        'tokens': token_scores,
    }


def build_prefill_text(repeats: int) -> str:
    lines = []
    for _ in range(repeats):
        lines.extend(fact['text'] for fact in FACTS)
    return '\n'.join(lines)


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

    dynamic_model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    no_update_model = build_model(deepcopy(checkpoint), int(tokenizer.vocab_size), device)
    disable_memory_updates(no_update_model)

    prefill_text = build_prefill_text(args.repeats)
    prefill_ids = tokenizer.encode(prefill_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        prefill_ids.append(tokenizer.eos_token_id)

    dynamic_states = prefill_memory(dynamic_model, prefill_ids, args.batch_tokens, device)
    no_update_states = prefill_memory(no_update_model, prefill_ids, args.batch_tokens, device)

    print(json.dumps({
        'checkpoint': str(args.checkpoint),
        'step': checkpoint.get('step'),
        'best_val': checkpoint.get('best_val'),
        'device': str(device),
        'tokenizer': tokenizer_name,
        'facts': len(FACTS),
        'repeats': args.repeats,
        'prefill_tokens': len(prefill_ids),
    }))

    for fact in FACTS:
        result = {
            'prompt': fact['prompt'],
            'target': fact['target'],
            'no_prefill': score_target(
                dynamic_model,
                tokenizer,
                fact['prompt'],
                fact['target'],
                states=None,
                device=device,
            ),
            'prefill_dynamic': score_target(
                dynamic_model,
                tokenizer,
                fact['prompt'],
                fact['target'],
                states=clone_states(dynamic_states),
                device=device,
            ),
            'prefill_no_update': score_target(
                no_update_model,
                tokenizer,
                fact['prompt'],
                fact['target'],
                states=clone_states(no_update_states),
                device=device,
            ),
        }
        print(json.dumps(result))


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
