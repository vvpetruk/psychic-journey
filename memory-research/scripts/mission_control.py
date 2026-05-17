from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / 'artifacts' / 'runs'


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_segments(metrics: Iterable[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    previous_step: int | None = None
    for row in metrics:
        step = row.get('step')
        if (
            current
            and isinstance(step, int)
            and isinstance(previous_step, int)
            and step <= previous_step
        ):
            segments.append(current)
            current = []
        current.append(row)
        if isinstance(step, int):
            previous_step = step
    if current:
        segments.append(current)
    return segments


def run_score(path: Path) -> float:
    candidates = [path / 'metrics.jsonl', path / 'config.json', path / 'best.pt']
    candidates.extend(path.glob('checkpoint-step-*.pt'))
    mtimes = [candidate.stat().st_mtime for candidate in candidates if candidate.exists()]
    return max(mtimes) if mtimes else path.stat().st_mtime


def latest_run_dir(runs_root: Path = RUNS_ROOT) -> Path | None:
    if not runs_root.exists():
        return None
    runs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not runs:
        return None
    return max(runs, key=run_score)


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser()
    if args.run_name:
        return RUNS_ROOT / args.run_name
    latest = latest_run_dir()
    if latest is not None:
        return latest
    return RUNS_ROOT / 'wikitext-real'


def print_recent_runs(limit: int) -> None:
    if not RUNS_ROOT.exists():
        print('runs_root_missing:', RUNS_ROOT)
        return
    runs = sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_dir()),
        key=run_score,
        reverse=True,
    )
    for path in runs[:limit]:
        print({'run': path.name, 'path': str(path), 'score': run_score(path)})


def summarize_losses(name: str, values: list[float]) -> None:
    if values:
        print(f'{name}_summary:', {'first': values[0], 'last': values[-1], 'min': min(values)})


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize Memory Workspace training artifacts.')
    parser.add_argument('--run-dir', type=str, default=None, help='Explicit artifact directory to inspect.')
    parser.add_argument('--run-name', type=str, default=None, help='Run directory name under artifacts/runs.')
    parser.add_argument('--list', action='store_true', help='List recent run directories and exit.')
    parser.add_argument('--limit', type=int, default=10, help='Number of runs to list with --list.')
    args = parser.parse_args()

    if args.list:
        print_recent_runs(args.limit)
        return

    root = resolve_run_dir(args)
    config = load_json(root / 'config.json')
    metrics = load_metrics(root / 'metrics.jsonl')
    segments = metric_segments(metrics)
    active_metrics = segments[-1] if segments else []
    best_ckpt = root / 'best.pt'
    checkpoints = sorted(root.glob('checkpoint-step-*.pt'))

    print('MISSION CONTROL')
    print('root:', root)
    print('config_present:', config is not None)
    print('metrics_rows:', len(metrics))
    print('metric_segments:', len(segments))
    print('active_segment_rows:', len(active_metrics))
    print('best_checkpoint_present:', best_ckpt.exists())
    print('periodic_checkpoints:', len(checkpoints))

    if config:
        print('run_config:', {
            'steps': config.get('steps'),
            'batch_size': config.get('batch_size'),
            'seq_len': config.get('seq_len'),
            'lr': config.get('lr'),
            'memory_decay': config.get('memory_decay'),
            'memory_lr': config.get('memory_lr'),
            'memory_ablation': config.get('memory_ablation'),
            'tokenizer_name': config.get('tokenizer_name'),
        })

    if active_metrics:
        last = active_metrics[-1]
        print('last_metric:', last)
        memory_update = last.get('memory_update') or {}
        if memory_update.get('summary'):
            print('memory_update_summary:', memory_update['summary'])
        summarize_losses('train_loss', [m['train_loss'] for m in active_metrics if 'train_loss' in m])
        summarize_losses('val_loss', [m['val_loss'] for m in active_metrics if 'val_loss' in m])


if __name__ == '__main__':
    main()
