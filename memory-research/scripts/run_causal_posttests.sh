#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_LOG="${TRAIN_LOG:-}"
RUN_DIR="${RUN_DIR:-}"
OUT_LOG="${OUT_LOG:-posttests.log}"

if [[ -z "$TRAIN_LOG" || -z "$RUN_DIR" ]]; then
  echo "Set TRAIN_LOG and RUN_DIR before running posttests." >&2
  exit 2
fi

cd "$PROJECT"
{
  echo "{'type': 'posttest_watcher', 'status': 'waiting', 'train_log': '$TRAIN_LOG', 'run_dir': '$RUN_DIR'}"
  while true; do
    if grep -q "'status': 'done'" "$TRAIN_LOG"; then
      break
    fi
    if grep -q "Traceback" "$TRAIN_LOG"; then
      echo "{'type': 'posttest_watcher', 'status': 'training_failed'}"
      exit 1
    fi
    sleep 120
  done

  CKPT="$RUN_DIR/best.pt"
  echo "{'type': 'posttest_watcher', 'status': 'training_done', 'checkpoint': '$CKPT'}"
  "$PYTHON_BIN" scripts/diagnose_mac_memory_usage.py \
    --checkpoint "$CKPT" \
    --segments 32 \
    --wrong-article-index 1000 \
    --prefill-batch-tokens 1
  "$PYTHON_BIN" scripts/probe_article_memory.py \
    --checkpoint "$CKPT" \
    --wrong-article-index 1000 \
    --prefill-batch-tokens 1
  echo "{'type': 'posttest_watcher', 'status': 'done'}"
} >> "$OUT_LOG" 2>&1
