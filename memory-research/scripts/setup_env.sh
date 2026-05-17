#!/usr/bin/env bash
set -euo pipefail

# Prepare a local environment for Memory Workspace without launching training.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "Environment prepared."
echo "Next commands:"
echo "  source .venv/bin/activate"
echo "  python scripts/check_ready.py"
