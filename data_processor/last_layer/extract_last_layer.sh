#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/fan/Soccernet_GSR/sn-gamestate/.venv/bin/python}"

"$PYTHON_BIN" "$SCRIPT_DIR/extract_last_layer_from_evolution.py" "$@"
