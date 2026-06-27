#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/fan/Soccernet_GSR/sn-gamestate/.venv/bin/python}"

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

models=("CLIP.py" "BLIP2.py" "LLaVA.py" "Qwen2-VL.py")
for script in "${models[@]}"; do
  echo
  echo "========== Running $script =========="
  "$PYTHON_BIN" "$SCRIPT_DIR/$script" "$@"
done

echo
echo "========== Done =========="
