#!/bin/bash
# ============================================================
# Model path & name configuration — edit these before running.
#
# Options:
#   1. HuggingFace model id (downloads from HF hub if online):
#      export QWEN_MODEL_PATH_2B="Qwen/Qwen3-VL-2B-Instruct"
#      export QWEN_MODEL_PATH_8B="Qwen/Qwen3-VL-8B-Instruct"
#
#   2. Local path (if models are pre-downloaded):
#      export QWEN_MODEL_PATH_2B="/data/models/Qwen3-VL-2B-Instruct"
#      export QWEN_MODEL_PATH_8B="/data/models/Qwen3-VL-8B-Instruct"
# ============================================================

# --- BLIP (BLIP-vqa-base, ~500MB) ---
export BLIP_MODEL_NAME="${BLIP_MODEL_NAME:-BLIP-vqa-base}"
export BLIP_MODEL_PATH="${BLIP_MODEL_PATH:-Salesforce/blip-vqa-base}"

# --- Qwen3-VL series ---
# 2B (~4GB VRAM)
export QWEN_MODEL_NAME_2B="${QWEN_MODEL_NAME_2B:-Qwen3-VL-2B}"
export QWEN_MODEL_PATH_2B="${QWEN_MODEL_PATH_2B:-Qwen/Qwen3-VL-2B-Instruct}"
# 8B (~16GB VRAM)
export QWEN_MODEL_NAME_8B="${QWEN_MODEL_NAME_8B:-Qwen3-VL-8B}"
export QWEN_MODEL_PATH_8B="${QWEN_MODEL_PATH_8B:-Qwen/Qwen3-VL-8B-Instruct}"

# --- LLaVA series ---
# 7B (~14GB VRAM)
export LLAVA_MODEL_NAME_7B="${LLAVA_MODEL_NAME_7B:-LLaVA-1.5-7B}"
export LLAVA_MODEL_PATH_7B="${LLAVA_MODEL_PATH_7B:-llava-hf/llava-1.5-7b-hf}"
# 13B (~26GB VRAM)
export LLAVA_MODEL_NAME_13B="${LLAVA_MODEL_NAME_13B:-LLaVA-1.5-13B}"
export LLAVA_MODEL_PATH_13B="${LLAVA_MODEL_PATH_13B:-llava-hf/llava-1.5-13b-hf}"

echo "Model configurations:"
echo "  BLIP       : $BLIP_MODEL_NAME @ $BLIP_MODEL_PATH"
echo "  Qwen3-VL   : $QWEN_MODEL_NAME_2B @ $QWEN_MODEL_PATH_2B"
echo "  Qwen3-VL   : $QWEN_MODEL_NAME_8B @ $QWEN_MODEL_PATH_8B"
echo "  LLaVA-1.5  : $LLAVA_MODEL_NAME_7B @ $LLAVA_MODEL_PATH_7B"
echo "  LLaVA-1.5  : $LLAVA_MODEL_NAME_13B @ $LLAVA_MODEL_PATH_13B"
