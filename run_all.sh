#!/bin/bash
# ============================================================
# Run attention extraction for all models (32 samples each)
#
# Usage:
#   source setup_env.sh        # configure model paths & names
#   bash run_all.sh            # run all models & sizes
#   bash run_all.sh qwen_2b    # run single model+size
#   bash run_all.sh qwen       # run all Qwen sizes
#   bash run_all.sh llava      # run all LLaVA sizes
# ============================================================

set -euo pipefail

RUN="${1:-all}"

# For Qwen scripts, env vars: QWEN_MODEL_NAME / QWEN_MODEL_PATH
run_qwen() {
    local label="$1" name="$2" path="$3"
    local script="data/qwen/run.py"
    if [ ! -f "$script" ]; then
        echo "[skip] $label: $script not found"
        return
    fi
    echo "========================================="
    echo "  Running: $label"
    echo "  Model:   $name"
    echo "  Path:    $path"
    echo "  Started: $(date)"
    echo "========================================="
    env QWEN_MODEL_NAME="$name" QWEN_MODEL_PATH="$path" python "$script"
    echo "  Finished: $label at $(date)"
}

# For LLaVA scripts, env vars: LLAVA_MODEL_NAME / LLAVA_MODEL_PATH
run_llava() {
    local label="$1" name="$2" path="$3"
    local script="data/llava/run.py"
    if [ ! -f "$script" ]; then
        echo "[skip] $label: $script not found"
        return
    fi
    echo "========================================="
    echo "  Running: $label"
    echo "  Model:   $name"
    echo "  Path:    $path"
    echo "  Started: $(date)"
    echo "========================================="
    env LLAVA_MODEL_NAME="$name" LLAVA_MODEL_PATH="$path" python "$script"
    echo "  Finished: $label at $(date)"
}

# For BLIP scripts, env vars: BLIP_MODEL_NAME / BLIP_MODEL_PATH
run_blip() {
    local label="$1" name="$2" path="$3"
    local script="data/blip/run.py"
    if [ ! -f "$script" ]; then
        echo "[skip] $label: $script not found"
        return
    fi
    echo "========================================="
    echo "  Running: $label"
    echo "  Model:   $name"
    echo "  Path:    $path"
    echo "  Started: $(date)"
    echo "========================================="
    env BLIP_MODEL_NAME="$name" BLIP_MODEL_PATH="$path" python "$script"
    echo "  Finished: $label at $(date)"
}

# For CLIP scripts, env vars: CLIP_MODEL_NAME / CLIP_MODEL_PATH
run_clip() {
    local label="$1" name="$2" path="$3"
    local script="data/clip/run.py"
    if [ ! -f "$script" ]; then
        echo "[skip] $label: $script not found"
        return
    fi
    echo "========================================="
    echo "  Running: $label"
    echo "  Model:   $name"
    echo "  Path:    $path"
    echo "  Started: $(date)"
    echo "========================================="
    env CLIP_MODEL_NAME="$name" CLIP_MODEL_PATH="$path" python "$script"
    echo "  Finished: $label at $(date)"
}

# ============================================================
# BLIP
# ============================================================
if [ "$RUN" = "all" ] || [ "$RUN" = "blip" ]; then
    run_blip "BLIP-base" \
        "${BLIP_MODEL_NAME:-BLIP-vqa-base}" \
        "${BLIP_MODEL_PATH:-Salesforce/blip-vqa-base}"
fi

# ============================================================
# CLIP (vision-only self-attention, zero-shot VQA)
# ============================================================
if [ "$RUN" = "all" ] || [ "$RUN" = "clip" ]; then
    run_clip "CLIP-ViT-B-32" \
        "${CLIP_MODEL_NAME:-CLIP-ViT-B-32}" \
        "${CLIP_MODEL_PATH:-openai/clip-vit-base-patch32}"
fi

# ============================================================
# Qwen3-VL series (same architecture, different sizes)
# ============================================================
if [ "$RUN" = "all" ] || [ "$RUN" = "qwen" ] || [ "$RUN" = "qwen_2b" ]; then
    run_qwen "Qwen3-VL-2B" \
        "${QWEN_MODEL_NAME_2B:-Qwen3-VL-2B}" \
        "${QWEN_MODEL_PATH_2B:-Qwen/Qwen3-VL-2B-Instruct}"
fi

if [ "$RUN" = "all" ] || [ "$RUN" = "qwen" ] || [ "$RUN" = "qwen_8b" ]; then
    run_qwen "Qwen3-VL-8B" \
        "${QWEN_MODEL_NAME_8B:-Qwen3-VL-8B}" \
        "${QWEN_MODEL_PATH_8B:-Qwen/Qwen3-VL-8B-Instruct}"
fi

# ============================================================
# LLaVA series (same architecture, different sizes)
# ============================================================
if [ "$RUN" = "all" ] || [ "$RUN" = "llava" ] || [ "$RUN" = "llava_7b" ]; then
    run_llava "LLaVA-1.5-7B" \
        "${LLAVA_MODEL_NAME_7B:-LLaVA-1.5-7B}" \
        "${LLAVA_MODEL_PATH_7B:-llava-hf/llava-1.5-7b-hf}"
fi

if [ "$RUN" = "all" ] || [ "$RUN" = "llava" ] || [ "$RUN" = "llava_13b" ]; then
    run_llava "LLaVA-1.5-13B" \
        "${LLAVA_MODEL_NAME_13B:-LLaVA-1.5-13B}" \
        "${LLAVA_MODEL_PATH_13B:-llava-hf/llava-1.5-13b-hf}"
fi

echo "========================================="
echo "  All done. Building index..."
echo "========================================="
python data/build_index.py
echo "  Index built at $(date)"
