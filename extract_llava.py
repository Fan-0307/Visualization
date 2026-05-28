"""
LLaVA Attention Extraction Script
按照 Comparative VLM Attention Analysis System 规范输出
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from pathlib import Path
import time
from local_vqa_loader import load_local_vqa_dataset

# ============ Configuration ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
OUTPUT_DIR = Path("data")
MODEL_NAME = "llava"
RANDOM_SEED = 42
VQA_DATA_DIR = Path("train/vqa_data")  # 本地 VQA 数据目录

# 抽样方式选择
USE_STRATIFIED_SAMPLING = True  # True=分层抽样(均衡), False=随机抽样
STRATIFIED_SAMPLE_FILE = VQA_DATA_DIR / "stratified_sample_ids.json"

# ============ Setup Directories ============
(OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models" / MODEL_NAME / "predictions").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models" / MODEL_NAME / "attention").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models" / MODEL_NAME / "summaries").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

# ============ Load Dataset ============
print("Loading VQA v2 validation set from local files (COCO images)...")
np.random.seed(RANDOM_SEED)

# 从本地加载 VQA v2 数据
if USE_STRATIFIED_SAMPLING and STRATIFIED_SAMPLE_FILE.exists():
    print("📊 使用分层抽样（分类均衡）")
    vqa_dataset = load_local_vqa_dataset(VQA_DATA_DIR, STRATIFIED_SAMPLE_FILE)
    samples = [vqa_dataset[i] for i in range(len(vqa_dataset))]
else:
    print("🎲 使用随机抽样")
    vqa_dataset = load_local_vqa_dataset(VQA_DATA_DIR)
    sample_indices = np.random.choice(len(vqa_dataset), NUM_SAMPLES, replace=False)
    samples = [vqa_dataset[int(i)] for i in sample_indices]

print(f"Loaded {len(samples)} samples from local VQA v2 (based on COCO images)")

# ============ Load LLaVA Model ============
print(f"\n=== Loading LLaVA-v1.6-vicuna-7b on {DEVICE} ===")
llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
llava_model.eval()

# ============ Attention Hook ============
attention_cache = {}

def get_attention_hook(name):
    def hook(module, input, output):
        # output is typically (hidden_states, attention_weights) or just hidden_states
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            if attn is not None and torch.is_tensor(attn):
                attention_cache[name] = attn.detach().cpu()
        # Some layers return AttentionOutput with attentions attribute
        elif hasattr(output, 'attentions') and output.attentions is not None:
            attention_cache[name] = output.attentions.detach().cpu()
    return hook

# Register hooks for last 4 layers
# LLaVA uses a language model (Vicuna/LLaMA) backbone
num_layers = len(llava_model.language_model.model.layers)
for layer_idx in range(num_layers - 4, num_layers):
    layer = llava_model.language_model.model.layers[layer_idx]
    if hasattr(layer, 'self_attn'):
        layer.self_attn.register_forward_hook(
            get_attention_hook(f"layer_{layer_idx}")
        )

# ============ Process Samples ============
samples_metadata = []

print(f"\n=== Processing {NUM_SAMPLES} samples ===")
for i, sample in enumerate(tqdm(samples, desc="LLaVA Extraction")):
    try:
        sample_id = f"vqa_{i:06d}"
        image = sample['image'].convert('RGB')
        question = sample['question']
        answers = sample['answers']
        gt_answer = answers[0]['answer'] if answers else ""

        # 获取分类信息
        question_type = sample.get('question_type', 'unknown')
        answer_type = sample.get('answer_type', 'other')

        # Save image
        image_path = f"images/{sample_id}.jpg"
        image.save(OUTPUT_DIR / image_path)

        # Prepare prompt in LLaVA format
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        # Prepare inputs - 图片和文字同时输入
        inputs = llava_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE, torch.float16)

        # Forward pass with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = llava_model.generate(
                **inputs,
                max_new_tokens=20,
                output_attentions=True,
                return_dict_in_generate=True
            )
            pred_answer = llava_processor.decode(outputs.sequences[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in pred_answer:
                pred_answer = pred_answer.split("ASSISTANT:")[-1].strip()
        inference_time = (time.time() - start_time) * 1000

        # Check correctness
        correct = gt_answer.lower() in pred_answer.lower() if gt_answer else False
        confidence = 0.8 if correct else 0.3  # Placeholder

        # ============ Save Prediction ============
        prediction_data = {
            "sample_id": sample_id,
            "model_name": "LLaVA-v1.6-vicuna-7b",
            "prediction": pred_answer,
            "confidence": float(confidence),
            "correct": correct,
            "inference_time_ms": int(inference_time),
            "tokens": question.split(),
            "generated_text": pred_answer,
            "error_type": None
        }

        pred_path = OUTPUT_DIR / "models" / MODEL_NAME / "predictions" / f"{sample_id}.json"
        with open(pred_path, 'w') as f:
            json.dump(prediction_data, f, indent=2)

        # ============ Save Attention (last 4 layers) ============
        attn_dir = OUTPUT_DIR / "models" / MODEL_NAME / "attention" / sample_id
        attn_dir.mkdir(parents=True, exist_ok=True)

        layer_stats = {}
        for layer_name, attn_tensor in attention_cache.items():
            # attn_tensor shape: [batch, num_heads, seq_len, seq_len]
            # For LLaVA: seq_len includes both image tokens and text tokens

            attn = attn_tensor[0]  # Remove batch: [num_heads, seq_len, seq_len]
            num_heads = attn.shape[0]
            seq_len = attn.shape[1]

            # LLaVA embeds image as tokens at the beginning
            # We need to extract cross-attention: text tokens attending to image tokens
            # Assume first N tokens are image, rest are text
            num_image_tokens = 576  # Standard for ViT
            num_text_tokens = len(question.split())

            if seq_len > num_image_tokens:
                # Extract text-to-image attention
                # Text tokens are after image tokens
                text_start = num_image_tokens
                text_end = min(text_start + num_text_tokens, seq_len)

                # [num_heads, text_tokens, image_tokens]
                cross_attn = attn[:, text_start:text_end, :num_image_tokens]

                # Ensure we have exactly num_text_tokens
                if cross_attn.shape[1] < num_text_tokens:
                    # Pad if needed
                    padding = torch.zeros(
                        num_heads,
                        num_text_tokens - cross_attn.shape[1],
                        num_image_tokens
                    )
                    cross_attn = torch.cat([cross_attn, padding], dim=1)
                elif cross_attn.shape[1] > num_text_tokens:
                    cross_attn = cross_attn[:, :num_text_tokens, :]

                # Ensure we have exactly 576 image patches
                if cross_attn.shape[2] != 576:
                    # Interpolate to 24x24 = 576
                    side = int(np.sqrt(cross_attn.shape[2]))
                    cross_attn_2d = cross_attn.reshape(num_heads, num_text_tokens, side, side)
                    cross_attn_2d = torch.nn.functional.interpolate(
                        cross_attn_2d,
                        size=(24, 24),
                        mode='bilinear',
                        align_corners=False
                    )
                    cross_attn = cross_attn_2d.reshape(num_heads, num_text_tokens, 576)

                # Save as float16
                attn_array = cross_attn.numpy().astype(np.float16)
                layer_num = layer_name.split('_')[1]
                np.save(attn_dir / f"layer_{layer_num}.npy", attn_array)

                # Compute statistics
                attn_flat = cross_attn.reshape(num_heads, -1)
                entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1).mean().item()
                sparsity = (attn_flat < 0.01).float().mean().item()

                # Center bias
                attn_spatial = cross_attn.mean(dim=1).reshape(24, 24)
                center_bias = attn_spatial[8:16, 8:16].mean().item()

                layer_stats[f"layer_{layer_num}"] = {
                    "entropy": float(entropy),
                    "sparsity": float(sparsity),
                    "center_bias": float(center_bias),
                    "text_image_ratio": float(cross_attn.mean().item())
                }

        # ============ Save Summary ============
        summary_data = {
            "sample_id": sample_id,
            "model_name": "LLaVA-v1.6-vicuna-7b",
            "layer_statistics": layer_stats,
            "object_coverage": {},
            "dominant_regions": [],
            "attention_behavior": "uniform"
        }

        summary_path = OUTPUT_DIR / "models" / MODEL_NAME / "summaries" / f"{sample_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        # ============ Save Sample Metadata ============
        sample_meta = {
            "sample_id": sample_id,
            "image_path": image_path,
            "question": question,
            "gt_answer": gt_answer,
            "question_type": question_type,  # 使用数据集自带的分类
            "answer_type": answer_type,      # yes/no, number, other
            "objects": [],
            "bbox_annotations": {}
        }
        samples_metadata.append(sample_meta)

        # Clear cache
        attention_cache.clear()

    except Exception as e:
        print(f"\n❌ Sample {i} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

    if (i + 1) % 10 == 0:  # More frequent cleanup for larger model
        torch.cuda.empty_cache()

# ============ Save Global Metadata ============
with open(OUTPUT_DIR / "metadata" / "samples.json", 'w') as f:
    json.dump(samples_metadata, f, indent=2)

print(f"\n✅ LLaVA extraction complete!")
print(f"   Processed: {len(samples_metadata)} samples")
print(f"   Output dir: {OUTPUT_DIR}")
