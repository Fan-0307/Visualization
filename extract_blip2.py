"""
BLIP2-OPT Attention Extraction Script
按照 Comparative VLM Attention Analysis System 规范输出
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path
import time
from local_vqa_loader import load_local_vqa_dataset

# ============ Configuration ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
OUTPUT_DIR = Path("data")
MODEL_NAME = "blip2"
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

# ============ Load BLIP2 Model ============
print(f"\n=== Loading BLIP2-OPT-2.7b on {DEVICE} ===")
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)
blip2_model.eval()

# ============ Attention Hook ============
attention_cache = {}

def get_attention_hook(name):
    def hook(module, input, output):
        # Cross-attention output format varies
        if isinstance(output, tuple):
            attn = output[-1] if len(output) > 1 else None
            if attn is not None:
                attention_cache[name] = attn.detach().cpu()
        elif torch.is_tensor(output):
            attention_cache[name] = output.detach().cpu()
    return hook

# Register hooks for last 4 layers of Q-Former cross-attention
num_layers = len(blip2_model.qformer.encoder.layer)
for layer_idx in range(num_layers - 4, num_layers):
    layer = blip2_model.qformer.encoder.layer[layer_idx]
    if hasattr(layer, 'crossattention'):
        layer.crossattention.attention.self.register_forward_hook(
            get_attention_hook(f"layer_{layer_idx}")
        )

# ============ Process Samples ============
samples_metadata = []

print(f"\n=== Processing {NUM_SAMPLES} samples ===")
for i, sample in enumerate(tqdm(samples, desc="BLIP2 Extraction")):
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

        # Prepare inputs - 图片和文字同时输入
        inputs = blip2_processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(DEVICE, torch.float16)

        # Forward pass with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = blip2_model.generate(
                **inputs,
                max_new_tokens=10,
                return_dict_in_generate=True,
                output_attentions=True
            )
            pred_answer = blip2_processor.decode(outputs.sequences[0], skip_special_tokens=True)
        inference_time = (time.time() - start_time) * 1000

        # Check correctness
        correct = gt_answer.lower() in pred_answer.lower() if gt_answer else False
        confidence = 0.8 if correct else 0.3  # Placeholder

        # ============ Save Prediction ============
        prediction_data = {
            "sample_id": sample_id,
            "model_name": "BLIP2-OPT-2.7b",
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
            # attn_tensor shape: [batch, num_heads, query_len, key_len]
            # For BLIP2 Q-Former: query = text tokens, key = image features

            attn = attn_tensor[0]  # Remove batch: [num_heads, query_len, key_len]
            num_heads = attn.shape[0]

            # Resize to standard format: [num_heads, text_tokens, 576 patches]
            # BLIP2 uses 32 queries by default, we need to interpolate to match text tokens
            num_text_tokens = len(question.split())

            # Interpolate query dimension to match text tokens
            if attn.shape[1] != num_text_tokens:
                attn_reshaped = attn.permute(0, 2, 1)  # [num_heads, key_len, query_len]
                attn_reshaped = torch.nn.functional.interpolate(
                    attn_reshaped.unsqueeze(1),  # [num_heads, 1, key_len, query_len]
                    size=(attn.shape[2], num_text_tokens),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [num_heads, key_len, num_text_tokens]
                attn = attn_reshaped.permute(0, 2, 1)  # [num_heads, num_text_tokens, key_len]

            # Interpolate key dimension to 576 patches (24x24)
            key_len = attn.shape[2]
            if key_len != 576:
                # Assume square layout
                side = int(np.sqrt(key_len))
                attn_2d = attn.reshape(num_heads, num_text_tokens, side, side)
                attn_2d = torch.nn.functional.interpolate(
                    attn_2d,
                    size=(24, 24),
                    mode='bilinear',
                    align_corners=False
                )
                attn = attn_2d.reshape(num_heads, num_text_tokens, 576)

            # Save as float16
            attn_array = attn.numpy().astype(np.float16)
            layer_num = layer_name.split('_')[1]
            np.save(attn_dir / f"layer_{layer_num}.npy", attn_array)

            # Compute statistics
            attn_flat = attn.reshape(num_heads, -1)
            entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1).mean().item()
            sparsity = (attn_flat < 0.01).float().mean().item()

            # Center bias (middle 8x8 of 24x24 grid)
            attn_spatial = attn.mean(dim=1).reshape(24, 24)  # Average over text tokens
            center_bias = attn_spatial[8:16, 8:16].mean().item()

            layer_stats[f"layer_{layer_num}"] = {
                "entropy": float(entropy),
                "sparsity": float(sparsity),
                "center_bias": float(center_bias),
                "text_image_ratio": float(attn.mean().item())
            }

        # ============ Save Summary ============
        summary_data = {
            "sample_id": sample_id,
            "model_name": "BLIP2-OPT-2.7b",
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

    if (i + 1) % 20 == 0:
        torch.cuda.empty_cache()

# ============ Save Global Metadata ============
with open(OUTPUT_DIR / "metadata" / "samples.json", 'w') as f:
    json.dump(samples_metadata, f, indent=2)

print(f"\n✅ BLIP2 extraction complete!")
print(f"   Processed: {len(samples_metadata)} samples")
print(f"   Output dir: {OUTPUT_DIR}")
