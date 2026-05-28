"""
CLIP Attention Extraction Script
按照 Comparative VLM Attention Analysis System 规范输出
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from pathlib import Path
from local_vqa_loader import load_local_vqa_dataset

# ============ Configuration ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
OUTPUT_DIR = Path("data")
MODEL_NAME = "clip"
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

# ============ Load CLIP Model ============
print(f"\n=== Loading CLIP (OpenCLIP ViT-B-32) on {DEVICE} ===")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(DEVICE).eval()

# ============ Attention Hook ============
attention_cache = {}

def get_attention_hook(name):
    def hook(module, input, output):
        # output is (output_tensor, attention_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            if attn is not None:
                attention_cache[name] = attn.detach().cpu()
    return hook

# Register hooks for last 4 layers (vision transformer)
num_layers = len(clip_model.visual.transformer.resblocks)
for layer_idx in range(num_layers - 4, num_layers):
    clip_model.visual.transformer.resblocks[layer_idx].attn.register_forward_hook(
        get_attention_hook(f"layer_{layer_idx}")
    )

# ============ Process Samples ============
samples_metadata = []

print(f"\n=== Processing {NUM_SAMPLES} samples ===")
for i, sample in enumerate(tqdm(samples, desc="CLIP Extraction")):
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

        # Prepare inputs - 同时输入图片和文字
        image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
        text_input = clip_tokenizer([question]).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)

            # Compute similarity for prediction
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            confidence = (similarity + 1) / 2  # Normalize to [0, 1]

        # Simple prediction logic (CLIP doesn't generate text, so we use similarity)
        prediction = "yes" if confidence > 0.5 else "no"
        correct = (prediction.lower() == gt_answer.lower())

        # ============ Save Prediction ============
        prediction_data = {
            "sample_id": sample_id,
            "model_name": "CLIP-ViT-B32",
            "prediction": prediction,
            "confidence": float(confidence),
            "correct": correct,
            "inference_time_ms": 0,  # TODO: measure if needed
            "tokens": question.split(),
            "generated_text": None,
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
            # We need: [num_heads, text_tokens, image_patches]
            # For CLIP vision: seq_len = 1 (CLS) + 49 (patches for 224x224 with patch_size=32)

            attn = attn_tensor[0]  # Remove batch dim: [num_heads, seq_len, seq_len]

            # Extract cross-attention between CLS token and patches
            # Shape: [num_heads, 50, 50] -> take [num_heads, 1, 49] (CLS attending to patches)
            if attn.shape[-1] >= 50:
                cross_attn = attn[:, 0, 1:50]  # [num_heads, 49]
                # Expand to match spec: [num_heads, text_tokens, image_patches]
                # For CLIP we'll use 1 "text token" (the CLS) attending to 49 patches
                # Reshape to 7x7 grid then interpolate to 24x24
                num_heads = cross_attn.shape[0]
                cross_attn_2d = cross_attn.reshape(num_heads, 7, 7)

                # Interpolate to 24x24 = 576 patches
                cross_attn_resized = torch.nn.functional.interpolate(
                    cross_attn_2d.unsqueeze(1),  # [num_heads, 1, 7, 7]
                    size=(24, 24),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [num_heads, 24, 24]

                cross_attn_flat = cross_attn_resized.reshape(num_heads, 576)

                # Expand to [num_heads, text_tokens, image_patches]
                num_text_tokens = len(question.split())
                final_attn = cross_attn_flat.unsqueeze(1).expand(num_heads, num_text_tokens, 576)

                # Save as float16
                attn_array = final_attn.numpy().astype(np.float16)
                layer_num = layer_name.split('_')[1]
                np.save(attn_dir / f"layer_{layer_num}.npy", attn_array)

                # Compute statistics
                entropy = -torch.sum(cross_attn_flat * torch.log(cross_attn_flat + 1e-10), dim=-1).mean().item()
                sparsity = (cross_attn_flat < 0.01).float().mean().item()
                center_bias = cross_attn_2d[:, 8:16, 8:16].mean().item()  # Center region

                layer_stats[f"layer_{layer_num}"] = {
                    "entropy": float(entropy),
                    "sparsity": float(sparsity),
                    "center_bias": float(center_bias),
                    "text_image_ratio": 0.0  # CLIP doesn't have explicit text-image ratio
                }

        # ============ Save Summary ============
        summary_data = {
            "sample_id": sample_id,
            "model_name": "CLIP-ViT-B32",
            "layer_statistics": layer_stats,
            "object_coverage": {},  # TODO: compute if bbox available
            "dominant_regions": [],
            "attention_behavior": "uniform"  # TODO: classify
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
        continue

    if (i + 1) % 20 == 0:
        torch.cuda.empty_cache()

# ============ Save Global Metadata ============
with open(OUTPUT_DIR / "metadata" / "samples.json", 'w') as f:
    json.dump(samples_metadata, f, indent=2)

print(f"\n✅ CLIP extraction complete!")
print(f"   Processed: {len(samples_metadata)} samples")
print(f"   Output dir: {OUTPUT_DIR}")
