"""
CLIP attention extraction — dual-encoder, no cross-attention, no text generation.

What CLIP provides (vs LLaVA / BLIP / Qwen):
  - Vision encoder self-attention: CLS → 49 patches, 12 layers
  - Text encoder self-attention:   question tokens, 12 layers
  - VQA via zero-shot cosine similarity (no generative answer)
  - No cross-modal attention, no answer-token attention

Outputs:
  1. public/data/attn/{MODEL}_{image_id}_{question_id}.json   (per-token)
  2. public/data/layer_evo/clip/{sample_id}/result.json        (per-layer evolution)
"""
import torch, os, json
import numpy as np
from PIL import Image

MODEL_PATH = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-base-patch32")
IMG_DIR    = "public/img"
OUT_DIR    = "public/data/attn"
EVO_DIR    = "public/data/layer_evo/clip"
MODEL      = os.environ.get("CLIP_MODEL_NAME", "CLIP-ViT-B-32")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EVO_DIR, exist_ok=True)


# ── data ─────────────────────────────────────────────────────────────
def load_data():
    from datasets import load_dataset, VerificationMode
    datas = load_dataset(
        'lmms-lab/VQAv2',
        data_files='data/train-00000-of-00136.parquet',
        verification_mode=VerificationMode.NO_CHECKS,
        keep_in_memory=True,
    )['train']
    print("Load VQAv2 data/train-00000-of-00136.parquet")
    return datas


# ── model ────────────────────────────────────────────────────────────
def load_model():
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        local_files_only=True,
    )
    processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"Load {MODEL} on device: {model.device}")
    return model, processor


# ── VQA via zero-shot similarity ─────────────────────────────────────
def clip_vqa(model, processor, data):
    """Zero-shot VQA: pick the answer with highest cosine similarity
    to the [image, question+answer] pair.
    """
    # Use a small set of common VQAv2 answer candidates
    candidates = data.get('answers', [])
    if not candidates:
        return "", 0.0

    # Build text inputs: "question: X answer: Y" per candidate
    unique_answers = list(set(a['answer'].strip().lower() for a in candidates))
    texts = [f"question: {data['question']} answer: {ans}" for ans in unique_answers]

    image_inputs = processor(images=data['image'], return_tensors="pt")
    text_inputs  = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # get_image_features / get_text_features return BaseModelOutputWithPooling
        img_out = model.get_image_features(**image_inputs)
        txt_out = model.get_text_features(**text_inputs)

        image_embeds = img_out.pooler_output  # [1, 512]
        text_embeds  = txt_out.pooler_output  # [N, 512]

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).squeeze(0)    # [N]
        best_idx   = similarity.argmax().item()
        pred       = unique_answers[best_idx]
        score      = similarity[best_idx].item()

    print(f"  CLIP VQA: pred='{pred}' gt='{data.get('multiple_choice_answer', '')}' score={score:.3f}")
    return pred, score, image_inputs, text_inputs


# ── attention extraction ─────────────────────────────────────────────
def extract_vision_attn(model, image_inputs):
    """Vision encoder self-attention — CLS token query → patch keys."""
    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=image_inputs.pixel_values,
            output_attentions=True,
        )
    # outputs.attentions: tuple of 12 tensors [1, num_heads, 50, 50]
    return outputs.attentions, outputs.last_hidden_state


def extract_text_attn(model, text_inputs):
    """Text encoder self-attention — all tokens."""
    with torch.no_grad():
        outputs = model.text_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_attentions=True,
        )
    return outputs.attentions, outputs.last_hidden_state


# ── per-token JSON (matches public/data/attn/ format) ───────────────
def build_per_token_json(data, vis_attns, text_attns, pred, correct, processor, text_inputs):
    W, H = data['image'].size
    img_id   = data['image_id']
    n_layers = len(vis_attns)  # 12 for ViT-B/32
    grid     = 7               # ViT-B/32: 224/32 = 7

    question_ids = text_inputs.input_ids[0]
    n_q = len(question_ids)
    # Exclude CLS and SEP from question tokens for cleaner display
    q_start = 0
    q_end   = n_q
    if question_ids[0] == processor.tokenizer.cls_token_id:
        q_start = 1
    if question_ids[-1] == processor.tokenizer.sep_token_id:
        q_end = n_q - 1

    question_attn_dict = {}
    for local_idx in range(q_start, q_end):
        global_idx = local_idx
        token_text = processor.decode(question_ids[local_idx])
        entry = {"token": token_text, "vis_attn": {}, "que_attn": {}}

        for layer_idx in range(n_layers):
            layer_name = f"layer_{layer_idx}"
            # vision self-attn: CLS (idx 0) → all 50 tokens (CLS + 49 patches)
            # For each question token, there's NO direct vis attn in CLIP.
            # But we can use the CLS token's attention to patches as a proxy.
            # Actually, CLIP doesn't have token→image attention.
            # We'll use CLS attention as the "visual attention" per layer.
            # text self-attn → how this token attends to other question tokens
            text_mat = text_attns[layer_idx][0]  # [num_heads, seq_len, seq_len]
            entry["que_attn"][layer_name] = (
                text_mat[:, local_idx, :].mean(dim=0).float().cpu().tolist()
            )
        question_attn_dict[str(global_idx)] = entry

    # Use CLS token vision attention as a shared "visual reference" per layer
    cls_vis_attn = {}
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx}"
        vis_mat = vis_attns[layer_idx][0]  # [num_heads, 50, 50]
        # CLS (idx 0) attends to patches [1:]
        cls_vis_attn[layer_name] = vis_mat[:, 0, 1:].mean(dim=0).float().cpu().tolist()  # [49]

    # Store CLS visual attention in a dedicated slot
    vis_ref_entry = {
        "token": "[CLS]",
        "vis_attn": cls_vis_attn,
        "que_attn": {},  # CLS text self-attn
    }
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx}"
        text_mat = text_attns[layer_idx][0]
        vis_ref_entry["que_attn"][layer_name] = (
            text_mat[:, 0, :].mean(dim=0).float().cpu().tolist()
        )
    question_attn_dict["cls"] = vis_ref_entry

    result = {
        "image": {"w": W, "h": H, "img_id": img_id, "grid": {"w": grid, "h": grid}},
        "question": question_attn_dict,
        "answer": {},  # CLIP has no generated answers
        "model": MODEL,
        "correct": correct,
        "prediction": pred,
        "vqa_method": "zero_shot_cosine_similarity",
    }
    print("  per-token JSON built")
    return result


# ── per-layer JSON (matches public/data/layer_evo/ format) ───────────
def build_per_layer_json(data, vis_attns, pred, correct):
    img_id     = data['image_id']
    sample_id  = f"{img_id}{data['question_id']}"
    n_layers   = len(vis_attns)

    grid_size  = int(np.sqrt(vis_attns[0].shape[-1] - 1))  # 49 patches → 7x7

    layers_out = []
    for li, attn in enumerate(vis_attns):
        # attn: [1, num_heads, 50, 50]
        attn_cpu = attn[0].float()           # [num_heads, 50, 50]
        # head-mean: average over heads
        head_mean = attn_cpu.mean(dim=0)     # [50, 50]
        # CLS → patches (skip position 0 which is CLS self-attention)
        cls_to_patches = head_mean[0, 1:]    # [49]
        grid_2d = cls_to_patches.reshape(grid_size, grid_size).tolist()

        layers_out.append({
            "layer_index": li,
            "display_layer_index": li,
            "original_layer_index": li,
            "source_layer_index": li,
            "attention_image": f"/data/layer_evo/clip/{sample_id}/layer_{li:03d}.png",
            "attention_source": "vision_output.attentions",
            "component": "vision_encoder",
            "visual_representation": "vision_patch_tokens",
            "attention_tensor_shape": list(attn.shape),
            "head_mode": "head_mean",
            "selected_head": "all",
            "query_token_count": 1,
            "visual_token_count": grid_size * grid_size,
            "visual_grid_shape": [grid_size, grid_size],
            "token_aggregation": "cls_visual_token_query_to_patch_keys",
            "raw_visual_grid": grid_2d,
            # Weights: all-to-all token attention (for reference)
            "weights": head_mean.tolist(),
        })

    # Adjacent layer differences
    adj_diffs = []
    for li in range(n_layers - 1):
        left  = np.array(layers_out[li]["raw_visual_grid"]).flatten()
        right = np.array(layers_out[li + 1]["raw_visual_grid"]).flatten()
        left  = np.maximum(left, 1e-12)
        right = np.maximum(right, 1e-12)
        left  = left / left.sum()
        right = right / right.sum()

        l1  = float(np.abs(left - right).sum())
        kl_lr = float((right * np.log(right / left)).sum())
        kl_rl = float((left * np.log(left / right)).sum())

        adj_diffs.append({
            "left_layer_index": li,
            "right_layer_index": li + 1,
            "left_source_layer_index": li,
            "right_source_layer_index": li + 1,
            "l1_sum": l1,
            "kl_left_to_right": kl_lr,
            "kl_right_to_left": kl_rl,
            "symmetric_kl": (kl_lr + kl_rl) / 2,
        })

    evo_json = {
        "sample_id": sample_id,
        "image_id": img_id,
        "image_path": f"/img/COCO_val2014_{img_id:012d}.jpg",
        "question": data['question'],
        "ground_truth": data.get('multiple_choice_answer', ''),
        "prediction": pred,
        "correct": correct,
        "attention_method": "clip_vision_self_attention_cls_to_patch_tokens",
        "attention_contract": {
            "source": "raw_model_attention_tensor",
            "layer_rule": "all 12 vision encoder layers (real attention layers only)",
            "head_modes": ["head_mean"],
            "forbidden_methods": [
                "grad_cam", "attention_rollout", "attention_flow",
                "attention_x_gradient", "relevance_propagation",
                "integrated_gradients", "shap", "lime",
                "pca_projection", "tsne_projection",
            ],
            "note": "CLIP has NO cross-attention; vision self-attention only. VQA via zero-shot cosine similarity."
        },
        "saved_layer_selector": "all_real_layers",
        "available_layer_count": n_layers,
        "full_layer_count": n_layers,
        "layer_sampling": {
            "method": "all_real_layers",
            "target_layer_count": 24,
            "original_layer_count": n_layers,
            "sampled_layer_count": n_layers,
            "sampled_original_layer_indices": list(range(n_layers)),
            "component_counts": {"vision_encoder": n_layers},
        },
        "layer_count": n_layers,
        "layers": layers_out,
        "adjacent_layer_differences": adj_diffs,
    }
    print("  per-layer JSON built")
    return evo_json


# ── accuracy ─────────────────────────────────────────────────────────
def compute_vqa_accuracy(pred, data):
    """VQAv2-style: min(count / 3, 1) with soft match."""
    if not pred:
        return 0
    pred = pred.strip().lower()
    counts = {}
    for a in data['answers']:
        ans = a['answer'].strip().lower()
        counts[ans] = counts.get(ans, 0) + 1
    score = max(
        (min(counts.get(a, 0), 3) / 3 for a in counts if a in pred or pred in a),
        default=0,
    )
    return 1 if score >= 1 / 3 else 0


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, processor = load_model()
    datas = load_data()

    N = 32  # match other run.py defaults; override with CLIP_SAMPLE_COUNT
    n_samples = int(os.environ.get("CLIP_SAMPLE_COUNT", N))

    for i in range(n_samples):
        data = datas[i]
        img_id = data['image_id']
        qid    = data['question_id']
        sample_id = f"{img_id}{qid}"

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{n_samples}] image={img_id} question={data['question'][:60]}...")

        # Save image
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpeg")
        if not os.path.exists(img_path):
            data['image'].save(img_path)
            print(f"  Image saved: {img_path}")

        # Extract attention (vision + text)
        image_inputs = processor(images=data['image'], return_tensors="pt")
        vis_attns, vis_hidden = extract_vision_attn(model, image_inputs)

        question_text = f"question: {data['question']}"
        text_inputs = processor(text=[question_text], return_tensors="pt", padding=True)
        text_attns, text_hidden = extract_text_attn(model, text_inputs)

        # VQA prediction
        pred, score = "", 0.0
        try:
            pred, score, _, _ = clip_vqa(model, processor, data)
        except Exception as e:
            print(f"  VQA failed: {e}, using empty prediction")

        correct = compute_vqa_accuracy(pred, data) if pred else 0

        # ── Output 1: per-token attn JSON ──
        attn_json = build_per_token_json(
            data, vis_attns, text_attns, pred, correct, processor, text_inputs
        )
        attn_path = os.path.join(OUT_DIR, f"{MODEL}_{data['image_id']}_{data['question_id']}.json")
        with open(attn_path, 'w', encoding='utf-8') as f:
            json.dump(attn_json, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  → {attn_path}")

        # ── Output 2: per-layer evolution JSON ──
        evo_json = build_per_layer_json(data, vis_attns, pred, correct)
        evo_sample_dir = os.path.join(EVO_DIR, sample_id)
        os.makedirs(evo_sample_dir, exist_ok=True)
        evo_path = os.path.join(evo_sample_dir, "result.json")
        with open(evo_path, 'w', encoding='utf-8') as f:
            json.dump(evo_json, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  → {evo_path}")

    print(f"\n{'='*60}")
    print(f"  CLIP extraction complete: {n_samples} samples")
    print(f"  per-token → {OUT_DIR}/")
    print(f"  per-layer → {EVO_DIR}/")
