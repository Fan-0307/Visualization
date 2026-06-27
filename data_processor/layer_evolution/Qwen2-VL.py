#!/usr/bin/env python3
"""Export Qwen2-VL whole-model layer-wise raw attention heatmaps."""

import argparse
import copy
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from common import (
    add_common_args,
    attention_debug_line,
    completed_sample_ids,
    load_samples,
    select_attention_heads,
    vector_to_grid,
    write_layer_sample,
    write_summary,
)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "Qwen2-VL"
MODEL_OUTPUT_DIR = "qwen"
ATTENTION_METHOD = "qwen2_vl_whole_model_raw_attentions"
DEFAULT_MODEL_PATH = Path("/home/jovyan/fan/hf_model/Qwen2_VL/Qwen2-VL-7B-Instruct")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def collect_vision_attention_heatmaps(model, pixel_values, image_grid_thw, head, sample_id):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    _, grid_h, grid_w = image_grid_thw[0].detach().cpu().tolist()
    layer_heatmaps = []
    original_forwards = []

    def make_forward(module, original_forward, layer_index):
        def wrapped_forward(*args, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            cu_seqlens = kwargs["cu_seqlens"] if "cu_seqlens" in kwargs else args[1]
            rotary_pos_emb = kwargs.get("rotary_pos_emb", args[2] if len(args) > 2 else None)
            seq_length = hidden_states.shape[0]
            num_heads = int(module.num_heads)
            head_dim = hidden_states.shape[-1] // num_heads
            q, k, _v = module.qkv(hidden_states).reshape(seq_length, 3, num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
            attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
            for index in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[index - 1]:cu_seqlens[index], cu_seqlens[index - 1]:cu_seqlens[index]] = 0
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights + attention_mask, dim=-1, dtype=torch.float32).to(q.dtype)
            if head >= 0:
                if head >= attn_weights.shape[0]:
                    raise ValueError(f"Selected head {head} out of range for {attn_weights.shape[0]} heads")
                selected_attention, head_mode, selected_head = attn_weights[head], "raw_head", head
            else:
                selected_attention, head_mode, selected_head = attn_weights.mean(dim=0), "head_mean", "all"
            visual_attention = selected_attention.mean(dim=0)
            tensor_shape = [1, int(attn_weights.shape[0]), int(seq_length), int(seq_length)]
            print(attention_debug_line(MODEL_NAME, sample_id, "model.visual.blocks[*].attn", len(model.visual.blocks), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "vision_encoder",
                "source_layer_index": layer_index,
                "attention_source": "model.visual.blocks[*].attn",
                "visual_representation": "vision_patch_tokens_before_merger",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": int(seq_length),
                "visual_token_count": int(seq_length),
                "visual_grid_shape": [int(grid_h), int(grid_w)],
                "token_aggregation": "mean_over_vision_token_queries",
                "weights": vector_to_grid(visual_attention.detach().float().cpu().numpy(), int(grid_h), int(grid_w)),
            })
            return original_forward(*args, **kwargs)
        return wrapped_forward

    for layer_index, block in enumerate(model.visual.blocks):
        original_forwards.append((block.attn, block.attn.forward))
        block.attn.forward = make_forward(block.attn, block.attn.forward, layer_index)
    try:
        with torch.no_grad():
            model.visual(pixel_values.type(model.visual.get_dtype()), grid_thw=image_grid_thw)
    finally:
        for module, original_forward in original_forwards:
            module.forward = original_forward
    return layer_heatmaps


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    model_dir = args.output_root / MODEL_OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (model_dir / "results.jsonl").exists():
        (model_dir / "results.jsonl").unlink()
    done = set() if args.overwrite else completed_sample_ids(model_dir)
    samples = [s for s in load_samples(args.data_dir, args.limit) if s["sample_id"] not in done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        max_memory={0: "22GiB", "cpu": "110GiB"},
        attn_implementation="eager",
    ).eval()
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None

    image_token_id = model.config.image_token_id
    for sample in tqdm(samples, desc=MODEL_NAME):
        image = Image.open(sample["image_path"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"Answer with a short phrase: {sample['question']}"}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, generation_config=generation_config)
        layer_heatmaps = collect_vision_attention_heatmaps(model, inputs["pixel_values"], inputs["image_grid_thw"], args.head, sample["sample_id"])
        generated_ids = generated[:, prompt_length:]
        full_ids = generated
        full_mask = torch.ones_like(full_ids)
        if hasattr(model, "rope_deltas"):
            model.rope_deltas = None
        with torch.no_grad():
            output = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = list(output.attentions or [])
        visual_positions = torch.where(full_ids[0] == image_token_id)[0]
        _, grid_h, grid_w = inputs["image_grid_thw"][0].detach().cpu().tolist()
        merge_size = int(model.config.vision_config.spatial_merge_size)
        merged_h, merged_w = int(grid_h) // merge_size, int(grid_w) // merge_size
        query_positions = [prompt_length + index for index, _ in enumerate(generated_ids[0].tolist())]
        for layer_index, attention in enumerate(attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            valid = [p for p in query_positions if p < selected_attention.shape[0]]
            if not valid or len(visual_positions) == 0:
                continue
            visual_attention = selected_attention[valid][:, visual_positions].mean(dim=0)
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "output.attentions", len(attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "language_model",
                "source_layer_index": layer_index,
                "attention_source": "output.attentions",
                "visual_representation": "merged_vision_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": len(valid),
                "visual_token_count": int(len(visual_positions)),
                "visual_grid_shape": [merged_h, merged_w],
                "token_aggregation": "mean_over_generated_answer_token_queries",
                "weights": vector_to_grid(visual_attention.float().cpu().numpy(), merged_h, merged_w),
            })
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        write_layer_sample(model_dir, sample, prediction, ATTENTION_METHOD, layer_heatmaps, "all", args.save_all_layers, args.target_layer_count)

    write_summary(model_dir, MODEL_NAME, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
