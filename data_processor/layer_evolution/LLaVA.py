#!/usr/bin/env python3
"""Export LLaVA whole-model layer-wise raw attention heatmaps."""

import argparse
import os
from pathlib import Path

import torch
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

MODEL_NAME = "LLaVA"
MODEL_OUTPUT_DIR = "llava"
ATTENTION_METHOD = "llava_whole_model_raw_attentions"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/llava-hf/llava-1.5-7b-hf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    args = parse_args()
    model_dir = args.output_root / MODEL_OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (model_dir / "results.jsonl").exists():
        (model_dir / "results.jsonl").unlink()
    done = set() if args.overwrite else completed_sample_ids(model_dir)
    samples = [s for s in load_samples(args.data_dir, args.limit) if s["sample_id"] not in done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    image_token_id = model.config.image_token_index
    for sample in tqdm(samples, desc=MODEL_NAME):
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt = f"USER: <image>\nAnswer the question with a short phrase. {sample['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        if dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            vision_output = model.vision_tower(inputs["pixel_values"], output_attentions=True, return_dict=True)
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_ids = generated[:, prompt_length:]
        full_ids = generated
        full_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            output = model(input_ids=full_ids, attention_mask=full_mask, pixel_values=inputs["pixel_values"], output_attentions=True, use_cache=False, return_dict=True)

        layer_heatmaps = []
        vision_attentions = list(vision_output.attentions or [])
        for layer_index, attention in enumerate(vision_attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            visual_attention = selected_attention[0, 1:]
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "vision_output.attentions", len(vision_attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "vision_tower",
                "source_layer_index": layer_index,
                "attention_source": "vision_output.attentions",
                "visual_representation": "vision_patch_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": 1,
                "visual_token_count": int(visual_attention.numel()),
                "visual_grid_shape": None,
                "token_aggregation": "cls_visual_token_query_to_patch_keys",
                "weights": vector_to_grid(visual_attention.detach().float().cpu().numpy()),
            })

        attentions = list(output.attentions or [])
        visual_positions = torch.where(full_ids[0] == image_token_id)[0]
        if len(visual_positions) == 1 and attentions and attentions[0].shape[-1] > full_ids.shape[1]:
            expanded_visual_count = attentions[0].shape[-1] - full_ids.shape[1] + 1
            start = int(visual_positions[0])
            visual_positions = torch.arange(start, start + expanded_visual_count, device=attentions[0].device)
        query_offset = attentions[0].shape[-2] - full_ids.shape[1] if attentions else 0
        query_positions = [prompt_length + i + query_offset for i, _ in enumerate(generated_ids[0].tolist())]
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
                "visual_representation": "projected_vision_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": len(valid),
                "visual_token_count": int(len(visual_positions)),
                "visual_grid_shape": None,
                "token_aggregation": "mean_over_generated_answer_token_queries",
                "weights": vector_to_grid(visual_attention.float().cpu().numpy()),
            })
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        write_layer_sample(model_dir, sample, prediction, ATTENTION_METHOD, layer_heatmaps, "all", args.save_all_layers, args.target_layer_count)

    write_summary(model_dir, MODEL_NAME, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
