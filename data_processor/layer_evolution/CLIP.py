#!/usr/bin/env python3
"""Export CLIP layer-wise raw vision self-attention heatmaps."""

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

MODEL_NAME = "CLIP"
MODEL_OUTPUT_DIR = "clip"
ATTENTION_METHOD = "clip_vision_self_attention_cls_to_patch_tokens"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/openai/clip-vit-base-patch32")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import CLIPModel, CLIPProcessor

    args = parse_args()
    model_dir = args.output_root / MODEL_OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (model_dir / "results.jsonl").exists():
        (model_dir / "results.jsonl").unlink()
    done = set() if args.overwrite else completed_sample_ids(model_dir)
    samples = [s for s in load_samples(args.data_dir, args.limit) if s["sample_id"] not in done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = CLIPProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    for sample in tqdm(samples, desc=MODEL_NAME):
        image = Image.open(sample["image_path"]).convert("RGB")
        inputs = processor(text=[sample["question"]], images=image, return_tensors="pt", padding=True).to(device)
        if dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            vision_output = model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
                return_dict=True,
            )

        layer_heatmaps = []
        attentions = list(vision_output.attentions or [])
        for layer_index, attention in enumerate(attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            patch_attention = selected_attention[0, 1:]
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "vision_output.attentions", len(attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "vision_encoder",
                "source_layer_index": layer_index,
                "attention_source": "vision_output.attentions",
                "visual_representation": "vision_patch_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": 1,
                "visual_token_count": int(patch_attention.numel()),
                "visual_grid_shape": None,
                "token_aggregation": "cls_visual_token_query_to_patch_keys",
                "weights": vector_to_grid(patch_attention.detach().float().cpu().numpy()),
            })
        write_layer_sample(model_dir, sample, "", ATTENTION_METHOD, layer_heatmaps, "all", args.save_all_layers, args.target_layer_count)

    write_summary(model_dir, MODEL_NAME, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
