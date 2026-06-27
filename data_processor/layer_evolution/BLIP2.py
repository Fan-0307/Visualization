#!/usr/bin/env python3
"""Export BLIP2 whole-model layer-wise raw attention heatmaps."""

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

MODEL_NAME = "BLIP2"
MODEL_OUTPUT_DIR = "blip"
ATTENTION_METHOD = "blip2_whole_model_raw_attentions"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/Salesforce/blip2-opt-2.7b")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    args = parse_args()
    model_dir = args.output_root / MODEL_OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and (model_dir / "results.jsonl").exists():
        (model_dir / "results.jsonl").unlink()
    done = set() if args.overwrite else completed_sample_ids(model_dir)
    samples = [s for s in load_samples(args.data_dir, args.limit) if s["sample_id"] not in done]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(args.model_path, local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    for sample in tqdm(samples, desc=MODEL_NAME):
        image = Image.open(sample["image_path"]).convert("RGB")
        prompt = f"Question: {sample['question']} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        if dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            prediction = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
            prediction = prediction.replace(prompt, "").strip().split("\n")[0]
            if generated.shape[1] >= inputs["input_ids"].shape[1] and torch.equal(generated[:, : inputs["input_ids"].shape[1]], inputs["input_ids"]):
                full_ids = generated
            else:
                full_ids = torch.cat([inputs["input_ids"], generated], dim=1)
            full_mask = torch.ones_like(full_ids)
            full_output = model(
                pixel_values=inputs["pixel_values"],
                input_ids=full_ids,
                attention_mask=full_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )

        layer_heatmaps = []
        vision_attentions = list(full_output.vision_outputs.attentions or [])
        for layer_index, attention in enumerate(vision_attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            visual_attention = selected_attention[0, 1:]
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "full_output.vision_outputs.attentions", len(vision_attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "vision_encoder",
                "source_layer_index": layer_index,
                "attention_source": "full_output.vision_outputs.attentions",
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

        qformer_attentions = list(full_output.qformer_outputs.attentions or [])
        for layer_index, attention in enumerate(qformer_attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            query_attention = selected_attention.mean(dim=0)
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "full_output.qformer_outputs.attentions", len(qformer_attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "qformer_self_attention",
                "source_layer_index": layer_index,
                "attention_source": "full_output.qformer_outputs.attentions",
                "visual_representation": "qformer_query_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": int(selected_attention.shape[0]),
                "visual_token_count": int(selected_attention.shape[-1]),
                "visual_grid_shape": None,
                "token_aggregation": "mean_over_qformer_query_token_queries",
                "weights": vector_to_grid(query_attention.detach().float().cpu().numpy()),
            })

        cross_attentions = [a for a in (full_output.qformer_outputs.cross_attentions or []) if a is not None]
        for layer_index, attention in enumerate(cross_attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            visual_attention = selected_attention.mean(dim=0)
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "full_output.qformer_outputs.cross_attentions", len(cross_attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "qformer_cross_attention",
                "source_layer_index": layer_index,
                "attention_source": "full_output.qformer_outputs.cross_attentions",
                "visual_representation": "vision_encoder_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": int(selected_attention.shape[0]),
                "visual_token_count": int(selected_attention.shape[-1]),
                "visual_grid_shape": None,
                "token_aggregation": "mean_over_qformer_query_token_queries",
                "weights": vector_to_grid(visual_attention.detach().float().cpu().numpy()),
            })

        language_attentions = list(full_output.language_model_outputs.attentions or [])
        query_prefix_count = int(model.query_tokens.shape[1])
        prompt_length = int(inputs["input_ids"].shape[1])
        answer_query_positions = list(range(query_prefix_count + prompt_length, query_prefix_count + full_ids.shape[1]))
        for layer_index, attention in enumerate(language_attentions):
            selected_attention, head_mode, selected_head = select_attention_heads(attention, args.head)
            valid = [p for p in answer_query_positions if p < selected_attention.shape[0]]
            if valid:
                prefix_attention = selected_attention[valid, :query_prefix_count].mean(dim=0)
                token_aggregation = "mean_over_generated_answer_token_queries"
                query_token_count = len(valid)
            else:
                prefix_attention = selected_attention[:, :query_prefix_count].mean(dim=0)
                token_aggregation = "mean_over_language_token_queries"
                query_token_count = int(selected_attention.shape[0])
            tensor_shape = list(attention.shape)
            print(attention_debug_line(MODEL_NAME, sample["sample_id"], "full_output.language_model_outputs.attentions", len(language_attentions), layer_index, selected_head, tensor_shape))
            layer_heatmaps.append({
                "component": "language_model",
                "source_layer_index": layer_index,
                "attention_source": "full_output.language_model_outputs.attentions",
                "visual_representation": "language_query_prefix_tokens",
                "attention_tensor_shape": tensor_shape,
                "head_mode": head_mode,
                "selected_head": selected_head,
                "query_token_count": query_token_count,
                "visual_token_count": query_prefix_count,
                "visual_grid_shape": None,
                "token_aggregation": token_aggregation,
                "weights": vector_to_grid(prefix_attention.detach().float().cpu().numpy()),
            })

        write_layer_sample(model_dir, sample, prediction, ATTENTION_METHOD, layer_heatmaps, "all", args.save_all_layers, args.target_layer_count)

    write_summary(model_dir, MODEL_NAME, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
