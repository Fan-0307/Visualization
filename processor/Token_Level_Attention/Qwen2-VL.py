#!/usr/bin/env python3
"""Export Qwen2-VL generated-token attention over visual tokens."""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from common import (
    add_common_args,
    completed_sample_ids,
    decode_token,
    load_samples,
    vector_to_grid,
    write_sample,
    write_summary,
)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "Qwen2-VL"
TOKEN_TYPE = "generated_answer_token"
ATTENTION_METHOD = "qwen2_vl_layerwise_llm_generated_token_to_visual_tokens_mean_heads"
DEFAULT_MODEL_PATH = Path("/home/jovyan/fan/hf_model/Qwen2_VL/Qwen2-VL-7B-Instruct")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    model_dir = args.output_root / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
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

    image_token_id = model.config.image_token_id
    for sample in tqdm(samples, desc=MODEL_NAME):
        image = Image.open(sample["image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Answer with a short phrase: {sample['question']}"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated[:, prompt_length:]
        full_ids = generated
        full_mask = torch.ones_like(full_ids)
        mm_token_type_ids = torch.zeros_like(full_ids, dtype=torch.int32)
        mm_token_type_ids[full_ids == image_token_id] = 1
        with torch.no_grad():
            output = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=mm_token_type_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = [
            attention[0].float().mean(dim=0)
            for attention in output.attentions
        ]
        visual_positions = torch.where(full_ids[0] == image_token_id)[0]
        _, grid_h, grid_w = inputs["image_grid_thw"][0].detach().cpu().tolist()
        merge_size = int(model.config.vision_config.spatial_merge_size)
        merged_h, merged_w = int(grid_h) // merge_size, int(grid_w) // merge_size
        tokens = []
        for index, token_id in enumerate(generated_ids[0].tolist()):
            query_position = prompt_length + index
            tokens.append(
                {
                    "token_id": int(token_id),
                    "token": decode_token(processor.tokenizer, token_id),
                    "layers": [
                        {
                            "source_layer_index": layer_index,
                            "weights": vector_to_grid(
                                attention[query_position, visual_positions].cpu().numpy(),
                                merged_h,
                                merged_w,
                            ),
                        }
                        for layer_index, attention in enumerate(attentions)
                    ],
                }
            )
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        write_sample(
            model_dir, sample, prediction, TOKEN_TYPE, ATTENTION_METHOD, tokens,
            args.layers, args.save_all_layers,
        )

    write_summary(model_dir, MODEL_NAME, TOKEN_TYPE, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
