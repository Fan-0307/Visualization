#!/usr/bin/env python3
"""Export LLaVA generated-token attention over visual tokens."""

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

MODEL_NAME = "LLaVA"
TOKEN_TYPE = "generated_answer_token"
ATTENTION_METHOD = "llava_layerwise_llm_generated_token_to_visual_tokens_mean_heads"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/llava-hf/llava-1.5-7b-hf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    args = parse_args()
    model_dir = args.output_root / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
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
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated[:, prompt_length:]
        full_ids = generated
        full_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            output = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = [
            attention[0].float().mean(dim=0)
            for attention in output.attentions
        ]
        visual_positions = torch.where(full_ids[0] == image_token_id)[0]
        if len(visual_positions) == 1 and attentions[0].shape[-1] > full_ids.shape[1]:
            expanded_visual_count = attentions[0].shape[-1] - full_ids.shape[1] + 1
            start = int(visual_positions[0])
            visual_positions = torch.arange(
                start, start + expanded_visual_count, device=attentions[0].device
            )

        tokens = []
        query_offset = attentions[0].shape[0] - full_ids.shape[1]
        for index, token_id in enumerate(generated_ids[0].tolist()):
            query_position = prompt_length + index + query_offset
            if query_position >= attentions[0].shape[0]:
                break
            tokens.append(
                {
                    "token_id": int(token_id),
                    "token": decode_token(processor.tokenizer, token_id),
                    "layers": [
                        {
                            "source_layer_index": layer_index,
                            "weights": vector_to_grid(
                                attention[query_position, visual_positions].cpu().numpy()
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
