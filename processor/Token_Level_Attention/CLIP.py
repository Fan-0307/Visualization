#!/usr/bin/env python3
"""Export CLIP question-token to image-patch similarity heatmaps."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
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

MODEL_NAME = "CLIP"
TOKEN_TYPE = "question_text_token"
ATTENTION_METHOD = "clip_layerwise_projected_token_patch_cosine_similarity_not_cross_attention"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/openai/clip-vit-base-patch32")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import CLIPModel, CLIPProcessor

    args = parse_args()
    model_dir = args.output_root / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
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
        inputs = processor(
            text=[sample["question"]],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)
        if dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            text_output = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            vision_output = model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
            )

        layer_similarities = []
        layer_count = min(len(text_output.hidden_states), len(vision_output.hidden_states))
        for layer_index in range(1, layer_count):
            text_features = F.normalize(
                model.text_projection(text_output.hidden_states[layer_index]).float(),
                dim=-1,
            )
            patch_features = F.normalize(
                model.visual_projection(
                    vision_output.hidden_states[layer_index][:, 1:]
                ).float(),
                dim=-1,
            )
            layer_similarities.append(
                torch.einsum("btd,bpd->btp", text_features, patch_features)[0]
            )
        input_ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0].bool()
        special_ids = set(processor.tokenizer.all_special_ids)
        tokens = []
        for position in torch.where(mask)[0].tolist():
            token_id = int(input_ids[position])
            if token_id in special_ids:
                continue
            tokens.append(
                {
                    "token_id": token_id,
                    "token": decode_token(processor.tokenizer, token_id),
                    "layers": [
                        {
                            "source_layer_index": layer_index,
                            "weights": vector_to_grid(
                                similarities[position].detach().cpu().numpy()
                            ),
                        }
                        for layer_index, similarities in enumerate(
                            layer_similarities, start=1
                        )
                    ],
                }
            )
        write_sample(
            model_dir, sample, "", TOKEN_TYPE, ATTENTION_METHOD, tokens,
            args.layers, args.save_all_layers,
        )

    write_summary(model_dir, MODEL_NAME, TOKEN_TYPE, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
