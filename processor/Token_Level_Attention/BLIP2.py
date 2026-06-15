#!/usr/bin/env python3
"""Export BLIP2 Q-Former query-token cross-attention heatmaps."""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from common import (
    add_common_args,
    completed_sample_ids,
    load_samples,
    vector_to_grid,
    write_sample,
    write_summary,
)

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_NAME = "BLIP2"
TOKEN_TYPE = "qformer_learned_query_token"
ATTENTION_METHOD = "blip2_layerwise_qformer_cross_attention_mean_heads"
DEFAULT_MODEL_PATH = Path("/home/jovyan/softwares/hf_model/Salesforce/blip2-opt-2.7b")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main():
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    args = parse_args()
    model_dir = args.output_root / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
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

            image_embeds = model.vision_model(inputs["pixel_values"]).last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            qformer_output = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        cross_attentions = [
            (layer_index, attention[0].float().mean(dim=0))
            for layer_index, attention in enumerate(qformer_output.cross_attentions)
            if attention is not None
        ]
        tokens = []
        query_count = cross_attentions[0][1].shape[0] if cross_attentions else 0
        for query_index in range(query_count):
            tokens.append(
                {
                    "token_id": query_index,
                    "token": f"<query_{query_index:02d}>",
                    "layers": [
                        {
                            "source_layer_index": layer_index,
                            "weights": vector_to_grid(
                                attention[query_index].cpu().numpy()
                            ),
                        }
                        for layer_index, attention in cross_attentions
                    ],
                }
            )
        write_sample(
            model_dir, sample, prediction, TOKEN_TYPE, ATTENTION_METHOD, tokens,
            args.layers, args.save_all_layers,
        )

    write_summary(model_dir, MODEL_NAME, TOKEN_TYPE, ATTENTION_METHOD)


if __name__ == "__main__":
    main()
