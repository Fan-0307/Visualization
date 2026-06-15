#!/usr/bin/env python3
"""Shared utilities for per-token visual attention exporters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw_data" / "train(1)" / "train" / "vqa_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "token_level_output"


def add_common_args(parser: argparse.ArgumentParser, default_model_path: Path) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--layers",
        default="last",
        help="Layers whose heatmaps are saved: first,middle,last or zero-based indices.",
    )
    parser.add_argument(
        "--save-all-layers",
        action="store_true",
        help="Save a heatmap for every layer. Overrides --layers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples already present in results.jsonl.",
    )


def load_samples(data_dir: Path, limit: int = 0) -> list[dict[str, Any]]:
    questions = json.loads(
        (data_dir / "v2_OpenEnded_mscoco_val2014_questions.json").read_text()
    )["questions"]
    annotations = {
        row["question_id"]: row
        for row in json.loads(
            (data_dir / "v2_mscoco_val2014_annotations.json").read_text()
        )["annotations"]
    }
    by_id = {row["question_id"]: row for row in questions}
    ids_path = data_dir / "stratified_sample_ids.json"
    wanted = json.loads(ids_path.read_text()).get("question_ids", []) if ids_path.exists() else []
    if not wanted:
        wanted = list(by_id)

    samples: list[dict[str, Any]] = []
    image_dir = data_dir / "val2014"
    for question_id in wanted:
        question = by_id.get(question_id)
        annotation = annotations.get(question_id)
        if not question or not annotation:
            continue
        image_path = image_dir / f"COCO_val2014_{question['image_id']:012d}.jpg"
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            print(f"[skip] unreadable image: {image_path}")
            continue
        samples.append(
            {
                "sample_id": str(question_id),
                "image_id": str(question["image_id"]),
                "image_path": image_path,
                "question": question["question"],
                "ground_truth": annotation["multiple_choice_answer"],
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


def normalize_grid(grid: Any) -> np.ndarray:
    array = np.asarray(grid, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array -= float(array.min())
    maximum = float(array.max())
    return array / maximum if maximum > 0 else array


def resize_grid(grid: Any, size: int = 14) -> np.ndarray:
    array = normalize_grid(grid)
    image = Image.fromarray(np.uint8(array * 255), mode="L")
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def vector_to_grid(values: Any, grid_h: int | None = None, grid_w: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if grid_h and grid_w:
        needed = int(grid_h * grid_w)
        if len(vector) >= needed:
            return vector[:needed].reshape(int(grid_h), int(grid_w))

    side = int(np.sqrt(len(vector)))
    if side * side == len(vector):
        return vector.reshape(side, side)
    if side * side == len(vector) - 1:
        return vector[1:].reshape(side, side)
    if side > 0:
        return vector[: side * side].reshape(side, side)
    return np.zeros((1, 1), dtype=np.float32)


def safe_token_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (name or "token")[:36]


def save_overlay(image: Image.Image, grid: Any, path: Path) -> None:
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(normalize_grid(grid) * 255), mode="L").resize(
        base.size, Image.Resampling.BILINEAR
    )
    color = Image.new("RGB", base.size, (255, 48, 0))
    overlay = Image.composite(color, base, heat)
    Image.blend(base, overlay, 0.45).save(path)


def completed_sample_ids(model_dir: Path) -> set[str]:
    path = model_dir / "results.jsonl"
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            completed.add(str(json.loads(line)["sample_id"]))
    return completed


def normalized_distribution(weights: Any, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None) + epsilon
    return values / values.sum()


def layer_difference(left: Any, right: Any) -> dict[str, float]:
    left_dist = normalized_distribution(resize_grid(left, 14))
    right_dist = normalized_distribution(resize_grid(right, 14))
    left_to_right = float(np.sum(left_dist * np.log(left_dist / right_dist)))
    right_to_left = float(np.sum(right_dist * np.log(right_dist / left_dist)))
    return {
        "l1_sum": float(np.sum(np.abs(left_dist - right_dist))),
        "kl_left_to_right": left_to_right,
        "kl_right_to_left": right_to_left,
        "symmetric_kl": (left_to_right + right_to_left) / 2.0,
    }


def selected_layer_indices(layer_count: int, layers: str, save_all: bool) -> set[int]:
    if save_all:
        return set(range(layer_count))
    selected: set[int] = set()
    aliases = {
        "first": 0,
        "middle": max(0, (layer_count - 1) // 2),
        "last": max(0, layer_count - 1),
    }
    for value in layers.split(","):
        value = value.strip().lower()
        if not value:
            continue
        if value in aliases:
            selected.add(aliases[value])
            continue
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"Unsupported layer selector: {value}") from error
        if index < 0:
            index += layer_count
        if 0 <= index < layer_count:
            selected.add(index)
    return selected


def write_sample(
    model_dir: Path,
    sample: dict[str, Any],
    prediction: str,
    token_type: str,
    attention_method: str,
    tokens: list[dict[str, Any]],
    layers: str = "last",
    save_all_layers: bool = False,
) -> dict[str, Any]:
    sample_dir = model_dir / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(sample["image_path"]).convert("RGB")

    serializable_tokens: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        token_text = str(token.get("token", ""))
        token_dir = sample_dir / f"token_{index:03d}_{safe_token_name(token_text)}"
        token_dir.mkdir(parents=True, exist_ok=True)
        for old_heatmap in token_dir.glob("layer_*.png"):
            old_heatmap.unlink()
        all_layers = token.get("layers", [])
        selected_indices = selected_layer_indices(
            len(all_layers), layers, save_all_layers
        )
        differences = []
        for layer_index in range(1, len(all_layers)):
            differences.append(
                {
                    "left_layer_index": layer_index - 1,
                    "right_layer_index": layer_index,
                    "left_source_layer_index": all_layers[layer_index - 1].get(
                        "source_layer_index", layer_index - 1
                    ),
                    "right_source_layer_index": all_layers[layer_index].get(
                        "source_layer_index", layer_index
                    ),
                    **layer_difference(
                        all_layers[layer_index - 1]["weights"],
                        all_layers[layer_index]["weights"],
                    ),
                }
            )
        serializable_layers: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(all_layers):
            if layer_index not in selected_indices:
                continue
            grid14 = resize_grid(layer["weights"], 14)
            source_layer_index = layer.get("source_layer_index", layer_index)
            image_path = token_dir / f"layer_{source_layer_index:03d}.png"
            save_overlay(image, grid14, image_path)
            serializable_layers.append(
                {
                    "layer_index": layer_index,
                    "source_layer_index": source_layer_index,
                    "attention_image": str(image_path),
                    "weights": grid14.astype(float).tolist(),
                }
            )
        serializable_tokens.append(
            {
                "token_index": index,
                "token_id": token.get("token_id"),
                "token": token_text,
                "available_layer_count": len(all_layers),
                "layer_count": len(serializable_layers),
                "layers": serializable_layers,
                "adjacent_layer_differences": differences,
            }
        )

    row = {
        "sample_id": sample["sample_id"],
        "image_id": sample["image_id"],
        "image_path": str(sample["image_path"]),
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prediction": prediction,
        "token_type": token_type,
        "attention_method": attention_method,
        "saved_layer_selector": "all" if save_all_layers else layers,
        "token_count": len(serializable_tokens),
        "max_layer_count": max(
            (token["layer_count"] for token in serializable_tokens), default=0
        ),
        "tokens": serializable_tokens,
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (model_dir / "results.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def write_summary(
    model_dir: Path,
    model_name: str,
    token_type: str,
    attention_method: str,
) -> None:
    results_path = model_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ] if results_path.exists() else []
    summary = {
        "status": "ok",
        "model": model_name,
        "sample_count": len(rows),
        "token_count": sum(row.get("token_count", 0) for row in rows),
        "token_layer_heatmap_count": sum(
            sum(token.get("layer_count", 0) for token in row.get("tokens", []))
            for row in rows
        ),
        "token_type": token_type,
        "attention_method": attention_method,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)
