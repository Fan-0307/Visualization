#!/usr/bin/env python3
"""Shared utilities for layer-wise visual attention exporters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "vqa_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "data" / "layer_evolution"
DEFAULT_TARGET_LAYER_COUNT = 24


def add_common_args(parser: argparse.ArgumentParser, default_model_path: Path) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=default_model_path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--head", type=int, default=-1)
    parser.add_argument("--save-all-layers", action="store_true")
    parser.add_argument("--target-layer-count", type=int, default=DEFAULT_TARGET_LAYER_COUNT)
    parser.add_argument("--overwrite", action="store_true")


def load_samples(data_dir: Path, limit: int = 0) -> list[dict[str, Any]]:
    questions = json.loads((data_dir / "v2_OpenEnded_mscoco_val2014_questions.json").read_text())["questions"]
    annotations = {
        row["question_id"]: row
        for row in json.loads((data_dir / "v2_mscoco_val2014_annotations.json").read_text())["annotations"]
    }
    by_id = {row["question_id"]: row for row in questions}
    ids_path = data_dir / "stratified_sample_ids.json"
    wanted = json.loads(ids_path.read_text()).get("question_ids", []) if ids_path.exists() else list(by_id)
    samples = []
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
        samples.append({
            "sample_id": str(question_id),
            "image_id": str(question["image_id"]),
            "image_path": image_path,
            "question": question["question"],
            "ground_truth": annotation["multiple_choice_answer"],
        })
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


def select_attention_heads(attention: Any, head: int) -> tuple[Any, str, int | str]:
    tensor = attention[0].float()
    if head >= 0:
        if head >= tensor.shape[0]:
            raise ValueError(f"Selected head {head} out of range for {tensor.shape[0]} heads")
        return tensor[head], "raw_head", head
    return tensor.mean(dim=0), "head_mean", "all"


def attention_debug_line(model_name, sample_id, source, layer_count, layer_index, selected_head, tensor_shape):
    head_count = tensor_shape[1] if len(tensor_shape) > 1 else 0
    token_count = tensor_shape[-1] if tensor_shape else 0
    return (
        f"[attention] model={model_name} sample={sample_id} source={source} "
        f"Layer Count={layer_count} Head Count={head_count} Token Count={token_count} "
        f"selected_layer={layer_index} selected_head={selected_head} tensor_shape={tensor_shape}"
    )


def save_overlay(image: Image.Image, grid: Any, path: Path) -> None:
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(normalize_grid(grid) * 255), mode="L").resize(base.size, Image.Resampling.BILINEAR)
    color = Image.new("RGB", base.size, (255, 48, 0))
    overlay = Image.composite(color, base, heat)
    Image.blend(base, overlay, 0.45).save(path)


def completed_sample_ids(model_dir: Path) -> set[str]:
    path = model_dir / "results.jsonl"
    if not path.exists():
        return set()
    return {str(json.loads(line)["sample_id"]) for line in path.read_text().splitlines() if line.strip()}


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


def _even_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted({int(round(float(v))) for v in np.linspace(0, length - 1, count)})


def sample_layer_heatmaps(layer_heatmaps: list[dict[str, Any]], target_count: int = DEFAULT_TARGET_LAYER_COUNT):
    total = len(layer_heatmaps)
    for index, layer in enumerate(layer_heatmaps):
        layer.setdefault("original_layer_index", index)
    if target_count <= 0 or total <= target_count:
        for i, layer in enumerate(layer_heatmaps):
            layer["display_layer_index"] = i
        return layer_heatmaps, {
            "method": "all_real_layers",
            "target_layer_count": target_count,
            "original_layer_count": total,
            "sampled_layer_count": total,
            "sampled_original_layer_indices": list(range(total)),
        }

    groups = []
    current_name, current_indices = None, []
    for index, layer in enumerate(layer_heatmaps):
        name = str(layer.get("component") or "model")
        if current_name is None or name == current_name:
            current_name = name
            current_indices.append(index)
        else:
            groups.append((current_name, current_indices))
            current_name, current_indices = name, [index]
    if current_name is not None:
        groups.append((current_name, current_indices))

    if len(groups) > target_count:
        selected = _even_indices(total, target_count)
    else:
        quotas = {name: 1 for name, _ in groups}
        remaining = target_count - len(groups)
        total_len = sum(len(indices) for _, indices in groups)
        fractions = []
        for name, indices in groups:
            exact = remaining * len(indices) / total_len if total_len else 0
            extra = int(np.floor(exact))
            quotas[name] += extra
            fractions.append((exact - extra, name))
        assigned = sum(quotas.values())
        for _fraction, name in sorted(fractions, reverse=True):
            if assigned >= target_count:
                break
            quotas[name] += 1
            assigned += 1
        selected = []
        for name, indices in groups:
            for local in _even_indices(len(indices), min(quotas[name], len(indices))):
                selected.append(indices[local])
        selected = sorted(set(selected))
        if len(selected) > target_count:
            selected = [selected[i] for i in _even_indices(len(selected), target_count)]
    sampled = [layer_heatmaps[i] for i in selected]
    for display_index, layer in enumerate(sampled):
        layer["display_layer_index"] = display_index
    return sampled, {
        "method": "component_stratified_even_real_layer_sampling",
        "target_layer_count": target_count,
        "original_layer_count": total,
        "sampled_layer_count": len(sampled),
        "sampled_original_layer_indices": selected,
    }


def write_layer_sample(
    model_dir: Path,
    sample: dict[str, Any],
    prediction: str,
    attention_method: str,
    layer_heatmaps: list[dict[str, Any]],
    _layers: str = "all",
    save_all_layers: bool = False,
    target_layer_count: int = DEFAULT_TARGET_LAYER_COUNT,
) -> dict[str, Any]:
    sample_dir = model_dir / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(sample["image_path"]).convert("RGB")
    for old_heatmap in sample_dir.glob("layer_*.png"):
        old_heatmap.unlink()
    original_count = len(layer_heatmaps)
    target = original_count if save_all_layers else target_layer_count
    sampled_layers, layer_sampling = sample_layer_heatmaps(layer_heatmaps, target)
    diffs = []
    for i in range(1, len(sampled_layers)):
        diffs.append({
            "left_layer_index": i - 1,
            "right_layer_index": i,
            "left_source_layer_index": sampled_layers[i - 1].get("source_layer_index", i - 1),
            "right_source_layer_index": sampled_layers[i].get("source_layer_index", i),
            **layer_difference(sampled_layers[i - 1]["weights"], sampled_layers[i]["weights"]),
        })
    serializable_layers = []
    for layer_index, layer in enumerate(sampled_layers):
        grid14 = resize_grid(layer["weights"], 14)
        image_path = sample_dir / f"layer_{layer_index:03d}.png"
        save_overlay(image, grid14, image_path)
        serializable_layers.append({
            "layer_index": layer_index,
            "display_layer_index": layer.get("display_layer_index", layer_index),
            "original_layer_index": layer.get("original_layer_index", layer_index),
            "source_layer_index": layer.get("source_layer_index", layer_index),
            "attention_image": str(image_path),
            "attention_source": layer.get("attention_source"),
            "component": layer.get("component"),
            "visual_representation": layer.get("visual_representation"),
            "attention_tensor_shape": layer.get("attention_tensor_shape"),
            "head_mode": layer.get("head_mode"),
            "selected_head": layer.get("selected_head"),
            "query_token_count": layer.get("query_token_count"),
            "visual_token_count": layer.get("visual_token_count"),
            "visual_grid_shape": layer.get("visual_grid_shape"),
            "token_aggregation": layer.get("token_aggregation"),
            "raw_visual_grid": np.asarray(layer["weights"], dtype=float).tolist(),
            "weights": grid14.astype(float).tolist(),
        })
    row = {
        "sample_id": sample["sample_id"],
        "image_id": sample["image_id"],
        "image_path": str(sample["image_path"]),
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prediction": prediction,
        "attention_method": attention_method,
        "saved_layer_selector": "all_real_layers" if save_all_layers else "sampled_real_layers",
        "available_layer_count": original_count,
        "full_layer_count": original_count,
        "layer_sampling": layer_sampling,
        "layer_count": len(serializable_layers),
        "layers": serializable_layers,
        "adjacent_layer_differences": diffs,
    }
    (sample_dir / "metadata.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (model_dir / "results.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def write_summary(model_dir: Path, model_name: str, attention_method: str) -> None:
    results_path = model_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()] if results_path.exists() else []
    summary = {
        "status": "ok",
        "model": model_name,
        "sample_count": len(rows),
        "layer_heatmap_count": sum(row.get("layer_count", 0) for row in rows),
        "max_layer_count": max((row.get("layer_count", 0) for row in rows), default=0),
        "max_full_layer_count": max((row.get("full_layer_count", 0) for row in rows), default=0),
        "target_layer_count": DEFAULT_TARGET_LAYER_COUNT,
        "attention_method": attention_method,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
