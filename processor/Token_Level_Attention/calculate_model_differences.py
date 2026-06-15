#!/usr/bin/env python3
"""Calculate L1 and KL differences from existing token-level attention outputs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import DEFAULT_OUTPUT_ROOT


MODEL_NAMES = ("CLIP", "BLIP2", "LLaVA", "Qwen2-VL")
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--token-slices", type=int, default=8)
    parser.add_argument("--layer-slices", type=int, default=8)
    parser.add_argument(
        "--difference-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "model_differences",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def relative_positions(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    return [index / (count - 1) for index in range(count)]


def relative_index(length: int, position: float) -> int:
    return min(length - 1, max(0, int(round(position * (length - 1)))))


def distribution(weights: Any) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None) + EPSILON
    return values / values.sum()


def metrics(left_weights: Any, right_weights: Any) -> dict[str, float]:
    left = distribution(left_weights)
    right = distribution(right_weights)
    left_to_right = float(np.sum(left * np.log(left / right)))
    right_to_left = float(np.sum(right * np.log(right / left)))
    return {
        "l1_sum": float(np.sum(np.abs(left - right))),
        "kl_left_to_right": left_to_right,
        "kl_right_to_left": right_to_left,
        "symmetric_kl": (left_to_right + right_to_left) / 2.0,
    }


def select_slice(
    sample: dict[str, Any],
    token_position: float,
    layer_position: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    tokens = [token for token in sample.get("tokens", []) if token.get("layers")]
    if not tokens:
        return None
    token = tokens[relative_index(len(tokens), token_position)]
    layers = token["layers"]
    layer = layers[relative_index(len(layers), layer_position)]
    return token, layer


def main() -> None:
    args = parse_args()
    args.difference_dir.mkdir(parents=True, exist_ok=True)
    model_results = {
        model: load_results(args.output_root / model / "results.jsonl")
        for model in args.models
    }
    token_positions = relative_positions(max(1, args.token_slices))
    layer_positions = relative_positions(max(1, args.layer_slices))
    output_path = args.difference_dir / "results.jsonl"
    row_count = 0
    sample_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output:
        for left_model, right_model in itertools.combinations(args.models, 2):
            common_ids = sorted(
                set(model_results[left_model]) & set(model_results[right_model])
            )
            for sample_id in common_ids:
                left_sample = model_results[left_model][sample_id]
                right_sample = model_results[right_model][sample_id]
                for token_position in token_positions:
                    for layer_position in layer_positions:
                        left_slice = select_slice(
                            left_sample, token_position, layer_position
                        )
                        right_slice = select_slice(
                            right_sample, token_position, layer_position
                        )
                        if left_slice is None or right_slice is None:
                            continue
                        left_token, left_layer = left_slice
                        right_token, right_layer = right_slice
                        row = {
                            "sample_id": sample_id,
                            "question": left_sample.get("question", ""),
                            "left_model": left_model,
                            "right_model": right_model,
                            "alignment": "nearest_relative_token_and_layer_position",
                            "relative_token_position": token_position,
                            "relative_layer_position": layer_position,
                            "left_token_index": left_token.get("token_index"),
                            "left_token": left_token.get("token"),
                            "left_layer_index": left_layer.get("layer_index"),
                            "left_source_layer_index": left_layer.get(
                                "source_layer_index"
                            ),
                            "right_token_index": right_token.get("token_index"),
                            "right_token": right_token.get("token"),
                            "right_layer_index": right_layer.get("layer_index"),
                            "right_source_layer_index": right_layer.get(
                                "source_layer_index"
                            ),
                            **metrics(left_layer["weights"], right_layer["weights"]),
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        row_count += 1
                        sample_ids.add(sample_id)

    summary = {
        "status": "ok",
        "models": args.models,
        "sample_count": len(sample_ids),
        "comparison_count": row_count,
        "token_slices": len(token_positions),
        "layer_slices": len(layer_positions),
        "alignment": "nearest_relative_token_and_layer_position",
        "metrics": {
            "l1_sum": "sum absolute difference between normalized 14x14 distributions; range [0, 2]",
            "symmetric_kl": "mean of KL(left||right) and KL(right||left)",
        },
    }
    (args.difference_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
