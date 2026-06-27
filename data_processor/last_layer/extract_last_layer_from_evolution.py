#!/usr/bin/env python3
"""Extract compact last-layer heatmap data from layer-evolution outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "src" / "data" / "layer_evolution"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "data" / "last_layer_attention"
MODEL_DIRS = ("clip", "blip", "llava", "qwen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(MODEL_DIRS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        results_path = args.input_root / model / "results.jsonl"
        if not results_path.exists():
            print(f"[skip] missing {results_path}")
            continue
        model_out = args.output_root / model
        model_out.mkdir(parents=True, exist_ok=True)
        output_jsonl = model_out / "results.jsonl"
        if args.overwrite and output_jsonl.exists():
            output_jsonl.unlink()
        with output_jsonl.open("a", encoding="utf-8") as output:
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                layers = row.get("layers", [])
                if not layers:
                    continue
                last = layers[-1]
                src_img = Path(last["attention_image"])
                sample_dir = model_out / str(row["sample_id"])
                sample_dir.mkdir(parents=True, exist_ok=True)
                dst_img = sample_dir / "last_layer.png"
                if src_img.exists() and (args.overwrite or not dst_img.exists()):
                    shutil.copy2(src_img, dst_img)
                compact = {
                    "sample_id": row["sample_id"],
                    "image_id": row.get("image_id"),
                    "image_path": row.get("image_path"),
                    "question": row.get("question"),
                    "ground_truth": row.get("ground_truth"),
                    "prediction": row.get("prediction"),
                    "model": model,
                    "attention_image": str(dst_img),
                    "layer": last,
                    "source": "last_display_layer_from_layer_evolution",
                }
                output.write(json.dumps(compact, ensure_ascii=False) + "\n")
                rows.append(compact)
    summary = {
        "status": "ok",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "model_count": len(set(row["model"] for row in rows)),
        "sample_model_count": len(rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
