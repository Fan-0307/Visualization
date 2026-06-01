#!/usr/bin/env python3
"""Build the processed VQA comparison dataset from model outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("qwen", "llava", "salesforce", "openai")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def index_rows(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id={sample_id!r} in {source}")
        indexed[sample_id] = row
    return indexed


def load_model_outputs(
    output_dir: Path,
    models: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        results_path = output_dir / model / "results.jsonl"
        if not results_path.exists():
            raise FileNotFoundError(f"missing results file: {results_path}")
        by_model[model] = index_rows(read_jsonl(results_path), results_path)
    return by_model


def validate_sample_alignment(
    by_model: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
) -> list[str]:
    reference_ids = list(by_model[models[0]].keys())
    reference_set = set(reference_ids)

    for model in models[1:]:
        sample_ids = set(by_model[model])
        missing = reference_set - sample_ids
        extra = sample_ids - reference_set
        if missing or extra:
            raise ValueError(
                f"sample_id mismatch for {model}: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    return reference_ids


def copy_attention_image(
    row: dict[str, Any],
    model: str,
    output_dir: Path,
    photos_dir: Path,
) -> Path:
    src_attention = Path(row["attention_image"])
    if not src_attention.exists():
        fallback_attention = output_dir / model / "attention_images" / src_attention.name
        if not fallback_attention.exists():
            raise FileNotFoundError(
                f"missing attention image: {src_attention} "
                f"and fallback {fallback_attention}"
            )
        src_attention = fallback_attention

    model_photo_dir = photos_dir / model
    model_photo_dir.mkdir(parents=True, exist_ok=True)
    dst_attention = model_photo_dir / src_attention.name

    if (
        not dst_attention.exists()
        or src_attention.stat().st_size != dst_attention.stat().st_size
    ):
        shutil.copy2(src_attention, dst_attention)

    return dst_attention


def build_results(
    by_model: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    models: list[str],
    output_dir: Path,
    photos_dir: Path,
) -> dict[str, Any]:
    questions: dict[str, dict[str, Any]] = {}

    for sample_id in sample_ids:
        reference = by_model[models[0]][sample_id]
        item: dict[str, Any] = {
            "sample_id": sample_id,
            "image_id": reference.get("image_id"),
            "image_path": reference.get("image_path"),
            "question": reference.get("question"),
            "ground_truth": reference.get("ground_truth"),
        }

        for model in models:
            row = by_model[model][sample_id]
            if row.get("question") != reference.get("question"):
                raise ValueError(
                    f"question mismatch for sample_id={sample_id}, model={model}"
                )

            attention_image = copy_attention_image(row, model, output_dir, photos_dir)
            item[model] = [
                {
                    "answer": row.get("prediction"),
                    "correct": row.get("correct"),
                    "confidence": row.get("confidence"),
                    "attention_image": str(attention_image),
                }
            ]

        questions[sample_id] = item

    return {
        "models": models,
        "data": questions,
    }


def load_source_summary(output_dir: Path, model: str) -> dict[str, Any]:
    summary_path = output_dir / model / "summary.json"
    if not summary_path.exists():
        return {"status": "missing_summary"}
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_summary(
    output_dir: Path,
    processed_dir: Path,
    results_path: Path,
    by_model: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    total_questions: int,
) -> dict[str, Any]:
    photos_dir = processed_dir / "photos"
    model_summaries: dict[str, dict[str, Any]] = {}

    for model in models:
        summary = load_source_summary(output_dir, model)
        summary["results_count"] = len(by_model[model])
        summary["heatmap_count"] = len(list((photos_dir / model).glob("*.png")))
        summary["photos_dir"] = str(photos_dir / model)
        model_summaries[model] = summary

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_output_dir": str(output_dir),
        "processed_data_dir": str(processed_dir),
        "results_json": str(results_path),
        "total_questions": total_questions,
        "models": model_summaries,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_processed_data(
    output_dir: Path,
    processed_dir: Path,
    models: list[str],
) -> dict[str, Any]:
    photos_dir = processed_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    by_model = load_model_outputs(output_dir, models)
    sample_ids = validate_sample_alignment(by_model, models)
    results = build_results(by_model, sample_ids, models, output_dir, photos_dir)

    results_path = processed_dir / "results.json"
    summary_path = processed_dir / "summary.json"
    write_json(results_path, results)

    summary = build_summary(
        output_dir=output_dir,
        processed_dir=processed_dir,
        results_path=results_path,
        by_model=by_model,
        models=models,
        total_questions=len(sample_ids),
    )
    write_json(summary_path, summary)

    return {
        "processed_dir": str(processed_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "total_questions": len(sample_ids),
        "models": {
            model: {
                "results_count": len(by_model[model]),
                "heatmap_count": len(list((photos_dir / model).glob("*.png"))),
            }
            for model in models
        },
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create data/processed_data from data/output model results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "output",
        help="Directory containing one subdirectory per model.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=repo_root / "data" / "processed_data",
        help="Destination directory for results.json, summary.json, and photos/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model directory names to merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_processed_data(
        output_dir=args.output_dir.resolve(),
        processed_dir=args.processed_dir.resolve(),
        models=list(args.models),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
