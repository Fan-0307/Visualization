#!/usr/bin/env python3
"""Generate model predictions and attention hotspot overlays for VQA samples."""

import argparse
import gc
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw_data" / "train(1)" / "train" / "vqa_data"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "output"
DEFAULT_MODEL_ROOT = Path("/home/jovyan/softwares/hf_model")
DEFAULT_QWEN2_PATH = Path("/home/jovyan/fan/hf_model/Qwen2_VL/Qwen2-VL-7B-Instruct")
DEFAULT_MODELS = ("openai", "salesforce", "llava", "qwen")


TOP_VQA_ANSWERS = [
    "yes", "no", "2", "1", "white", "3", "red", "blue", "4", "green",
    "black", "yellow", "brown", "right", "left", "man", "woman", "wood",
    "table", "5", "stop", "6", "gray", "tennis", "baseball", "snow",
    "grass", "water", "tree", "cat", "dog", "bird", "standing", "sitting",
    "walking", "cake", "laptop", "phone", "clock", "car", "truck", "bus",
    "train", "bike", "horse", "kitchen", "bathroom", "beach", "street",
    "trees", "food", "pizza", "skateboarding", "surfing", "eating",
    "sleeping", "sheep", "cow", "elephant", "giraffe", "zebra", "bear",
    "living room", "bedroom", "park", "ocean", "sky", "day", "night",
    "summer", "winter", "plastic", "metal", "glass", "paper", "book",
    "computer", "tv", "sandwich", "apple", "orange", "banana", "coffee",
]

YES_NO_ANSWERS = ["yes", "no"]
NUMBER_ANSWERS = [str(number) for number in range(0, 21)]
COLOR_ANSWERS = [
    "white", "black", "gray", "red", "blue", "green", "yellow", "brown",
    "orange", "pink", "purple",
]
TIME_ANSWERS = [
    "day", "night", "morning", "afternoon", "evening", "summer", "winter",
]
PLACE_ANSWERS = [
    "kitchen", "bathroom", "bedroom", "living room", "beach", "street",
    "park", "ocean", "inside", "outside", "indoors", "outdoors",
]
MATERIAL_ANSWERS = [
    "wood", "metal", "plastic", "glass", "paper", "ceramic", "leather",
    "cotton",
]


def question_type(question):
    question = norm_answer(question)
    if re.match(
        r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had|"
        r"should|may|might)\b",
        question,
    ):
        return "yes_no"
    if re.search(r"\b(how many|number of)\b", question):
        return "number"
    if re.search(r"\b(what|which) colou?r\b|\bcolou?r is\b", question):
        return "color"
    if re.search(r"\bwhat time\b|\btime of day\b|\bdaytime\b", question):
        return "time"
    if re.search(r"\bwhere\b|\bwhat room\b|\bwhat place\b", question):
        return "place"
    if re.search(r"\bwhat material\b|\bmade of\b", question):
        return "material"
    return "open"


def candidate_answers(question):
    answer_type = question_type(question)
    candidates = {
        "yes_no": YES_NO_ANSWERS,
        "number": NUMBER_ANSWERS,
        "color": COLOR_ANSWERS,
        "time": TIME_ANSWERS,
        "place": PLACE_ANSWERS,
        "material": MATERIAL_ANSWERS,
    }.get(answer_type, TOP_VQA_ANSWERS)
    return answer_type, candidates


def answer_instruction(question):
    answer_type = question_type(question)
    if answer_type == "yes_no":
        return "Answer only yes or no."
    if answer_type == "number":
        return "Answer only with the number."
    if answer_type == "color":
        return "Answer only with the color."
    if answer_type == "time":
        return "Answer only with the time or time period."
    if answer_type == "place":
        return "Answer only with the place."
    if answer_type == "material":
        return "Answer only with the material."
    return "Answer with one short phrase."


def clean_generated_answer(question, text):
    answer_type = question_type(question)
    cleaned = norm_answer((text or "").split("\n")[0])
    if answer_type == "yes_no":
        matches = re.findall(r"\b(yes|no)\b", cleaned)
        return matches[0] if matches else "unknown"
    if answer_type == "number":
        match = re.search(r"\b(?:[0-9]+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", cleaned)
        return match.group(0) if match else cleaned
    typed_candidates = {
        "color": COLOR_ANSWERS,
        "time": TIME_ANSWERS,
        "place": PLACE_ANSWERS,
        "material": MATERIAL_ANSWERS,
    }.get(answer_type, [])
    for candidate in sorted(typed_candidates, key=len, reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", cleaned):
            return candidate
    return cleaned


def norm_answer(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct(pred, anno):
    pred = norm_answer(pred)
    answers = [norm_answer(anno.get("multiple_choice_answer", ""))]
    answers += [norm_answer(a.get("answer", "")) for a in anno.get("answers", [])]
    return pred in set(a for a in answers if a)


def normalize_grid(grid):
    grid = np.asarray(grid, dtype=np.float32)
    grid = grid - float(grid.min())
    return grid / (float(grid.max()) + 1e-8)


def resize_grid_to_14(grid):
    grid = normalize_grid(grid)
    h, w = grid.shape
    y_idx = np.linspace(0, h - 1, 14).round().astype(int)
    x_idx = np.linspace(0, w - 1, 14).round().astype(int)
    return grid[np.ix_(y_idx, x_idx)]


def as_tensor(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output
    if hasattr(model_output, "last_hidden_state"):
        return model_output.last_hidden_state[:, 0]
    return model_output[0]


def save_overlay(image, grid14, out_path):
    base = image.convert("RGB").resize((336, 336))
    heat = Image.fromarray(np.uint8(normalize_grid(grid14) * 255)).resize(base.size)
    heat = heat.convert("L")
    color = Image.new("RGB", base.size, (255, 48, 0))
    overlay = Image.composite(color, base, heat)
    blended = Image.blend(base, overlay, 0.45)
    blended.save(out_path)


def load_samples(data_dir, limit):
    q_path = data_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    a_path = data_dir / "v2_mscoco_val2014_annotations.json"
    ids_path = data_dir / "stratified_sample_ids.json"
    img_dir = data_dir / "val2014"

    questions = json.load(open(q_path))["questions"]
    annos = {a["question_id"]: a for a in json.load(open(a_path))["annotations"]}
    by_id = {q["question_id"]: q for q in questions}
    wanted = json.load(open(ids_path)).get("question_ids", []) if ids_path.exists() else []
    if not wanted:
        wanted = [q["question_id"] for q in questions]

    samples = []
    for qid in wanted:
        q = by_id.get(qid)
        if not q:
            continue
        img_path = img_dir / f"COCO_val2014_{q['image_id']:012d}.jpg"
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            print(f"[skip] unreadable image: {img_path}")
            continue
        samples.append({
            "sample_id": str(qid),
            "image_id": str(q["image_id"]),
            "image_path": img_path,
            "question": q["question"],
            "annotation": annos[qid],
            "ground_truth": annos[qid]["multiple_choice_answer"],
        })
        if limit and len(samples) >= limit:
            break
    return samples


def load_existing_rows(model_dir):
    path = model_dir / "results.jsonl"
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def write_result(model_dir, sample, pred, confidence, weights):
    model_dir.mkdir(parents=True, exist_ok=True)
    attn_dir = model_dir / "attention_images"
    attn_dir.mkdir(exist_ok=True)
    image = Image.open(sample["image_path"]).convert("RGB")
    attn_file = attn_dir / f"{sample['sample_id']}.png"
    save_overlay(image, weights, attn_file)
    row = {
        "sample_id": sample["sample_id"],
        "image_id": sample["image_id"],
        "image_path": str(sample["image_path"]),
        "attention_image": str(attn_file),
        "question": sample["question"],
        "question_type": question_type(sample["question"]),
        "ground_truth": sample["ground_truth"],
        "prediction": pred,
        "correct": is_correct(pred, sample["annotation"]),
        "confidence": float(confidence),
        "weights": np.asarray(weights, dtype=float).tolist(),
    }
    with open(model_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def summarize(model_dir, rows, status="ok", error=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status,
        "count": len(rows),
        "correct": sum(1 for r in rows if r.get("correct")),
        "accuracy": (sum(1 for r in rows if r.get("correct")) / len(rows)) if rows else 0,
    }
    if error:
        summary["error"] = str(error)
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_openai_clip(samples, out_root, model_path, device, dtype, overwrite=False):
    from transformers import CLIPModel, CLIPProcessor
    model_dir = out_root / "openai"
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype, attn_implementation="eager").to(device).eval()

    rows = [] if overwrite else load_existing_rows(model_dir)
    if overwrite:
        (model_dir / "results.jsonl").unlink(missing_ok=True)
    done = {r["sample_id"] for r in rows}
    for s in tqdm([x for x in samples if x["sample_id"] not in done], desc="openai/clip"):
        image = Image.open(s["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        if dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        _, candidates = candidate_answers(s["question"])
        candidate_prompts = [
            f"Question: {s['question']} Answer: {answer}."
            for answer in candidates
        ]
        text_inputs = processor(
            text=candidate_prompts, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            text_feat = as_tensor(model.get_text_features(**text_inputs))
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            img_feat = as_tensor(model.get_image_features(**inputs))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat.float() @ text_feat.float().T).squeeze(0)
            pred = candidates[int(sims.argmax())]
            conf = sims.softmax(-1).max().item()
            vout = model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
        attn = vout.attentions[-1][0].float().mean(0)
        patch = attn[0, 1:].detach().cpu().numpy()
        side = int(math.sqrt(len(patch)))
        weights = resize_grid_to_14(patch[: side * side].reshape(side, side))
        rows.append(write_result(model_dir, s, pred, conf, weights))
    summarize(model_dir, rows)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_salesforce_blip2(samples, out_root, model_path, device, dtype, overwrite=False):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    model_dir = out_root / "salesforce"
    processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, device_map="auto"
    ).eval()
    model.qformer.config.output_attentions = True
    cap = {}

    def hook(_m, _i, output):
        if isinstance(output, tuple):
            for x in output:
                if torch.is_tensor(x) and x.dim() == 4:
                    cap["weights"] = x.detach()
                    return

    last = next(l for l in reversed(model.qformer.encoder.layer) if l.has_cross_attention)
    target = getattr(last.crossattention.attention, "self", last.crossattention.attention)
    handle = target.register_forward_hook(hook)

    rows = [] if overwrite else load_existing_rows(model_dir)
    if overwrite:
        (model_dir / "results.jsonl").unlink(missing_ok=True)
    done = {r["sample_id"] for r in rows}
    for s in tqdm([x for x in samples if x["sample_id"] not in done], desc="salesforce/blip2"):
        cap.clear()
        image = Image.open(s["image_path"]).convert("RGB")
        prompt = f"{answer_instruction(s['question'])} Question: {s['question']} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        pred = clean_generated_answer(s["question"], text.replace(prompt, "").strip())
        attn = cap.get("weights")
        if attn is None:
            weights = np.zeros((14, 14), dtype=np.float32)
        else:
            a = attn[0].float().mean(0).cpu().numpy()
            patch = a[:, 1:].mean(0)
            side = int(math.sqrt(len(patch)))
            weights = resize_grid_to_14(patch[: side * side].reshape(side, side))
        rows.append(write_result(model_dir, s, pred, 0.5, weights))
    handle.remove()
    summarize(model_dir, rows)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_llava(samples, out_root, model_path, device, dtype, overwrite=False):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_dir = out_root / "llava"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
    ).eval()

    rows = [] if overwrite else load_existing_rows(model_dir)
    if overwrite:
        (model_dir / "results.jsonl").unlink(missing_ok=True)
    done = {r["sample_id"] for r in rows}
    for s in tqdm([x for x in samples if x["sample_id"] not in done], desc="llava"):
        image = Image.open(s["image_path"]).convert("RGB")
        prompt = f"USER: <image>\n{answer_instruction(s['question'])} {s['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        if "pixel_values" in inputs and dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        text = processor.decode(out[0], skip_special_tokens=True)
        pred = clean_generated_answer(
            s["question"], text.split("ASSISTANT:")[-1].strip()
        )
        with torch.no_grad():
            vision_tower = getattr(model, "vision_tower", None) or getattr(getattr(model, "model", None), "vision_tower")
            vout = vision_tower(pixel_values=inputs["pixel_values"], output_attentions=True)
        attn = vout.attentions[-1][0].float().mean(0)
        patch = attn[0, 1:].detach().cpu().numpy()
        side = int(math.sqrt(len(patch)))
        weights = resize_grid_to_14(patch[: side * side].reshape(side, side))
        rows.append(write_result(model_dir, s, pred, 0.5, weights))
    summarize(model_dir, rows)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_qwen(samples, out_root, model_path, device, dtype, overwrite=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_dir = out_root / "qwen"
    rows = [] if overwrite else load_existing_rows(model_dir)
    if overwrite:
        (model_dir / "results.jsonl").unlink(missing_ok=True)
    done = {r["sample_id"] for r in rows}
    model_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else dtype
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=model_dtype,
            device_map="auto",
            max_memory={0: "22GiB", "cpu": "110GiB"},
            attn_implementation="eager",
        ).eval()
    except Exception as e:
        summarize(model_dir, rows, status="failed_to_load", error=e)
        return

    for s in tqdm([x for x in samples if x["sample_id"] not in done], desc="qwen2-vl"):
        image = Image.open(s["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"{answer_instruction(s['question'])} {s['question']}",
                },
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        pred = clean_generated_answer(
            s["question"],
            processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0],
        )
        with torch.no_grad():
            visual_tower = getattr(model, "visual", None) or getattr(model.model, "visual")
            visual = visual_tower(
                inputs["pixel_values"].type(visual_tower.dtype),
                grid_thw=inputs["image_grid_thw"],
            )
        token_scores = visual.last_hidden_state.float().norm(dim=-1).detach().cpu().numpy()
        _, grid_h, grid_w = inputs["image_grid_thw"][0].detach().cpu().tolist()
        patch_count = int(grid_h * grid_w)
        if len(token_scores) >= patch_count:
            grid = token_scores[:patch_count].reshape(int(grid_h), int(grid_w))
            weights = resize_grid_to_14(grid)
        else:
            weights = np.zeros((14, 14), dtype=np.float32)
        rows.append(write_result(model_dir, s, pred, 0.5, weights))
    summarize(model_dir, rows)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model_paths(args):
    model_root = Path(args.model_root)
    return {
        "openai": Path(args.openai_model_path) if args.openai_model_path else model_root / "openai" / "clip-vit-base-patch32",
        "salesforce": Path(args.salesforce_model_path) if args.salesforce_model_path else model_root / "Salesforce" / "blip2-opt-2.7b",
        "llava": Path(args.llava_model_path) if args.llava_model_path else model_root / "llava-hf" / "llava-1.5-7b-hf",
        "qwen": Path(args.qwen_model_path) if args.qwen_model_path else DEFAULT_QWEN2_PATH,
    }


def run_attention_hotspots(data_dir, out_root, models, model_paths, limit=0, overwrite=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    samples = load_samples(Path(data_dir), limit)
    out_root = Path(out_root)
    print(f"[env] device={device} dtype={dtype} samples={len(samples)}")

    for name in models:
        try:
            if name == "openai":
                run_openai_clip(samples, out_root, model_paths[name], device, dtype, overwrite)
            elif name == "salesforce":
                run_salesforce_blip2(samples, out_root, model_paths[name], device, dtype, overwrite)
            elif name == "llava":
                run_llava(samples, out_root, model_paths[name], device, dtype, overwrite)
            elif name == "qwen":
                run_qwen(samples, out_root, model_paths[name], device, dtype, overwrite)
            else:
                print(f"[skip] unknown model group: {name}")
        except Exception as e:
            summarize(out_root / name, [], status="failed", error=e)
            print(f"[error] {name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute samples instead of skipping existing results.jsonl rows.",
    )
    parser.add_argument("--openai-model-path", type=Path)
    parser.add_argument("--salesforce-model-path", type=Path)
    parser.add_argument("--llava-model-path", type=Path)
    parser.add_argument("--qwen-model-path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_attention_hotspots(
        data_dir=args.data_dir.resolve(),
        out_root=args.out_root.resolve(),
        models=models,
        model_paths=build_model_paths(args),
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
