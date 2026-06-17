import glob, json, os, shutil

def build(src_dirs, img_src, public_dir, out_dir):
    attn_public = os.path.join(public_dir, "data", "attn")
    img_public = os.path.join(public_dir, "img")
    os.makedirs(attn_public, exist_ok=True)
    os.makedirs(img_public, exist_ok=True)

    # Copy JSONs from all source dirs → public/data/attn/
    for d in src_dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            shutil.copy2(f, os.path.join(attn_public, os.path.basename(f)))

    # Copy images from img_src → public/img/
    for f in glob.glob(os.path.join(img_src, "*.jpeg")):
        shutil.copy2(f, os.path.join(img_public, os.path.basename(f)))

    # Scan public/data/attn/ for index
    files = sorted(glob.glob(os.path.join(attn_public, "*.json")))
    samples = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".json", "").split("_", 2)
        if len(parts) < 3:
            continue
        model, image_id, question_id = parts
        try:
            with open(f) as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            continue
        samples.append({
            "image_id": image_id,
            "question_id": question_id,
            "file": basename,
            "correct": data.get("correct", 0),
            "model": data.get("model", model),
        })

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "attn_index.json")
    with open(idx_path, "w") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"Index written: {idx_path} ({len(samples)} samples)")

if __name__ == "__main__":
    build(
        src_dirs=["data/qwen", "data/blip"],
        img_src="data/img",
        public_dir="public",
        out_dir="src/data",
    )
