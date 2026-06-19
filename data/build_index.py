import glob, json, os, shutil
import numpy as np

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
    # Collect vectors per model for global sink stats
    # { model_name: { num_patches: [vec, vec, ...] } }
    model_vecs = {}

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

        # Collect vis_attn vectors for global stats (skip rollout, use last 1/3 layers)
        model_name = data.get("model", model)
        if model_name not in model_vecs:
            model_vecs[model_name] = {}

        all_tokens = list(data.get("question", {}).values()) + list(data.get("answer", {}).values())
        for tok in all_tokens:
            vis = tok.get("vis_attn", {})
            if not vis:
                continue
            # Find which layers exist (exclude rollout)
            layers = sorted([k for k in vis if k.startswith("layer_")],
                            key=lambda x: int(x.split("_")[1]))
            if not layers:
                continue
            # Use last 1/3 of layers for representative signal
            use_layers = layers[max(0, len(layers) * 2 // 3):]
            for lk in use_layers:
                vec = vis[lk]
                if not vec:
                    continue
                n = len(vec)
                if n not in model_vecs[model_name]:
                    model_vecs[model_name][n] = []
                model_vecs[model_name][n].append(vec)

    # Compute global mean/std per model per patch count
    # global_stats: { model_name: { "n_patches": n, "mean": [...], "std": [...] } }
    global_stats = {}
    for model_name, size_map in model_vecs.items():
        # Use the most common patch count for this model
        if not size_map:
            continue
        n_patches = max(size_map, key=lambda n: len(size_map[n]))
        vecs = np.array(size_map[n_patches], dtype=np.float32)  # (N, n_patches)
        mean = vecs.mean(axis=0).tolist()
        std  = (vecs.std(axis=0) + 1e-12).tolist()
        global_stats[model_name] = {
            "n_patches": n_patches,
            "n_samples": len(vecs),
            "mean": mean,
            "std":  std,
        }
        print(f"  global_stats [{model_name}]: {len(vecs)} vectors, {n_patches} patches")

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "attn_index.json")
    with open(idx_path, "w") as f:
        json.dump({"samples": samples, "global_stats": global_stats}, f,
                  ensure_ascii=False, indent=2)
    print(f"Index written: {idx_path} ({len(samples)} samples)")

if __name__ == "__main__":
    build(
        src_dirs=["data/qwen", "data/blip", "data/llava", "data/clip"],
        img_src="data/img",
        public_dir="public",
        out_dir="src/data",
    )
