import torch, os, json
from PIL import Image

MODEL_PATH = os.environ.get("LLAVA_MODEL_PATH", "llava-hf/llava-1.5-7b-hf")
IMG_DIR = "public/img"
OUT_DIR = "public/data/attn"
MODEL   = os.environ.get("LLAVA_MODEL_NAME", "LLaVA-1.5-7B")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    from datasets import load_dataset, VerificationMode
    datas = load_dataset('lmms-lab/VQAv2', data_files='data/train-00000-of-00136.parquet', verification_mode=VerificationMode.NO_CHECKS, keep_in_memory=True)['train']
    print("Load VQAv2 data/train-00000-of-00136.parquet")
    return datas

def load_model():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"Load {MODEL} on device: {model.device}")
    return model, processor

def inference(model, processor, data):
    prompt = f"USER: <image>\n{data['question']}\nASSISTANT:"
    inputs = processor(text=prompt, images=data['image'], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print("Finish Inference")
    return inputs, outputs

def all_attn(model, inputs, outputs):
    """Re-forward full sequence with output_attentions=True.

    LLaVA replaces <image> tokens with visual patch embeddings during forward,
    so the attention matrix is larger than input_ids. We detect the expansion
    and compute position offsets.
    """
    full_ids = outputs  # [1, prompt_len + answer_len]

    reoutputs = model(
        input_ids=full_ids,
        pixel_values=inputs.pixel_values,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )

    image_token_id = model.config.image_token_index
    img_positions = torch.where(full_ids[0] == image_token_id)[0]
    attn_seq_len = reoutputs.attentions[0].shape[-1]   # key/value dim
    attn_query_len = reoutputs.attentions[0].shape[-2]  # query dim
    ids_len = full_ids.shape[1]
    prompt_len = inputs.input_ids.shape[1]

    # Visual tokens in the merged attention sequence
    # One <image> token in input_ids → N visual patches in attention
    if len(img_positions) == 1 and attn_seq_len > ids_len:
        num_vis = attn_seq_len - ids_len + len(img_positions)
        vis_start = int(img_positions[0])
    elif len(img_positions) > 1:
        vis_start = int(img_positions[0])
        num_vis = attn_seq_len - ids_len + len(img_positions)
    else:
        vis_start = 0
        num_vis = 0

    vis_end = vis_start + num_vis

    # query_offset: how much attention query positions are shifted vs input_ids positions
    # Tokens before vis_start: no shift. Tokens after vis_start: shifted by (num_vis - n_img_tokens)
    query_offset = attn_query_len - ids_len

    # Question tokens (in input_ids space): all prompt tokens except image placeholders
    que_in_ids = [i for i in range(prompt_len) if full_ids[0, i].item() != image_token_id]

    # Answer tokens (in input_ids space): all tokens after prompt
    ans_start_ids = prompt_len
    ans_end_ids = ids_len

    print(f"  vis=[{vis_start},{vis_end}) num_vis={num_vis} "
          f"attn_seq={attn_seq_len} attn_q={attn_query_len} ids={ids_len} "
          f"prompt={prompt_len} q_offset={query_offset}")
    print("Finish Attention Fetch")
    return {
        "attentions": reoutputs.attentions,
        "vis_start": vis_start,
        "vis_end": vis_end,
        "num_vis": num_vis,
        "query_offset": query_offset,
        "que_in_ids": que_in_ids,
        "ans_start_ids": ans_start_ids,
        "ans_end_ids": ans_end_ids,
        "full_ids": full_ids,
        "prompt_len": prompt_len,
    }

def attention_rollout(attentions, attn_seq_len):
    """Attention Rollout (Abnar & Zuidema 2020).
    Ã_l = 0.5 * mean_heads(A_l) + 0.5 * I, row-normalized.
    Rollout = Ã_L · ... · Ã_1  (on CPU).
    Returns (attn_seq_len, attn_seq_len) float32 tensor.
    """
    eye = torch.eye(attn_seq_len, dtype=torch.float32)
    rollout = eye.clone()
    for layer_attn in attentions:
        A = layer_attn[0].float().mean(dim=0).cpu()
        A_tilde = 0.5 * A + 0.5 * eye
        A_tilde = A_tilde / A_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        rollout = A_tilde @ rollout
    return rollout

def to_json(attn_info, data, outputs, processor):
    W, H = data['image'].size
    img_id = data['image_id']

    attentions = attn_info["attentions"]
    vis_start = attn_info["vis_start"]
    vis_end = attn_info["vis_end"]
    num_vis = attn_info["num_vis"]
    query_offset = attn_info["query_offset"]
    que_in_ids = attn_info["que_in_ids"]
    ans_start_ids = attn_info["ans_start_ids"]
    ans_end_ids = attn_info["ans_end_ids"]
    full_ids = attn_info["full_ids"]

    # Grid: infer 2D shape from visual patch count + image aspect ratio
    aspect_ratio = W / H
    merged_rows = int(round((num_vis / aspect_ratio) ** 0.5))
    merged_cols = int(round(num_vis / merged_rows))
    while merged_rows * merged_cols < num_vis:
        if (merged_cols + 1) / merged_rows <= aspect_ratio:
            merged_cols += 1
        else:
            merged_rows += 1

    # Map input_ids position → attention query position
    def attn_query_pos(ids_pos):
        if ids_pos < vis_start:
            return ids_pos
        else:
            return ids_pos + query_offset

    # Compute rollout once for the full sequence (in attention-space coordinates)
    attn_seq_len = attentions[0].shape[-1]
    rollout = attention_rollout(attentions, attn_seq_len)
    print("Finish Rollout")

    # Question tokens
    question_attn_dict = {}
    for local_idx, ids_pos in enumerate(que_in_ids):
        tid = full_ids[0, ids_pos].item()
        text = processor.decode(tid)
        entry = {"token": text, "vis_attn": {}, "que_attn": {}}
        qpos = attn_query_pos(ids_pos)

        for layer_idx, layer_attn in enumerate(attentions):
            layer_name = f"layer_{layer_idx}"
            attn_matrix = layer_attn[0]
            entry["vis_attn"][layer_name] = attn_matrix[:, qpos, vis_start:vis_end].mean(dim=0).float().cpu().tolist()
            que_attn_vecs = [attn_matrix[:, qpos, attn_query_pos(p)].mean().item() for p in que_in_ids]
            entry["que_attn"][layer_name] = que_attn_vecs

        entry["vis_attn"]["rollout"] = rollout[qpos, vis_start:vis_end].tolist()
        entry["que_attn"]["rollout"] = [rollout[qpos, attn_query_pos(p)].item() for p in que_in_ids]
        question_attn_dict[ids_pos] = entry

    # Answer tokens
    answer_attn_dict = {}
    for local_idx in range(ans_start_ids, ans_end_ids):
        ids_pos = local_idx
        tid = full_ids[0, ids_pos].item()
        text = processor.decode(tid)
        entry = {"token": text, "vis_attn": {}, "que_attn": {}}
        qpos = attn_query_pos(ids_pos)

        for layer_idx, layer_attn in enumerate(attentions):
            layer_name = f"layer_{layer_idx}"
            attn_matrix = layer_attn[0]
            entry["vis_attn"][layer_name] = attn_matrix[:, qpos, vis_start:vis_end].mean(dim=0).float().cpu().tolist()
            que_attn_vecs = [attn_matrix[:, qpos, attn_query_pos(p)].mean().item() for p in que_in_ids]
            entry["que_attn"][layer_name] = que_attn_vecs

        entry["vis_attn"]["rollout"] = rollout[qpos, vis_start:vis_end].tolist()
        entry["que_attn"]["rollout"] = [rollout[qpos, attn_query_pos(p)].item() for p in que_in_ids]
        answer_attn_dict[ids_pos] = entry

    result = {
        "image": {"w": W, "h": H, "img_id": img_id, "grid": {"w": merged_cols, "h": merged_rows}},
        "question": question_attn_dict,
        "answer": answer_attn_dict
    }
    print("Finish Json Construct")
    return result

def is_correct(outputs, data, processor):
    pred_text = processor.decode(outputs[0], skip_special_tokens=True)
    pred = pred_text.split("ASSISTANT:")[-1].strip().lower() if "ASSISTANT:" in pred_text else pred_text.strip().lower()
    # VQAv2 official accuracy: min(#annotators_who_said_pred / 3, 1)
    counts = {}
    for a in data['answers']:
        ans = a['answer'].strip().lower()
        counts[ans] = counts.get(ans, 0) + 1
    score = max((min(counts.get(a, 0), 3) / 3 for a in counts if a in pred or pred in a), default=0)
    return 1 if score >= 1/3 else 0


model, processor = load_model()
IMAGE_TOKEN_ID = model.config.image_token_index
print(f"Image token id: {IMAGE_TOKEN_ID}")

datas = load_data()
for i in range(32):
    data = datas[i]
    img_path = os.path.join(IMG_DIR, f'{data["image_id"]}.jpeg')
    if not os.path.exists(img_path):
        data['image'].save(img_path)
        print(f'Image saved to {img_path}')
    else:
        print(f'Image already exists: {img_path}')
    inputs, outputs = inference(model, processor, data)
    attn_info = all_attn(model, inputs, outputs)
    json_data = to_json(attn_info, data, outputs, processor)
    json_data['correct'] = is_correct(outputs, data, processor)
    json_data['model'] = MODEL
    json_path = os.path.join(OUT_DIR, f'{MODEL}_{data["image_id"]}_{data["question_id"]}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
        print(f'Json saved to {json_path}')
