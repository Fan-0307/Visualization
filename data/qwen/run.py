import torch, os, json
from PIL import Image

path = "D:\\.cache\\hf_models_manual_download\\Qwen3-VL-2B-Instruct"
IMG_DIR = "public/img"
OUT_DIR = "public/data/attn"
MODEL   = "Qwen3-VL-2B"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    from datasets import load_dataset, VerificationMode
    datas = load_dataset('lmms-lab/VQAv2', data_files='data/train-00000-of-00136.parquet', verification_mode=VerificationMode.NO_CHECKS, keep_in_memory=True)['train']
    print("Load VQAv2 data/train-00000-of-00136.parquet")
    return datas

def load_model():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(path)
    model.eval()
    torch.set_grad_enabled(False)
    print(f"Load Qwen3-VL-2B-Instruct on device: {model.device}")
    return model, processor

def inference(model, processor, data):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": data['image']},
            {"type": "text", "text": data['question']}
        ]},]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
    )
    print("Finish Inference")
    return inputs, outputs

def get_indices(inputs, outputs, processor):
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end_id   = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    vis_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vis_end_id   = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    ids = outputs[0]
    def find(id, pos): return int(torch.where(ids == id)[0][0].item())
    vis_start = find(vis_start_id, 0) + 1
    vis_end   = find(vis_end_id, 0)
    que_start = vis_end + 1
    que_end = find(im_end_id, 0)
    ans_start = len(inputs['input_ids'][0])
    ans_end = (len(ids) - 1) if ids[-1]==im_end_id else len(ids)
    return {"vis_start":vis_start, "vis_end":vis_end, "que_start":que_start, "que_end":que_end, "ans_start":ans_start, "ans_end":ans_end}

def all_attn(model, inputs, outputs, indices):
    new_img_token_mask = torch.cat([
        inputs.mm_token_type_ids,
        torch.zeros(1, len(outputs[0]) - len(inputs['input_ids'][0]), device=model.device)
    ], dim=1)
    reoutputs = model(
        input_ids=outputs,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        mm_token_type_ids=new_img_token_mask,
        output_attentions=True,
    )
    print("Finish Attention Fetch")
    return reoutputs.attentions

def to_json(attentions, data, outputs, processor, indices):
    W, H = data['image'].size
    img_id = data['image_id']
    num_vis = indices['vis_end'] - indices['vis_start']
    aspect_ratio = W / H
    merged_rows = int(round((num_vis / aspect_ratio) ** 0.5))
    merged_cols = int(round(num_vis / merged_rows))
    while merged_rows * merged_cols < num_vis:
        if (merged_cols + 1) / merged_rows <= aspect_ratio:
            merged_cols += 1
        else:
            merged_rows += 1

    question_tokens = []
    for local_idx, tid in enumerate(outputs[0][indices['que_start']:indices['que_end']]):
        global_idx = indices['que_start'] + local_idx
        text = processor.decode(tid)
        question_tokens.append({"index": global_idx, "text": text})

    answer_tokens = []
    for local_idx, tid in enumerate(outputs[0][indices['ans_start']:indices['ans_end']]):
        global_idx = indices['ans_start'] + local_idx
        text = processor.decode(tid)
        answer_tokens.append({"index": global_idx, "text": text})

    question_attn_dict = {}
    answer_attn_dict = {}

    for layer_idx, layer_attn in enumerate(attentions):
        layer_name = f"layer_{layer_idx}"
        attn_matrix = layer_attn[0]

        for q_item in question_tokens:
            global_idx = q_item["index"]
            if global_idx not in question_attn_dict:
                question_attn_dict[global_idx] = {
                    "token": q_item["text"],
                    "vis_attn": {},
                    "que_attn": {}
                }
            vis_attn_vec = attn_matrix[:, global_idx, indices['vis_start']:indices['vis_end']].mean(dim=0).float().cpu().tolist()
            question_attn_dict[global_idx]["vis_attn"][layer_name] = vis_attn_vec
            que_attn_vec = attn_matrix[:, global_idx, indices['que_start']:indices['que_end']].mean(dim=0).float().cpu().tolist()
            question_attn_dict[global_idx]["que_attn"][layer_name] = que_attn_vec

        for a_item in answer_tokens:
            global_idx = a_item["index"]
            if global_idx not in answer_attn_dict:
                answer_attn_dict[global_idx] = {
                    "token": a_item["text"],
                    "vis_attn": {},
                    "que_attn": {}
                }
            vis_attn_vec = attn_matrix[:, global_idx, indices['vis_start']:indices['vis_end']].mean(dim=0).float().cpu().tolist()
            answer_attn_dict[global_idx]["vis_attn"][layer_name] = vis_attn_vec
            que_attn_vec = attn_matrix[:, global_idx, indices['que_start']:indices['que_end']].mean(dim=0).float().cpu().tolist()
            answer_attn_dict[global_idx]["que_attn"][layer_name] = que_attn_vec

    result = {
        "image": {"w": W, "h": H, "img_id": img_id, "grid": {"w": merged_cols, "h": merged_rows}},
        "question": question_attn_dict,
        "answer": answer_attn_dict
    }
    print("Finish Json Construct")
    return result

def is_correct(outputs, data, indices, processor):
    ans_ids = outputs[0][indices['ans_start']:indices['ans_end']]
    pred = processor.decode(ans_ids, skip_special_tokens=True).strip().lower()
    gt = data['multiple_choice_answer'].strip().lower()
    if pred == gt or gt in pred or pred in gt:
        return 1
    for a in data['answers']:
        ans = a['answer'].strip().lower()
        if pred == ans or ans in pred or pred in ans:
            return 1
    return 0


model, processor = load_model()
datas = load_data()
for i in range(32):
    data = datas[i]
    img_path = os.path.join(IMG_DIR, f'{data['image_id']}.jpeg')
    if not os.path.exists(img_path):
        data['image'].save(img_path)
        print(f'Image saved to {img_path}')
    else:
        print(f'Image already exists: {img_path}')
    inputs, outputs = inference(model, processor, data)
    indices = get_indices(inputs, outputs, processor)
    attn = all_attn(model, inputs, outputs, indices)
    json_data = to_json(attn, data, outputs, processor, indices)
    json_data['correct'] = is_correct(outputs, data, indices, processor)
    json_data['model'] = MODEL
    json_path = os.path.join(OUT_DIR, f'{MODEL}_{data['image_id']}_{data['question_id']}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
        print(f'Json saved to {json_path}')
