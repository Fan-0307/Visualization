import torch, os, json
from PIL import Image

IMG_DIR = "public/img"
OUT_DIR = "public/data/attn"
MODEL   = "BLIP-vqa-base"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    from datasets import load_dataset, VerificationMode
    datas = load_dataset('lmms-lab/VQAv2', data_files='data/train-00000-of-00136.parquet', verification_mode=VerificationMode.NO_CHECKS, keep_in_memory=True)['train']
    print("Load VQAv2 data/train-00000-of-00136.parquet")
    return datas

def load_model():
    from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor
    model = AutoModelForVisualQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model.eval()
    torch.set_grad_enabled(False)
    print(f"Load BLIP-vqa-base on device: {model.device}")
    return model, processor

def inference(model, processor, data):
    inputs = processor(data['image'], data['question'], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40, return_dict_in_generate=True)
    print("Finish Inference")
    return inputs, outputs

def all_attn(model, inputs, outputs):
    vision_outputs = model.vision_model(
        pixel_values=inputs.pixel_values,
        output_attentions=False,
    )
    image_embeds = vision_outputs.last_hidden_state

    encoder_kwargs = dict(input_ids=inputs.input_ids, output_attentions=True)
    if hasattr(inputs, 'attention_mask'):
        encoder_kwargs['attention_mask'] = inputs.attention_mask
    encoder_outputs = model.text_encoder(
        **encoder_kwargs,
        encoder_hidden_states=image_embeds,
    )
    question_embeds = encoder_outputs.last_hidden_state

    decoder_outputs = model.text_decoder(
        input_ids=outputs.sequences,
        encoder_hidden_states=question_embeds,
        output_attentions=True,
    )

    vis_config = model.config.vision_config
    grid_size = vis_config.image_size // vis_config.patch_size

    print("Finish Attention Fetch")
    return {
        "encoder_self_attentions": encoder_outputs.attentions,
        "encoder_cross_attentions": encoder_outputs.cross_attentions,
        "decoder_cross_attentions": decoder_outputs.cross_attentions,
        "grid_w": grid_size,
        "grid_h": grid_size,
        "num_vis": 1 + grid_size * grid_size,
    }

def to_json(attn_dict, data, inputs, outputs, processor):
    W, H = data['image'].size
    img_id = data['image_id']
    grid_w, grid_h = attn_dict['grid_w'], attn_dict['grid_h']

    question_ids = inputs.input_ids[0]
    num_question = len(question_ids)

    answer_ids = outputs.sequences[0]
    ans_start = 1
    ans_end = (len(answer_ids) - 1) if answer_ids[-1] == processor.tokenizer.sep_token_id else len(answer_ids)
    answer_ids = answer_ids[ans_start:ans_end]
    num_answer = len(answer_ids)

    encoder_self = attn_dict['encoder_self_attentions']
    encoder_cross = attn_dict['encoder_cross_attentions']
    decoder_cross = attn_dict['decoder_cross_attentions']
    num_encoder_layers = len(encoder_self)
    num_decoder_layers = len(decoder_cross)

    question_attn_dict = {}
    for local_idx in range(num_question):
        global_idx = local_idx
        text = processor.decode(question_ids[local_idx])
        entry = {"token": text, "vis_attn": {}, "que_attn": {}}
        for layer_idx in range(num_encoder_layers):
            layer_name = f"layer_{layer_idx}"
            cross_mat = encoder_cross[layer_idx][0]
            entry["vis_attn"][layer_name] = cross_mat[:, local_idx, :].mean(dim=0).float().cpu().tolist()
            self_mat = encoder_self[layer_idx][0]
            entry["que_attn"][layer_name] = self_mat[:, local_idx, :].mean(dim=0).float().cpu().tolist()
        question_attn_dict[global_idx] = entry

    answer_attn_dict = {}
    for local_idx in range(num_answer):
        global_idx = num_question + local_idx
        text = processor.decode(answer_ids[local_idx])
        entry = {"token": text, "vis_attn": {}, "que_attn": {}}
        for layer_idx in range(num_decoder_layers):
            layer_name = f"layer_{layer_idx}"
            cross_mat = decoder_cross[layer_idx][0]
            entry["que_attn"][layer_name] = cross_mat[:, local_idx, :].mean(dim=0).float().cpu().tolist()
        answer_attn_dict[global_idx] = entry

    result = {
        "image": {"w": W, "h": H, "img_id": img_id, "grid": {"w": grid_w, "h": grid_h}},
        "question": question_attn_dict,
        "answer": answer_attn_dict
    }
    print("Finish Json Construct")
    return result

def is_correct(outputs, data, processor):
    ids = outputs.sequences[0]
    ans_start = 1
    ans_end = (len(ids) - 1) if ids[-1] == processor.tokenizer.sep_token_id else len(ids)
    pred = processor.decode(ids[ans_start:ans_end], skip_special_tokens=True).strip().lower()
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
    attn = all_attn(model, inputs, outputs)
    json_data = to_json(attn, data, inputs, outputs, processor)
    json_data['correct'] = is_correct(outputs, data, processor)
    json_data['model'] = MODEL
    json_path = os.path.join(OUT_DIR, f'{MODEL}_{data['image_id']}_{data['question_id']}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)
        print(f'Json saved to {json_path}')
