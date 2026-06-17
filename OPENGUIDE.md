# VLM Attention Visualization — Project Overview

## Goal

Compare vision-language model (VLM) attention mechanisms per layer per token on VQAv2. Extract cross-attention from vision patches to text tokens for Qwen3-VL-2B and BLIP-vqa-base, display as interactive heatmaps in a Vue 3 frontend.

## Directory Structure

```
Visualization/
├── data/
│   ├── qwen/run.py          # Qwen3-VL-2B extraction script
│   ├── blip/run.py          # BLIP-vqa-base extraction script
│   └── build_index.py       # Scan public/data/attn/ → src/data/attn_index.json
├── public/
│   ├── data/attn/           # Per-token JSONs served statically (64 files)
│   └── img/                 # Input images served statically (8 JPEGs)
├── src/
│   ├── App.vue              # 4-tab layout, imports TokenAttentionDetail
│   ├── main.js
│   ├── data/
│   │   ├── attn_index.json  # Index listing all available samples (imported by Vue)
│   │   └── vl_attention_data.json  # Legacy aggregated data (tab 1-3)
│   └── components/
│       ├── TokenAttentionDetail.vue  # Per-token heatmap tab (tab 4, NEW)
│       ├── ModelComparison.vue       # Tab 1
│       ├── MatrixAttention.vue       # Tab 2
│       └── CorrectErrorPattern.vue   # Tab 3
├── index.html
├── package.json
├── vite.config.js
└── .venv/                   # Python virtual environment
```

## Data Flow

```
run.py (GPU) ──→ public/data/attn/{Model}_{ImgId}_{QId}.json
               ──→ public/img/{ImgId}.jpeg
build_index.py ──→ src/data/attn_index.json (scans public/data/attn/)
TokenAttentionDetail.vue  (imports attn_index.json, fetches JSONs at runtime)
```

## Python Scripts

### `data/qwen/run.py`

- Model: `Qwen/Qwen3-VL-2B-Instruct` (local cache `D:\.cache\hf_models_manual_download\`)
- Dataset: `lmms-lab/VQAv2`, shard `data/train-00000-of-00136.parquet` (32 samples, indices 0–31)
- Extracts attention via `model(..., output_attentions=True)` with `mm_token_type_mask` to include generated tokens
- Token regions identified by special tokens: `<|vision_start|>` / `<|vision_end|>`, `<|im_end|>`
- Output: each per-token JSON has `vis_attn` (→vision patches, averaged across heads, 28 layers) and `que_attn` (→question tokens) for every question and answer token
- Grid: computed from aspect ratio (`(V/aspect_ratio)^0.5`), not square
- `is_correct`: substring match (Qwen generates full sentences like "Based on… the snow is white.")

### `data/blip/run.py`

- Model: `Salesforce/blip-vqa-base` (AutoModelForVisualQuestionAnswering)
- Dataset: same VQAv2 shard
- Uses model subcomponents directly to capture attentions:
  - `model.vision_model` → image_embeds
  - `model.text_encoder` → question tokens' cross-attn to vision patches (encoder_cross_attentions)
  - `model.text_decoder` → answer tokens' cross-attn to question_embeds (decoder_cross_attentions)
- **Answer tokens have `vis_attn: {}`** because BLIP's decoder cross-attends to `question_embeds`, not vision patches directly
- Grid: square 24×24 (image_size=384, patch_size=16)
- is_correct also uses substring match

### `data/build_index.py`

- Copies JSONs from `data/qwen/` and `data/blip/` to `public/data/attn/`
- Copies JPEGs from `data/img/` to `public/img/`
- Scans `public/data/attn/` to write `src/data/attn_index.json`
- Format: `{"samples": [{"image_id","question_id","file","correct","model"}, ...]}`

## JSON Schema (per-token, one file per sample)

```jsonc
{
  "image": { "w": 640, "h": 480, "img_id": 458752, "grid": { "w": 15, "h": 20 } },
  "question": {
    "0": {
      "token": "What",
      "vis_attn": { "layer_0": [0.01, 0.02, ...], "layer_1": [...], ... },
      "que_attn": { "layer_0": [...], ... }
    },
    ...
  },
  "answer": {
    "6": {
      "token": "Based",
      "vis_attn": { "layer_0": [...], ... },
      "que_attn": { "layer_0": [...], ... }
    },
    ...
  },
  "correct": 1,
  "model": "Qwen3-VL-2B"
}
```

- `vis_attn` vector length = number of vision patches (excluding CLS for Qwen, including CLS for BLIP — component does `vec.slice(1)` to skip CLS)
- Qwen: 28 layers, BLIP: encoder has 12 layers (question), decoder has 12 layers (answer→vis_attn empty)
- Keys are stringified integer token indices

## Frontend Component: `TokenAttentionDetail.vue`

- 4th tab "逐层 Token 注意力"
- Shows a dropdown of all samples from `attn_index.json` (imported statically)
- On selection, fetches full JSON from `/data/attn/{file}` at runtime
- Layer slider (range input) controls which layer is displayed
- For each selected layer: question tokens show vis_attn as D3 heatmap + que_attn as D3 bar; answer tokens show que_attn bar
- Correct/error badge

## Current State

- ✅ Both scripts run on GPU (tested: 32 samples each, ~30 min per model)
- ✅ 64 per-token JSONs generated, stored in `public/data/attn/`
- ✅ 8 images stored in `public/img/`
- ✅ Index generated at `src/data/attn_index.json`
- ✅ Frontend builds successfully (`npm run build`, 588 modules)
- ✅ `is_correct` uses substring matching (Qwen generates long sentences)
- ✅ BLIP answer vis_attn = {} (architectural limitation)
- ✅ `data/qwen/run.py` and `data/blip/run.py` output directly to `public/`
- ✅ `dist/`, `public/photos`, `public/val2014` (stale files) deleted

## Environment

```
Python: .venv\Scripts\python.exe (project-local venv)
Node:  requires npm/pnpm
GPU:   cuda:0 available
OS:    Windows (PowerShell)
Working dir: D:\Dev\DV\Project\Visualization
Model cache: D:\.cache\hf_models_manual_download\Qwen3-VL-2B-Instruct
```

## Commands

```powershell
# Generate data (GPU required)
.venv\Scripts\python.exe data\qwen\run.py
.venv\Scripts\python.exe data\blip\run.py

# Rebuild index (after generating)
.venv\Scripts\python.exe data\build_index.py

# Dev server
npm run dev

# Build
npm run build
```

## Key Design Decisions

1. **BLIP answer vis_attn = {}** — `BlipForQuestionAnswering` decoder cross-attends to `question_embeds`, not vision patches. Only encoder cross-attn (question→vision) is available.
2. **Substring matching for is_correct** — Qwen generates verbose sentences; "white" matches within "the snow is white."
3. **Per-token JSONs fetched at runtime** — Not merged into a single file. `attn_index.json` (small, imported) lists available files, actual data loaded via `fetch()` per sample.
4. **Model name in filename** — `{Model}_{ImgId}_{QId}.json` avoids collision when both models process the same sample.
5. **Both scripts output to `public/`** — `build_index.py` copies old data from `data/qwen|blip/` → `public/` for backwards compatibility, but new runs write directly to `public/`.
6. **Eager attention** — Both models run with `attn_implementation="eager"` to extract full attention matrices.

## Potential Next Steps

- Add "select all layers" mode (aggregate or animate)
- Support model comparison within the per-token view
- Display more samples (currently 32 per model, indices 0–31)
- Add legend/color scale to heatmaps
- Optimize BLIP to extract encoder cross-attention for answer tokens via chained attention
