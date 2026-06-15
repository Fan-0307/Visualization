# Token-Level Visual Attention

Each model has its own exporter because the architectures expose different
relationships between tokens and image regions.

| Script | Token heatmap meaning |
|---|---|
| `CLIP.py` | Each question text token's layer-wise projected cosine similarity to image patches. CLIP has no cross-modal attention. |
| `BLIP2.py` | Each learned Q-Former query token's cross-attention over vision patches at every available cross-attention layer. |
| `LLaVA.py` | Each generated answer token's attention to visual tokens at every LLM layer. |
| `Qwen2-VL.py` | Each generated answer token's attention to visual tokens at every LLM layer. |

Default input:

`/home/jovyan/Visualization/data/raw_data/train(1)/train/vqa_data`

Default output:

`/home/jovyan/Visualization/data/token_level_output/<model>/`

Run one sample first:

```bash
cd /home/jovyan/Visualization/processor/Token_Level_Attention
python CLIP.py --limit 1
python BLIP2.py --limit 1
python LLaVA.py --limit 1
python Qwen2-VL.py --limit 1
```

Each model writes only to its same-name directory:

```text
data/token_level_output/
├── BLIP2/
├── CLIP/
├── LLaVA/
└── Qwen2-VL/
```

Each sample contains `token_*/layer_*.png` heatmaps and a `metadata.json`.
The model directory also contains `results.jsonl` and `summary.json`.

By default, only the last-layer heatmap is saved for each token. All available
layers are still used to calculate adjacent-layer L1/KL values stored in the
JSON output. Use `--layers first,middle,last`, explicit zero-based indices such
as `--layers 0,6,12`, or `--save-all-layers` when additional PNGs are needed.

## Inter-model L1/KL differences

After model outputs exist, calculate differences without loading any model:

```bash
python calculate_model_differences.py
```

The four architectures do not expose equivalent token types: CLIP uses question
tokens, BLIP2 uses learned query tokens, and LLaVA/Qwen2-VL use generated answer
tokens. Therefore, the comparator aligns slices by nearest relative token and
layer position, records the selected source indices, then computes L1 and
symmetric KL over normalized 14x14 maps.
