# VLM Attention 提取 - 完整操作指南

## 📋 项目概述

从 CLIP、BLIP2、LLaVA 三个视觉语言模型中提取 attention 数据，用于分析模型在 VQA 任务上的注意力模式。

**关键特性：**
- ✅ 图片和文字**同时输入**模型（不是只有文字）
- ✅ 提取 cross-attention（文字 token 对图像 patch 的注意力）
- ✅ 使用分层抽样，保证分类均衡
- ✅ 输出符合统一规范，便于后续分析


## 📦 前置准备

### 1. 环境依赖

```bash
pip install torch torchvision transformers pillow tqdm numpy
pip install open-clip-torch  # CLIP 需要
```

### 2. 数据准备

**必需文件（已准备好）：**
```
train/vqa_data/
├── val2014/                                    # 200 张 COCO 图片
├── v2_OpenEnded_mscoco_val2014_questions.json  # 问题
├── v2_mscoco_val2014_annotations.json          # 答案+分类
└── stratified_sample_ids.json                  # 分层抽样的样本 ID
```

**验证数据完整性：**
```bash
ls train/vqa_data/val2014/ | wc -l  # 应该显示 200
ls train/vqa_data/*.json | wc -l    # 应该显示 3
```

### 3. 确认配置

检查脚本顶部配置（三个脚本都一样）：

```python
# ============ Configuration ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 200
OUTPUT_DIR = Path("data")
MODEL_NAME = "clip"  # 或 "blip2" 或 "llava"
RANDOM_SEED = 42
VQA_DATA_DIR = Path("train/vqa_data")

# 抽样方式
USE_STRATIFIED_SAMPLING = True  # ✅ 使用分层抽样（分类均衡）
STRATIFIED_SAMPLE_FILE = VQA_DATA_DIR / "stratified_sample_ids.json"
```

**重要：** 必须使用相同的配置，确保处理相同的 200 个样本！

---

## 🚀 执行步骤

### 运行 CLIP

```bash
cd /home/ansiw/try/VisProject
python extract_clip.py
```

**预期输出：**
```
Loading VQA v2 validation set from local files (COCO images)...
📊 使用分层抽样（分类均衡）
✅ 使用分层抽样: 200 个样本
   分布: {'number': 50, 'other': 80, 'yes/no': 70}
Loaded 200 samples from local VQA v2 (based on COCO images)

=== Loading CLIP (OpenCLIP ViT-B-32) on cuda ===

=== Processing 200 samples ===
CLIP Extraction: 100%|████████| 200/200 [15:23<00:00,  4.62s/it]

✅ CLIP extraction complete!
   Processed: 200 samples
   Output dir: data
```

### 运行 BLIP2

```bash
cd /home/ansiw/try/VisProject
python extract_blip2.py
```

**预期输出：**
```
Loading VQA v2 validation set from local files (COCO images)...
📊 使用分层抽样（分类均衡）
✅ 使用分层抽样: 200 个样本
   分布: {'number': 50, 'other': 80, 'yes/no': 70}

=== Loading BLIP2-OPT-2.7b on cuda ===

=== Processing 200 samples ===
BLIP2 Extraction: 100%|████████| 200/200 [28:45<00:00,  8.63s/it]

✅ BLIP2 extraction complete!
   Processed: 200 samples
   Output dir: data
```

### 运行 LLaVA

```bash
cd /home/ansiw/try/VisProject
python extract_llava.py
```

**预期输出：**
```
Loading VQA v2 validation set from local files (COCO images)...
📊 使用分层抽样（分类均衡）
✅ 使用分层抽样: 200 个样本
   分布: {'number': 50, 'other': 80, 'yes/no': 70}

=== Loading LLaVA-v1.6-vicuna-7b on cuda ===

=== Processing 200 samples ===
LLaVA Extraction: 100%|████████| 200/200 [42:18<00:00, 12.69s/it]

✅ LLaVA extraction complete!
   Processed: 200 samples
   Output dir: data
```

---

## 📂 输出结果

运行完后会生成以下目录结构：

```
data/
├── metadata/
│   └── samples.json                    # 全局样本元数据（200 个样本）
│
├── models/
│   ├── clip/                           # 输出
│   │   ├── predictions/                # 200 个预测结果 JSON
│   │   │   ├── vqa_000000.json
│   │   │   ├── vqa_000001.json
│   │   │   └── ...
│   │   ├── attention/                  # 200 个样本的 attention 数据
│   │   │   ├── vqa_000000/
│   │   │   │   ├── layer_8.npy        # 最后 4 层
│   │   │   │   ├── layer_9.npy
│   │   │   │   ├── layer_10.npy
│   │   │   │   └── layer_11.npy
│   │   │   └── ...
│   │   └── summaries/                  # 200 个统计摘要
│   │       ├── vqa_000000.json
│   │       └── ...
│   │
│   ├── blip2/                          # 输出（结构同上）
│   └── llava/                          # 输出（结构同上）
│
└── images/                             # 200 张 COCO 图片
    ├── vqa_000000.jpg
    ├── vqa_000001.jpg
    └── ...
```

---

## 📊 数据格式说明

### 1. Prediction 文件 (`predictions/vqa_000000.json`)

```json
{
  "sample_id": "vqa_000000",
  "model_name": "CLIP-ViT-B32",
  "prediction": "yes",
  "confidence": 0.71,
  "correct": false,
  "inference_time_ms": 183,
  "tokens": ["Is", "the", "tennis", "court", "enclosed", "?"],
  "generated_text": null,
  "error_type": null
}
```

### 2. Attention 文件 (`attention/vqa_000000/layer_11.npy`)

**格式：** NumPy 数组，shape = `[num_heads, text_tokens, 576]`

```python
import numpy as np
attn = np.load('data/models/clip/attention/vqa_000000/layer_11.npy')
print(attn.shape)  # 例如: (12, 6, 576)
# 12 = attention heads
# 6 = 问题的 token 数量
# 576 = 24×24 图像 patches
```

**含义：** 每个文字 token 对 576 个图像 patch 的注意力权重

### 3. Summary 文件 (`summaries/vqa_000000.json`)

```json
{
  "sample_id": "vqa_000000",
  "model_name": "CLIP-ViT-B32",
  "layer_statistics": {
    "layer_8": {
      "entropy": 2.31,
      "sparsity": 0.41,
      "center_bias": 0.67,
      "text_image_ratio": 0.22
    },
    "layer_9": {...}
  },
  "object_coverage": {},
  "dominant_regions": [],
  "attention_behavior": "uniform"
}
```

### 4. Metadata 文件 (`metadata/samples.json`)

```json
[
  {
    "sample_id": "vqa_000000",
    "image_path": "images/vqa_000000.jpg",
    "question": "Is the tennis court enclosed?",
    "gt_answer": "yes",
    "question_type": "is the",
    "answer_type": "yes/no",
    "objects": [],
    "bbox_annotations": {}
  },
  ...
]
```

---

## 🔍 验证结果

运行完成后，使用验证脚本检查数据完整性：

```bash
python merge_metadata.py
```

**预期输出：**
```
✅ 合并完成！共 200 个唯一样本

=== 验证模型数据完整性 ===

CLIP:
  Predictions: 200
  Attention:   200
  Summaries:   200
  ✅ 数据完整！

BLIP2:
  Predictions: 200
  Attention:   200
  Summaries:   200
  ✅ 数据完整！

LLAVA:
  Predictions: 200
  Attention:   200
  Summaries:   200
  ✅ 数据完整！

=== 检查样本一致性 ===
clip: 200 个样本
blip2: 200 个样本
llava: 200 个样本

共同样本数: 200  ✅

=== 数据统计 ===
clip: 85/200 = 42.50%
blip2: 136/200 = 68.00%
llava: 148/200 = 74.00%
```

---

## 📈 分类分布分析

运行分类分析脚本：

```bash
python analyze_classification.py
```

**预期输出：**
```
=== Answer Type 分布 ===
other          :  80 ( 40.0%)
yes/no         :  70 ( 35.0%)
number         :  50 ( 25.0%)

=== 模型在不同 Answer Type 上的准确率 ===

yes/no:
  clip      :  45/ 70 = 64.29%
  blip2     :  58/ 70 = 82.86%
  llava     :  62/ 70 = 88.57%

number:
  clip      :  12/ 50 = 24.00%
  blip2     :  35/ 50 = 70.00%
  llava     :  40/ 50 = 80.00%

other:
  clip      :  28/ 80 = 35.00%
  blip2     :  43/ 80 = 53.75%
  llava     :  46/ 80 = 57.50%
```

---

## 💾 数据打包与分享

### 单个模型打包

```bash
tar -czf clip_data.tar.gz data/models/clip/ data/metadata/ data/images/

tar -czf blip2_data.tar.gz data/models/blip2/ data/metadata/ data/images/

tar -czf llava_data.tar.gz data/models/llava/ data/metadata/ data/images/
```

### 合并所有数据

```bash
# 解压所有数据到同一个 data/ 目录
tar -xzf clip_data.tar.gz
tar -xzf blip2_data.tar.gz
tar -xzf llava_data.tar.gz

# 验证合并结果
python merge_metadata.py
```

---

## ⚠️ 常见问题

### 1. CUDA Out of Memory

**解决方案：**
```python
# 在脚本中减少批处理频率
if (i + 1) % 10 == 0:  # 改为每 10 个样本清理一次
    torch.cuda.empty_cache()
```

### 2. 模型下载失败

**解决方案：**
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 图片加载失败

**检查：**
```bash
# 确认图片存在
ls train/vqa_data/val2014/ | wc -l  # 应该是 200
```

### 4. Attention hook 没有捕获数据

**调试：**
```python
# 在脚本中添加调试输出
print(f"Captured attention layers: {attention_cache.keys()}")
```

---

## 📊 预期数据量

| 项目 | 大小 |
|------|------|
| 单个模型 | ~100 MB |
| 三个模型合计 | ~300 MB |
| 图片 | ~50 MB |
| **总计** | **~350 MB** |

---

## ✅ 核心确认

### 模型输入方式

**✅ 正确：图片 + 文字同时输入**

```python
# CLIP
image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
text_input = clip_tokenizer([question]).to(DEVICE)
image_features = clip_model.encode_image(image_input)  # 图片
text_features = clip_model.encode_text(text_input)     # 文字

# BLIP2
inputs = blip2_processor(images=image, text=question, return_tensors="pt")
outputs = blip2_model.generate(**inputs)

# LLaVA
inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
outputs = llava_model.generate(**inputs)
```

**❌ 错误：只输入文字**
```python
# 这是错误的做法（旧版本）
text_input = tokenizer([question])
outputs = model.generate(text_input)  # 没有图片！
```
