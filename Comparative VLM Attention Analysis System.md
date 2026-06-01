# Comparative VLM Attention Analysis System

# Unified Output Schema Specification (v1)

本规范用于统一所有模型输出、attention 数据、metadata、统计指标与诊断信息。

目标：

1. 支持多模型统一读取
2. 支持 attention difference 分析
3. 支持未来扩展更多视图
4. 支持前端联动
5. 保持数据量可控

------

# 一、目录结构

```txt
data/
│
├── metadata/
│   ├── samples.json
│   └── question_types.json
│
├── models/
│   ├── clip/
│   │   ├── predictions/
│   │   ├── attention/
│   │   └── summaries/
│   │
│   ├── blip2/
│   └── llava/
│
├── embeddings/
│
├── statistics/
│
└── cache/
```

------

# 二、全局 Sample Metadata

文件：

```txt
metadata/samples.json
```

格式：

```json
[
  {
    "sample_id": "vqa_000001",

    "image_path": "images/vqa_000001.jpg",

    "question": "Is the tennis court enclosed?",

    "gt_answer": "yes",

    "question_type": "spatial_relation",

    "answer_type": "yes/no",

    "objects": [
      "tennis court",
      "fence",
      "trees"
    ],

    "bbox_annotations": {
      "tennis court": [120, 85, 410, 350],
      "fence": [100, 60, 430, 370]
    }
  }
]
```

字段说明：

| 字段             | 说明                              |
| ---------------- | --------------------------------- |
| sample_id        | 全局唯一 ID                       |
| image_path       | 图片路径                          |
| question         | VQA 问题                          |
| gt_answer        | 标准答案                          |
| question_type    | counting / spatial / attribute 等 |
| answer_type      | yes/no / number / open            |
| objects          | 图像主要对象                      |
| bbox_annotations | 可选，用于 coverage analysis      |

------

# 三、模型 Prediction 输出

每个模型：

```txt
models/{model_name}/predictions/{sample_id}.json
```

例如：

```txt
models/clip/predictions/vqa_000001.json
```

格式：

```json
{
  "sample_id": "vqa_000001",

  "model_name": "CLIP-ViT-B32",

  "prediction": "no",

  "confidence": 0.71,

  "correct": false,

  "inference_time_ms": 183,

  "tokens": [
    "Is",
    "the",
    "tennis",
    "court",
    "enclosed",
    "?"
  ],

  "generated_text": "No, the court is open.",

  "error_type": "background_bias"
}
```

字段说明：

| 字段              | 说明               |
| ----------------- | ------------------ |
| prediction        | 最终预测           |
| confidence        | softmax/confidence |
| correct           | 是否正确           |
| inference_time_ms | 推理耗时           |
| tokens            | tokenized question |
| generated_text    | 可选               |
| error_type        | 手工或自动标注     |

------

# 四、Attention 数据格式

文件：

```txt
models/{model_name}/attention/{sample_id}/layer_x.npy
```

例如：

```txt
models/clip/attention/vqa_000001/layer_11.npy
```

------

# Attention Shape

必须：

```python
[num_heads, text_tokens, image_patches]
```

例如：

```python
(32, 12, 576)
```

含义：

| 维度          | 说明              |
| ------------- | ----------------- |
| num_heads     | attention heads   |
| text_tokens   | question token 数 |
| image_patches | 图像 patch 数     |

------

# Patch Layout

所有模型：

```txt
24 x 24 = 576 patches
```

patch index：

```txt
0 → top-left
575 → bottom-right
```

必须统一 row-major ordering。

------

# 五、Attention Summary

文件：

```txt
models/{model_name}/summaries/{sample_id}.json
```

格式：

```json
{
  "sample_id": "vqa_000001",

  "model_name": "CLIP-ViT-B32",

  "layer_statistics": {
    "layer_8": {
      "entropy": 2.31,

      "sparsity": 0.41,

      "center_bias": 0.67,

      "text_image_ratio": 0.22
    },

    "layer_9": {
      "entropy": 2.08,

      "sparsity": 0.46,

      "center_bias": 0.61,

      "text_image_ratio": 0.19
    }
  },

  "object_coverage": {
    "tennis court": 0.11,
    "fence": 0.08,
    "trees": 0.52
  },

  "dominant_regions": [
    "background",
    "trees"
  ],

  "attention_behavior": "background_heavy"
}
```

------

# 六、Attention Difference 

用于 heatmap diff。

文件：

```txt
statistics/differences/
```

例如：

```txt
clip_vs_blip2_vqa_000001.json
```

格式：

```json
{
  "sample_id": "vqa_000001",

  "model_a": "clip",

  "model_b": "blip2",

  "difference_statistics": {
    "attention_l1_distance": 0.37,

    "attention_cosine_similarity": 0.52,

    "object_focus_difference": {
      "tennis court": 0.48,
      "fence": 0.43
    }
  },

  "diagnosis": {
    "failure_reason": "background_bias",

    "summary": "CLIP attends more to background trees while BLIP2 focuses on enclosure regions."
  }
}
```

------

# 七、Attention Flow 数据

用于 layer evolution。

文件：

```txt
statistics/flows/
```

格式：

```json
{
  "sample_id": "vqa_000001",

  "model_name": "blip2",

  "tracked_object": "fence",

  "layer_flow": [
    {
      "layer": 37,
      "attention_mass": 0.21
    },
    {
      "layer": 38,
      "attention_mass": 0.34
    },
    {
      "layer": 39,
      "attention_mass": 0.51
    }
  ]
}
```

------

# 八、Head Embedding / Clustering 

文件：

```txt
embeddings/head_features.csv
```

格式：

```csv
model_name,layer,head,entropy,sparsity,center_bias,text_ratio,cluster
clip,11,3,2.31,0.41,0.67,0.22,1
clip,11,4,1.92,0.53,0.71,0.18,2
blip2,39,12,1.41,0.66,0.42,0.37,4
```

用于：

- UMAP
- tSNE
- clustering view

------

# 九、Question-Type Aggregation

文件：

```txt
statistics/question_type_summary.json
```

格式：

```json
{
  "counting": {
    "clip": {
      "accuracy": 0.42,
      "avg_entropy": 2.71
    },

    "blip2": {
      "accuracy": 0.68,
      "avg_entropy": 1.88
    }
  },

  "spatial_relation": {
    "clip": {
      "accuracy": 0.39
    },

    "blip2": {
      "accuracy": 0.74
    }
  }
}
```

------

# 十、Error Matrix

文件：

```txt
statistics/error_matrix.json
```

格式：

```json
{
  "models": [
    "clip",
    "blip2",
    "llava"
  ],

  "samples": [
    {
      "sample_id": "vqa_000001",

      "results": {
        "clip": false,
        "blip2": true,
        "llava": true
      }
    }
  ]
}
```

------

# 十一、推荐只保留的数据

必须保留：

- prediction
- correctness
- last 4 layers attention
- summary statistics
- question metadata

------

# 十二、不建议保留的数据

不要保存：

- full seq_len x seq_len attention
- 所有层
- float64
- 原始 hidden states
- KV cache

否则数据量会爆炸。

------

# 十三、推荐 Attention 提取策略

推荐：

```python
layers = last_4_layers
attention = cross_attention_only
dtype = float16
```

------

# 十四、推荐最终数据规模

| 项目             | 推荐                 |
| ---------------- | -------------------- |
| 模型             | 3                    |
| 样本             | 200                  |
| attention layers | 4                    |
| attention type   | cross-attention only |

------

# 十五、最终系统对应关系

| 视图                 | 使用的数据                 |
| -------------------- | -------------------------- |
| Error Matrix         | error_matrix.json          |
| Attention Heatmap    | attention/*.npy            |
| Attention Difference | differences/*.json         |
| Attention Flow       | flows/*.json               |
| Head Clustering      | head_features.csv          |
| Language Bias View   | question_type_summary.json |
| Diagnosis Panel      | summaries/*.json           |

