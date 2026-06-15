逐 Token、逐层注意力热力图运行说明
================================

一、目录说明
------------

脚本目录：
/home/jovyan/Visualization/processor/Token_Level_Attention

默认输入目录：
/home/jovyan/Visualization/data/raw_data/train(1)/train/vqa_data

默认输出目录：
/home/jovyan/Visualization/data/token_level_output

四个模型会分别输出到以下同名目录：

- CLIP
- BLIP2
- LLaVA
- Qwen2-VL


二、进入脚本目录
----------------

cd /home/jovyan/Visualization/processor/Token_Level_Attention


三、运行四个模型
----------------

建议先使用 --limit 1 测试单个样本，确认模型路径和显存正常后，再运行完整数据。

1. CLIP

单样本测试：
python CLIP.py --limit 1

完整运行：
python CLIP.py

2. BLIP2

单样本测试：
python BLIP2.py --limit 1

完整运行：
python BLIP2.py

3. LLaVA

单样本测试：
python LLaVA.py --limit 1

完整运行：
python LLaVA.py

4. Qwen2-VL

单样本测试：
python Qwen2-VL.py --limit 1

完整运行：
python Qwen2-VL.py


四、常用运行参数
----------------

限制处理样本数量：
--limit 10

指定输入目录：
--data-dir /path/to/vqa_data

指定输出根目录：
--output-root /path/to/token_level_output

指定模型目录：
--model-path /path/to/model

指定最多生成的回答 Token 数：
--max-new-tokens 12

默认情况下，每个 Token 只保存最后一层热力图。所有层仍会用于计算相邻层的 L1/KL 数值。

保存首层、中间层和最后层热力图：
--layers first,middle,last

保存指定层热力图，层编号从 0 开始：
--layers 0,6,12

保存全部层热力图：
--save-all-layers

重新处理已经存在于 results.jsonl 中的样本：
--overwrite

示例：

python Qwen2-VL.py \
  --limit 10 \
  --max-new-tokens 12 \
  --layers first,middle,last \
  --model-path /home/jovyan/fan/hf_model/Qwen2_VL/Qwen2-VL-7B-Instruct


五、输出结构
------------

每个模型的输出目录结构类似：

token_level_output/Qwen2-VL/
  results.jsonl
  summary.json
  <sample_id>/
    metadata.json
    token_000_<token文本>/
      layer_000.png
      layer_001.png
      ...

其中：

- layer_*.png：当前 Token 在所选模型层上的注意力热力图，默认仅最后一层。
- metadata.json：当前样本的 Token、层编号、热力图路径、权重和相邻层 L1/KL 差异。
- results.jsonl：该模型全部样本的逐 Token、逐层结果。
- summary.json：样本数、Token 数和热力图数量汇总。


六、计算模型间 L1/KL 差异
-------------------------

四个模型的热力图全部生成后，运行：

python calculate_model_differences.py

差异结果默认输出到：

/home/jovyan/Visualization/data/token_level_output/model_differences

输出文件：

- results.jsonl：模型两两之间每个切片的 L1、双向 KL 和对称 KL。
- summary.json：差异计算汇总信息。

由于四种模型的 Token 含义和层数不同，差异计算会按照相对 Token 位置和相对层位置进行最近邻对齐。

指定参与比较的模型：

python calculate_model_differences.py --models CLIP LLaVA Qwen2-VL

指定 Token 和层的对齐切片数量：

python calculate_model_differences.py --token-slices 8 --layer-slices 8


七、注意事项
------------

- 运行模型脚本需要可用 GPU 和足够显存。
- CLIP 没有跨模态注意力，其热力图表示逐层文本 Token 与图像 Patch 的投影余弦相似度。
- BLIP2 使用 Q-Former 查询 Token 到图像 Patch 的跨注意力。
- LLaVA 和 Qwen2-VL 使用生成回答 Token 到视觉 Token 的逐层注意力。
- calculate_model_differences.py 只读取已有输出，不加载模型，也不需要 GPU。
- 不建议默认使用 --save-all-layers，否则会生成大量 PNG 文件。
