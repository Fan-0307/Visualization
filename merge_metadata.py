"""
合并三个模型的 metadata/samples.json
确保没有重复，并验证三个模型处理了相同的样本
"""

import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("data")
METADATA_FILE = OUTPUT_DIR / "metadata" / "samples.json"

def merge_metadata():
    """合并并验证 samples.json"""

    if not METADATA_FILE.exists():
        print("❌ metadata/samples.json 不存在！")
        return

    # 读取现有的 metadata
    with open(METADATA_FILE, 'r') as f:
        samples = json.load(f)

    # 去重（基于 sample_id）
    unique_samples = {}
    for sample in samples:
        sample_id = sample['sample_id']
        if sample_id not in unique_samples:
            unique_samples[sample_id] = sample

    # 保存去重后的数据
    samples_list = list(unique_samples.values())
    samples_list.sort(key=lambda x: x['sample_id'])

    with open(METADATA_FILE, 'w') as f:
        json.dump(samples_list, f, indent=2)

    print(f"✅ 合并完成！共 {len(samples_list)} 个唯一样本")

    # 验证三个模型的数据完整性
    print("\n=== 验证模型数据完整性 ===")
    models = ['clip', 'blip2', 'llava']

    for model in models:
        model_dir = OUTPUT_DIR / "models" / model
        if not model_dir.exists():
            print(f"⚠️  {model} 目录不存在，跳过")
            continue

        # 检查 predictions
        pred_dir = model_dir / "predictions"
        pred_files = list(pred_dir.glob("*.json")) if pred_dir.exists() else []

        # 检查 attention
        attn_dir = model_dir / "attention"
        attn_dirs = list(attn_dir.glob("vqa_*")) if attn_dir.exists() else []

        # 检查 summaries
        summ_dir = model_dir / "summaries"
        summ_files = list(summ_dir.glob("*.json")) if summ_dir.exists() else []

        print(f"\n{model.upper()}:")
        print(f"  Predictions: {len(pred_files)}")
        print(f"  Attention:   {len(attn_dirs)}")
        print(f"  Summaries:   {len(summ_files)}")

        # 检查是否所有样本都有数据
        missing_pred = []
        missing_attn = []
        missing_summ = []

        for sample in samples_list:
            sample_id = sample['sample_id']

            if not (pred_dir / f"{sample_id}.json").exists():
                missing_pred.append(sample_id)

            if not (attn_dir / sample_id).exists():
                missing_attn.append(sample_id)

            if not (summ_dir / f"{sample_id}.json").exists():
                missing_summ.append(sample_id)

        if missing_pred:
            print(f"  ⚠️  缺失 {len(missing_pred)} 个 predictions")
        if missing_attn:
            print(f"  ⚠️  缺失 {len(missing_attn)} 个 attention")
        if missing_summ:
            print(f"  ⚠️  缺失 {len(missing_summ)} 个 summaries")

        if not (missing_pred or missing_attn or missing_summ):
            print(f"  ✅ 数据完整！")

    # 检查三个模型是否处理了相同的样本
    print("\n=== 检查样本一致性 ===")
    model_samples = {}

    for model in models:
        pred_dir = OUTPUT_DIR / "models" / model / "predictions"
        if pred_dir.exists():
            sample_ids = set([f.stem for f in pred_dir.glob("*.json")])
            model_samples[model] = sample_ids
            print(f"{model}: {len(sample_ids)} 个样本")

    if len(model_samples) >= 2:
        all_models = list(model_samples.keys())
        common_samples = model_samples[all_models[0]]

        for model in all_models[1:]:
            common_samples = common_samples.intersection(model_samples[model])

        print(f"\n共同样本数: {len(common_samples)}")

        # 找出每个模型独有的样本
        for model in all_models:
            unique = model_samples[model] - common_samples
            if unique:
                print(f"⚠️  {model} 独有 {len(unique)} 个样本: {list(unique)[:5]}...")

    print("\n=== 数据统计 ===")

    # 统计正确率
    for model in models:
        pred_dir = OUTPUT_DIR / "models" / model / "predictions"
        if not pred_dir.exists():
            continue

        correct_count = 0
        total_count = 0

        for pred_file in pred_dir.glob("*.json"):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                if pred_data.get('correct'):
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"{model}: {correct_count}/{total_count} = {accuracy:.2%}")

if __name__ == "__main__":
    merge_metadata()
