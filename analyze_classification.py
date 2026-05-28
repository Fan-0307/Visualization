"""
分析 VQA v2 数据的分类分布
生成 question_type_summary.json (符合 MD 规范)
"""

import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("data")
METADATA_FILE = OUTPUT_DIR / "metadata" / "samples.json"
STATS_DIR = OUTPUT_DIR / "statistics"
STATS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_classification():
    """分析问题类型和答案类型的分布"""

    # 读取 metadata
    with open(METADATA_FILE, 'r') as f:
        samples = json.load(f)

    print(f"总样本数: {len(samples)}")

    # 统计 answer_type 分布
    answer_type_count = defaultdict(int)
    for sample in samples:
        answer_type_count[sample['answer_type']] += 1

    print("\n=== Answer Type 分布 ===")
    for atype, count in sorted(answer_type_count.items(), key=lambda x: -x[1]):
        print(f"{atype:15s}: {count:3d} ({count/len(samples)*100:.1f}%)")

    # 统计 question_type 分布
    question_type_count = defaultdict(int)
    for sample in samples:
        question_type_count[sample['question_type']] += 1

    print("\n=== Question Type 分布 (Top 20) ===")
    for qtype, count in sorted(question_type_count.items(), key=lambda x: -x[1])[:20]:
        print(f"{qtype:30s}: {count:3d} ({count/len(samples)*100:.1f}%)")

    # 按 question_type 统计每个模型的准确率
    models = ['clip', 'blip2', 'llava']
    question_type_summary = {}

    for qtype in question_type_count.keys():
        question_type_summary[qtype] = {}

        for model in models:
            pred_dir = OUTPUT_DIR / "models" / model / "predictions"
            if not pred_dir.exists():
                continue

            correct = 0
            total = 0
            entropy_sum = 0.0

            for sample in samples:
                if sample['question_type'] != qtype:
                    continue

                sample_id = sample['sample_id']
                pred_file = pred_dir / f"{sample_id}.json"

                if pred_file.exists():
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                        if pred_data.get('correct'):
                            correct += 1
                        total += 1

                    # 读取 summary 获取 entropy
                    summ_file = OUTPUT_DIR / "models" / model / "summaries" / f"{sample_id}.json"
                    if summ_file.exists():
                        with open(summ_file, 'r') as f:
                            summ_data = json.load(f)
                            layer_stats = summ_data.get('layer_statistics', {})
                            if layer_stats:
                                # 取最后一层的 entropy
                                last_layer = list(layer_stats.values())[-1]
                                entropy_sum += last_layer.get('entropy', 0.0)

            if total > 0:
                question_type_summary[qtype][model] = {
                    "accuracy": correct / total,
                    "avg_entropy": entropy_sum / total if total > 0 else 0.0,
                    "sample_count": total
                }

    # 保存到 statistics/question_type_summary.json
    output_file = STATS_DIR / "question_type_summary.json"
    with open(output_file, 'w') as f:
        json.dump(question_type_summary, f, indent=2)

    print(f"\n✅ 分类统计已保存到: {output_file}")

    # 打印每个模型在不同 answer_type 上的表现
    print("\n=== 模型在不同 Answer Type 上的准确率 ===")

    for atype in ['yes/no', 'number', 'other']:
        print(f"\n{atype}:")

        for model in models:
            pred_dir = OUTPUT_DIR / "models" / model / "predictions"
            if not pred_dir.exists():
                continue

            correct = 0
            total = 0

            for sample in samples:
                if sample['answer_type'] != atype:
                    continue

                sample_id = sample['sample_id']
                pred_file = pred_dir / f"{sample_id}.json"

                if pred_file.exists():
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                        if pred_data.get('correct'):
                            correct += 1
                        total += 1

            if total > 0:
                accuracy = correct / total
                print(f"  {model:10s}: {correct:3d}/{total:3d} = {accuracy:.2%}")

    # 找出每个模型表现最好和最差的 question_type
    print("\n=== 每个模型的强项和弱项 ===")

    for model in models:
        model_performance = []

        for qtype, model_stats in question_type_summary.items():
            if model in model_stats and model_stats[model]['sample_count'] >= 5:
                model_performance.append((
                    qtype,
                    model_stats[model]['accuracy'],
                    model_stats[model]['sample_count']
                ))

        if model_performance:
            model_performance.sort(key=lambda x: x[1], reverse=True)

            print(f"\n{model.upper()}:")
            print("  强项 (Top 3):")
            for qtype, acc, count in model_performance[:3]:
                print(f"    {qtype:30s}: {acc:.2%} (n={count})")

            print("  弱项 (Bottom 3):")
            for qtype, acc, count in model_performance[-3:]:
                print(f"    {qtype:30s}: {acc:.2%} (n={count})")

if __name__ == "__main__":
    analyze_classification()
