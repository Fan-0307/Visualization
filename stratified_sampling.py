"""
分层抽样脚本 - 保证分类均衡
"""

import json
import numpy as np
from collections import defaultdict

# 配置
QUESTIONS_FILE = "train/vqa_data/v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FILE = "train/vqa_data/v2_mscoco_val2014_annotations.json"
NUM_SAMPLES = 200
RANDOM_SEED = 42

# 加载数据
with open(QUESTIONS_FILE, 'r') as f:
    questions_data = json.load(f)

with open(ANNOTATIONS_FILE, 'r') as f:
    annotations_data = json.load(f)

# 构建索引
questions = {q['question_id']: q for q in questions_data['questions']}
annotations = {a['question_id']: a for a in annotations_data['annotations']}

# 按 answer_type 分组
answer_type_groups = defaultdict(list)
for qid, ann in annotations.items():
    answer_type_groups[ann['answer_type']].append(qid)

print("=" * 60)
print("原始数据分布")
print("=" * 60)
for atype, qids in sorted(answer_type_groups.items()):
    print(f"{atype:15s}: {len(qids):6d} 个样本")

# 分层抽样：每个 answer_type 按比例抽样
np.random.seed(RANDOM_SEED)

# 目标分布（更均衡）
target_distribution = {
    'yes/no': 70,   # 35%
    'number': 50,   # 25%
    'other': 80     # 40%
}

sampled_question_ids = []

for atype, target_count in target_distribution.items():
    available = answer_type_groups[atype]
    if len(available) < target_count:
        print(f"⚠️  {atype} 只有 {len(available)} 个样本，少于目标 {target_count}")
        target_count = len(available)

    # 从该类型中随机抽样
    sampled = np.random.choice(available, target_count, replace=False)
    sampled_question_ids.extend(sampled.tolist())

print("\n" + "=" * 60)
print(f"分层抽样结果 (总共 {len(sampled_question_ids)} 个样本)")
print("=" * 60)

# 验证分布
from collections import Counter
sampled_answer_types = [annotations[qid]['answer_type'] for qid in sampled_question_ids]
sampled_question_types = [annotations[qid]['question_type'] for qid in sampled_question_ids]

answer_type_count = Counter(sampled_answer_types)
for atype, count in sorted(answer_type_count.items()):
    percentage = count / len(sampled_question_ids) * 100
    print(f"{atype:15s}: {count:3d} ({percentage:5.1f}%)")

print("\n" + "=" * 60)
print("Question Type 分布 (Top 15)")
print("=" * 60)
question_type_count = Counter(sampled_question_types)
for qtype, count in sorted(question_type_count.items(), key=lambda x: -x[1])[:15]:
    percentage = count / len(sampled_question_ids) * 100
    print(f"{qtype:30s}: {count:3d} ({percentage:5.1f}%)")

# 保存抽样的 question_ids
output_file = "train/vqa_data/stratified_sample_ids.json"
with open(output_file, 'w') as f:
    json.dump({
        'question_ids': sampled_question_ids,
        'random_seed': RANDOM_SEED,
        'num_samples': len(sampled_question_ids),
        'distribution': dict(answer_type_count)
    }, f, indent=2)

print(f"\n✅ 抽样结果已保存到: {output_file}")
print("\n使用方法：")
print("  修改提取脚本，使用这个文件中的 question_ids 而不是随机抽样")
