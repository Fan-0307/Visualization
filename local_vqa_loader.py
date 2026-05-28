"""
从本地 VQA v2 数据加载样本
替代 HuggingFace datasets
"""

import json
from PIL import Image
from pathlib import Path

class LocalVQADataset:
    def __init__(self, vqa_data_dir, stratified_sample_file=None):
        """
        vqa_data_dir: 包含 questions.json, annotations.json, val2014/ 的目录
        stratified_sample_file: 可选，指定使用分层抽样的 question_ids
        """
        self.vqa_data_dir = Path(vqa_data_dir)

        # 加载 questions
        questions_file = self.vqa_data_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
        with open(questions_file, 'r') as f:
            self.questions_data = json.load(f)

        # 加载 annotations
        annotations_file = self.vqa_data_dir / "v2_mscoco_val2014_annotations.json"
        with open(annotations_file, 'r') as f:
            self.annotations_data = json.load(f)

        # 构建索引
        self.questions = {q['question_id']: q for q in self.questions_data['questions']}
        self.annotations = {a['question_id']: a for a in self.annotations_data['annotations']}

        # 如果指定了分层抽样文件，使用它
        if stratified_sample_file and Path(stratified_sample_file).exists():
            with open(stratified_sample_file, 'r') as f:
                stratified_data = json.load(f)
                self.question_ids = stratified_data['question_ids']
                print(f"✅ 使用分层抽样: {len(self.question_ids)} 个样本")
                print(f"   分布: {stratified_data['distribution']}")
        else:
            # 否则使用全部数据
            self.question_ids = list(self.questions.keys())
            print(f"Loaded {len(self.question_ids)} questions from local VQA v2 data")

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx):
        """
        返回格式与 HuggingFace datasets 兼容
        """
        question_id = self.question_ids[idx]
        question_data = self.questions[question_id]
        annotation_data = self.annotations[question_id]

        # 加载图片
        image_id = question_data['image_id']
        image_filename = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = self.vqa_data_dir / "val2014" / image_filename

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            # 创建一个空白图片作为 fallback
            image = Image.new('RGB', (224, 224), color='gray')

        # 返回与 HuggingFace 格式兼容的字典
        return {
            'image': image,
            'question': question_data['question'],
            'question_id': question_id,
            'image_id': image_id,
            'answers': annotation_data['answers'],
            'question_type': annotation_data['question_type'],
            'answer_type': annotation_data['answer_type'],
            'multiple_choice_answer': annotation_data['multiple_choice_answer']
        }

def load_local_vqa_dataset(vqa_data_dir, stratified_sample_file=None):
    """
    加载本地 VQA v2 数据集

    Args:
        vqa_data_dir: 包含 questions.json, annotations.json, val2014/ 的目录
        stratified_sample_file: 可选，指定使用分层抽样的 question_ids 文件

    Returns:
        LocalVQADataset 实例
    """
    return LocalVQADataset(vqa_data_dir, stratified_sample_file)
