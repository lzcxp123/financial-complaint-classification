import os
import time
import torch
import pandas as pd

from config import conf
from distill_model import DistilBERTClassifier, get_num_params
from utils import build_dataloader, evaluate, measure_inference_time, init_tokenizer


def load_teacher_model():
    """加载教师模型（FinBERT）"""
    from transformers import BertModel

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(conf.teacher_path)
            self.dropout = torch.nn.Dropout(0.1)
            self.fc = torch.nn.Linear(conf.teacher_hidden_size, conf.num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            pooled = outputs[1]
            pooled = self.dropout(pooled)
            out = self.fc(pooled)
            return out

    teacher = TeacherModel()
    teacher.load_state_dict(torch.load(conf.teacher_model_path, map_location='cpu'))
    teacher.to('cpu')
    teacher.eval()
    return teacher


def load_student_baseline():
    """加载学生模型基线"""
    model = DistilBERTClassifier()
    model.load_state_dict(torch.load(conf.student_model_path, map_location='cpu'))
    model.to('cpu')
    model.eval()
    return model


def load_student_distilled():
    """加载蒸馏后的学生模型"""
    model = DistilBERTClassifier()
    model.load_state_dict(torch.load(conf.model_save_path, map_location='cpu'))
    model.to('cpu')
    model.eval()
    return model


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    acc, f1, _, _ = evaluate(model, test_loader, device)
    return acc, f1


def compare_models():
    """对比三个模型的性能"""
    print("=" * 70)
    print("知识蒸馏效果对比")
    print("=" * 70)

    # 初始化
    init_tokenizer()
    _, _, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, batch_size=16
    )

    results = []

    # ========== 1. 教师模型 (FinBERT) ==========
    print("\n" + "-" * 50)
    print("评估 1: 教师模型 (FinBERT)")
    print("-" * 50)

    if os.path.exists(conf.teacher_model_path):
        teacher = load_teacher_model()
        teacher_params = get_num_params(teacher)
        teacher_time = measure_inference_time(teacher, test_loader, 'cpu', num_samples=50)
        teacher_acc, teacher_f1 = evaluate_model(teacher, test_loader, 'cpu')

        print(f"参数量: {teacher_params:,}")
        print(f"测试准确率: {teacher_acc:.4f}")
        print(f"测试Macro-F1: {teacher_f1:.4f}")
        print(f"推理时间: {teacher_time:.2f} ms/sample")

        results.append({
            'Model': 'Teacher (FinBERT)',
            'Test Acc': teacher_acc,
            'Test Macro-F1': teacher_f1,
            'Params (M)': teacher_params / 1e6,
            'Inference Time (ms)': teacher_time
        })
    else:
        print(f"❌ 教师模型不存在: {conf.teacher_model_path}")

    # ========== 2. 学生模型基线 (DistilBERT 无蒸馏) ==========
    print("\n" + "-" * 50)
    print("评估 2: 学生模型基线 (DistilBERT 无蒸馏)")
    print("-" * 50)

    if os.path.exists(conf.student_model_path):
        baseline = load_student_baseline()
        baseline_params = get_num_params(baseline)
        baseline_time = measure_inference_time(baseline, test_loader, 'cpu', num_samples=50)
        baseline_acc, baseline_f1 = evaluate_model(baseline, test_loader, 'cpu')

        print(f"参数量: {baseline_params:,}")
        print(f"测试准确率: {baseline_acc:.4f}")
        print(f"测试Macro-F1: {baseline_f1:.4f}")
        print(f"推理时间: {baseline_time:.2f} ms/sample")

        if teacher_params:
            print(f"\n参数量压缩: {(1 - baseline_params/teacher_params)*100:.1f}%")
            print(f"推理加速: {teacher_time/baseline_time:.2f}x")

        results.append({
            'Model': 'Student-B (DistilBERT)',
            'Test Acc': baseline_acc,
            'Test Macro-F1': baseline_f1,
            'Params (M)': baseline_params / 1e6,
            'Inference Time (ms)': baseline_time
        })
    else:
        print(f"❌ 学生基线模型不存在: {conf.student_model_path}")
        print("请先运行: python train_student_baseline.py")

    # ========== 3. 学生模型蒸馏 (DistilBERT + 蒸馏) ==========
    print("\n" + "-" * 50)
    print("评估 3: 学生模型蒸馏 (DistilBERT + 蒸馏)")
    print("-" * 50)

    if os.path.exists(conf.model_save_path):
        distilled = load_student_distilled()
        distilled_params = get_num_params(distilled)
        distilled_time = measure_inference_time(distilled, test_loader, 'cpu', num_samples=50)
        distilled_acc, distilled_f1 = evaluate_model(distilled, test_loader, 'cpu')

        print(f"参数量: {distilled_params:,}")
        print(f"测试准确率: {distilled_acc:.4f}")
        print(f"测试Macro-F1: {distilled_f1:.4f}")
        print(f"推理时间: {distilled_time:.2f} ms/sample")

        if teacher_params:
            print(f"\n参数量压缩: {(1 - distilled_params/teacher_params)*100:.1f}%")
            print(f"推理加速: {teacher_time/distilled_time:.2f}x")

        if baseline_f1:
            print(f"\n相比基线提升: Acc {(distilled_acc-baseline_acc)*100:+.2f}%, F1 {(distilled_f1-baseline_f1)*100:+.2f}%")

        results.append({
            'Model': 'Student-D (DistilBERT+KD)',
            'Test Acc': distilled_acc,
            'Test Macro-F1': distilled_f1,
            'Params (M)': distilled_params / 1e6,
            'Inference Time (ms)': distilled_time
        })
    else:
        print(f"❌ 蒸馏模型不存在: {conf.model_save_path}")
        print("请先运行: python train_distill.py")

    # ========== 对比总结 ==========
    if results:
        print("\n" + "=" * 70)
        print("模型对比总结")
        print("=" * 70)

        df = pd.DataFrame(results)
        print("\n")
        print(df.to_string(index=False))

        # 保存CSV
        csv_path = "../logs/distill_comparison.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n对比结果已保存: {csv_path}")

        # 打印对比表格
        print("\n" + "-" * 70)
        print(f"{'Model':<25} {'Acc':<10} {'F1':<10} {'Params(M)':<12} {'Time(ms)':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['Model']:<25} {r['Test Acc']:<10.4f} {r['Test Macro-F1']:<10.4f} "
                  f"{r['Params (M)']:<12.2f} {r['Inference Time (ms)']:<12.2f}")
        print("-" * 70)

    else:
        print("\n❌ 没有找到任何模型，请先训练模型！")

    print("\n对比完成！")


if __name__ == "__main__":
    compare_models()
