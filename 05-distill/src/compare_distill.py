import os
import time
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from config import conf
from distill_model import TeacherFinBERT, StudentBiLSTM, get_num_params
from utils import build_dataloader, evaluate_student


def evaluate_teacher_model(model, data_loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            teacher_input_ids, teacher_attention_mask, _, labels = batch
            teacher_input_ids = teacher_input_ids.to(device)
            teacher_attention_mask = teacher_attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(teacher_input_ids, teacher_attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels


def measure_inference_time(model, data_loader, device, is_teacher=False, num_batches=20):
    model.eval()
    times = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            if is_teacher:
                input_ids, attention_mask, _, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                start = time.time()
                _ = model(input_ids, attention_mask)
                end = time.time()
                batch_size = input_ids.size(0)
            else:
                _, _, student_input_ids, _ = batch
                student_input_ids = student_input_ids.to(device)
                start = time.time()
                _ = model(student_input_ids)
                end = time.time()
                batch_size = student_input_ids.size(0)

            times.append((end - start) / batch_size * 1000)

    return sum(times) / len(times) if times else 0


def load_teacher_model():
    teacher = TeacherFinBERT()
    teacher.load_state_dict(
        torch.load(conf.teacher_model_path, map_location=conf.device)
    )
    teacher.to(conf.device)
    teacher.eval()
    return teacher


def load_student_baseline():
    vocab_size = len(conf.student_vocab)
    model = StudentBiLSTM(vocab_size)
    baseline_path = "../save_models/bilstm_baseline.pt"
    model.load_state_dict(torch.load(baseline_path, map_location=conf.device))
    model.to(conf.device)
    model.eval()
    return model


def load_student_distilled():
    vocab_size = len(conf.student_vocab)
    model = StudentBiLSTM(vocab_size)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    model.to(conf.device)
    model.eval()
    return model


def compare_models():
    print("=" * 80)
    print("知识蒸馏效果对比")
    print("  教师模型: FinBERT (ProsusAI/finbert)")
    print("  学生模型: BiLSTM+Attention")
    print("  对比: Teacher vs Student-Baseline vs Student-Distilled")
    print("=" * 80)

    print("\n[1/3] 加载数据...")
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, batch_size=16
    )

    results = []

    teacher_path = conf.teacher_model_path
    baseline_path = "../save_models/bilstm_baseline.pt"
    distilled_path = conf.model_save_path

    # 1. 教师模型
    print("\n" + "-" * 60)
    print("[1/3] 评估教师模型 (FinBERT)...")
    print("-" * 60)
    if os.path.exists(teacher_path):
        try:
            teacher = load_teacher_model()
            teacher_params = get_num_params(teacher)

            teacher_acc, teacher_f1, teacher_preds, teacher_labels = evaluate_teacher_model(
                teacher, test_loader, conf.device
            )
            teacher_time = measure_inference_time(teacher, test_loader, conf.device, is_teacher=True)

            results.append({
                'Model': 'Teacher (FinBERT)',
                'Accuracy': teacher_acc,
                'Macro-F1': teacher_f1,
                'Params (M)': round(teacher_params / 1e6, 1),
                'Inference Time (ms)': round(teacher_time, 2)
            })
            print(f"  Acc: {teacher_acc:.4f}, Macro-F1: {teacher_f1:.4f}")
            print(f"  参数量: {teacher_params:,} ({teacher_params/1e6:.1f}M)")
            print(f"  单条推理: {teacher_time:.2f} ms")
            del teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ 教师模型评估失败: {e}")
    else:
        print(f"  ⚠️  教师模型不存在: {teacher_path}")

    # 2. 学生基线模型
    print("\n" + "-" * 60)
    print("[2/3] 评估学生基线模型 (BiLSTM, 无蒸馏)...")
    print("-" * 60)
    if os.path.exists(baseline_path):
        try:
            student_base = load_student_baseline()
            base_params = get_num_params(student_base)

            base_acc, base_f1, base_preds, base_labels = evaluate_student(
                student_base, test_loader, conf.device
            )
            base_time = measure_inference_time(student_base, test_loader, conf.device)

            results.append({
                'Model': 'Student-Baseline (BiLSTM)',
                'Accuracy': base_acc,
                'Macro-F1': base_f1,
                'Params (M)': round(base_params / 1e6, 1),
                'Inference Time (ms)': round(base_time, 2)
            })
            print(f"  Acc: {base_acc:.4f}, Macro-F1: {base_f1:.4f}")
            print(f"  参数量: {base_params:,} ({base_params/1e6:.1f}M)")
            print(f"  单条推理: {base_time:.2f} ms")
            del student_base
        except Exception as e:
            print(f"  ❌ 基线模型评估失败: {e}")
    else:
        print(f"  ⚠️  基线模型不存在: {baseline_path}")
        print("  请先运行: python train_student_baseline.py")

    # 3. 学生蒸馏模型
    print("\n" + "-" * 60)
    print("[3/3] 评估学生蒸馏模型 (BiLSTM, 有蒸馏)...")
    print("-" * 60)
    if os.path.exists(distilled_path):
        try:
            student_dist = load_student_distilled()
            dist_params = get_num_params(student_dist)

            dist_acc, dist_f1, dist_preds, dist_labels = evaluate_student(
                student_dist, test_loader, conf.device
            )
            dist_time = measure_inference_time(student_dist, test_loader, conf.device)

            results.append({
                'Model': 'Student-Distilled (BiLSTM)',
                'Accuracy': dist_acc,
                'Macro-F1': dist_f1,
                'Params (M)': round(dist_params / 1e6, 1),
                'Inference Time (ms)': round(dist_time, 2)
            })
            print(f"  Acc: {dist_acc:.4f}, Macro-F1: {dist_f1:.4f}")
            print(f"  参数量: {dist_params:,} ({dist_params/1e6:.1f}M)")
            print(f"  单条推理: {dist_time:.2f} ms")
            del student_dist
        except Exception as e:
            print(f"  ❌ 蒸馏模型评估失败: {e}")
    else:
        print(f"  ⚠️  蒸馏模型不存在: {distilled_path}")
        print("  请先运行: python train_distill.py")

    if not results:
        print("\n❌ 没有可评估的模型！请先训练至少一个模型。")
        return

    print("\n" + "=" * 80)
    print("对比结果汇总")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    os.makedirs("../outputs", exist_ok=True)
    csv_path = "../outputs/distill_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n对比表格已保存: {csv_path}")

    md_path = "../outputs/distill_comparison.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 知识蒸馏效果对比\n\n")
        f.write("| Model | Accuracy | Macro-F1 | Params (M) | Inference Time (ms) |\n")
        f.write("|-------|----------|----------|------------|---------------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Macro-F1']:.4f} | {row['Params (M)']} | {row['Inference Time (ms)']} |\n")
        f.write(f"\n*蒸馏配置: alpha={conf.alpha}, temperature={conf.temperature}*\n")
        f.write(f"*教师模型: FinBERT, 学生模型: BiLSTM+Attention*\n")
    print(f"Markdown表格已保存: {md_path}")

    if len(results) >= 2:
        teacher_f1_val = None
        dist_f1_val = None
        base_f1_val = None
        for r in results:
            if 'Teacher' in r['Model']:
                teacher_f1_val = r['Macro-F1']
            elif 'Distilled' in r['Model']:
                dist_f1_val = r['Macro-F1']
            elif 'Baseline' in r['Model']:
                base_f1_val = r['Macro-F1']

        print("\n" + "=" * 80)
        print("蒸馏效果分析")
        print("=" * 80)
        if teacher_f1_val and dist_f1_val:
            retention = dist_f1_val / teacher_f1_val * 100
            print(f"  学生模型保留教师模型精度: {retention:.1f}%")
        if base_f1_val and dist_f1_val:
            improvement = (dist_f1_val - base_f1_val) / base_f1_val * 100
            print(f"  蒸馏相比基线提升: {improvement:+.2f}%")
        if teacher_f1_val and dist_f1_val:
            teacher_params = [r['Params (M)'] for r in results if 'Teacher' in r['Model']][0]
            dist_params = [r['Params (M)'] for r in results if 'Distilled' in r['Model']][0]
            compression = (1 - dist_params / teacher_params) * 100
            print(f"  模型压缩率: {compression:.1f}%")

    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)

    return df


if __name__ == "__main__":
    compare_models()
