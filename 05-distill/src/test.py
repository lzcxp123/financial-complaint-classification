"""
蒸馏BiLSTM 测试集评估脚本
用法: python test.py
功能: 加载蒸馏训练好的BiLSTM学生模型，在测试集上评估 Acc、Macro-F1、分类报告
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import conf
from distill_model import StudentBiLSTM
from utils import build_dataloader, evaluate_student, init_student_vocab


def test():
    """在测试集上评估蒸馏后的BiLSTM学生模型"""
    print("=" * 70)
    print("蒸馏模型 (FinBERT -> BiLSTM+Attention) 测试集评估")
    print("=" * 70)

    # 第一步：加载词表
    init_student_vocab()
    vocab_size = len(conf.student_vocab)
    print(f"词表大小: {vocab_size}")

    # 第二步：加载蒸馏后模型
    if not os.path.exists(conf.model_save_path):
        raise FileNotFoundError(f"蒸馏模型文件不存在: {conf.model_save_path}，请先运行 train_distill.py")

    model = StudentBiLSTM(vocab_size)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    model.to(conf.device)
    model.eval()
    print(f"蒸馏模型已加载: {conf.model_save_path}")

    # 第三步：加载测试集
    _, _, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    # 第四步：批量预测
    all_preds = []
    all_labels = []
    all_probs = []

    test_bar = tqdm(test_loader, desc="测试中")
    with torch.no_grad():
        for batch in test_bar:
            _, _, student_input_ids, labels = batch
            student_input_ids = student_input_ids.to(conf.device)
            labels = labels.to(conf.device)

            outputs = model(student_input_ids)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 第五步：计算指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'=' * 70}")
    print(f"蒸馏BiLSTM 测试集结果")
    print(f"{'=' * 70}")
    print(f"  Accuracy:    {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro-F1:    {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")

    print(f"\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=conf.class_list))

    # 混淆矩阵
    print(f"混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    header = "  " + "  ".join([f"{i}" for i in range(len(conf.class_list))])
    print(header)
    for i, row in enumerate(cm):
        print(f"{i} {row}")

    # 保存结果
    os.makedirs("../logs", exist_ok=True)
    report_path = "../logs/test_report_distill.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"蒸馏BiLSTM 测试集评估报告 (FinBERT -> BiLSTM+Attention)\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Accuracy:    {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"Macro-F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=conf.class_list))
        f.write(f"\n混淆矩阵:\n{cm}\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    test()
