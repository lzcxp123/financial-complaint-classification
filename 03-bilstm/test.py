"""
BiLSTM+Attention 测试集评估脚本
用法: python test.py
功能: 加载训练好的模型，在测试集上评估 Acc、Macro-F1、分类报告
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import conf
from dataEDA_Processing import load_vocab, load_dataset, TextDataset
from torch.utils.data import DataLoader
from bilstm_model import BiLSTM_Attention


def test():
    """在测试集上评估BiLSTM+Attention模型"""
    print("=" * 70)
    print("BiLSTM+Attention 测试集评估")
    print("=" * 70)

    # 第一步：加载词表
    if not os.path.exists(conf.vocab_path):
        raise FileNotFoundError(f"词表文件不存在: {conf.vocab_path}，请先运行 train.py")
    vocab = load_vocab(conf.vocab_path)
    print(f"词表大小: {len(vocab)}")

    # 第二步：加载模型
    if not os.path.exists(conf.save_model_path):
        raise FileNotFoundError(f"模型文件不存在: {conf.save_model_path}，请先运行 train.py")

    model = BiLSTM_Attention(len(vocab))
    model.load_state_dict(torch.load(conf.save_model_path, map_location=conf.device))
    model.to(conf.device)
    model.eval()
    print(f"模型已加载: {conf.save_model_path}")

    # 第三步：加载测试集
    test_texts, test_labels = load_dataset(conf.test_path)
    print(f"测试集数量: {len(test_texts)} 条")

    test_dataset = TextDataset(test_texts, test_labels, vocab, conf.pad_size)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

    # 第四步：批量预测
    all_preds = []
    all_labels = []
    all_probs = []

    test_bar = tqdm(test_loader, desc="测试中")
    with torch.no_grad():
        for batch_x, batch_y in test_bar:
            batch_x = batch_x.to(conf.device)
            batch_y = batch_y.to(conf.device)

            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 第五步：计算指标
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'=' * 70}")
    print(f"BiLSTM+Attention 测试集结果")
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
    os.makedirs(conf.result_path, exist_ok=True)
    report_path = os.path.join(conf.result_path, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BiLSTM+Attention 测试集评估报告\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Accuracy:    {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"Macro-F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=conf.class_list))
        f.write(f"\n混淆矩阵:\n{cm}\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    test()
