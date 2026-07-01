import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import conf
from distill_model import DistilBERTClassifier, get_num_params
from utils import build_dataloader, evaluate, init_tokenizer


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_baseline():
    """学生模型基线训练（不加蒸馏，直接用硬标签训练）"""
    set_seed(conf.seed)

    print("=" * 60)
    print("DistilBERT 基线训练（无蒸馏）")
    print("=" * 60)

    # ========== 2个初始化 ==========
    # 初始化1：数据
    init_tokenizer()
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    # 初始化2：模型 + 优化器 + 损失函数
    student = DistilBERTClassifier()
    student.to(conf.device)

    student_params = get_num_params(student)
    print(f"\n学生模型参数量: {student_params:,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=conf.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 创建保存目录
    os.makedirs(os.path.dirname(conf.student_model_path), exist_ok=True)

    # ========== 2个遍历 ==========
    best_f1 = 0
    best_epoch = 0

    print("\n" + "=" * 60)
    print("开始基线训练")
    print("=" * 60)

    for epoch in range(conf.num_epochs):
        # 遍历1：epoch遍历
        student.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{conf.num_epochs} [Train]")

        for batch in train_bar:
            # 遍历2：batch遍历
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            # ========== 5个步骤 ==========
            # 第1步：前向传播
            outputs = student(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # 第2步：梯度归零
            optimizer.zero_grad()

            # 第3步：反向传播
            loss.backward()

            # 第4步：参数更新
            optimizer.step()

            # 第5步：记录损失
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # 验证
        dev_acc, dev_f1 = evaluate(student, dev_loader, conf.device)

        print(f"\nEpoch {epoch + 1}/{conf.num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev   Acc:  {dev_acc:.4f}  Macro-F1: {dev_f1:.4f}")

        # 保存最优模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            torch.save(student.state_dict(), conf.student_model_path)
            print(f"  ★ 最优模型已保存 (F1={best_f1:.4f})")

    print(f"\n最佳模型: Epoch {best_epoch}, F1={best_f1:.4f}")

    # ========== 测试集评估 ==========
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    student = DistilBERTClassifier()
    student.load_state_dict(torch.load(conf.student_model_path, map_location=conf.device))
    student.to(conf.device)

    test_acc, test_f1, test_preds, test_labels = evaluate(student, test_loader, conf.device)

    print(f"\n测试集 Acc:  {test_acc:.4f}")
    print(f"测试集 Macro-F1: {test_f1:.4f}")

    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=conf.class_list))

    # 保存测试报告
    report = classification_report(test_labels, test_preds, target_names=conf.class_list)
    with open("../logs/test_report_baseline.txt", "w", encoding="utf-8") as f:
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n\n")
        f.write(report)

    print("\n基线训练完成！")
    return student


if __name__ == "__main__":
    train_baseline()
