import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from config import conf
from distill_model import StudentBiLSTM, get_num_params
from utils import build_dataloader, evaluate_student, init_student_vocab


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_baseline():
    set_seed(conf.seed)

    print("=" * 70)
    print("BiLSTM+Attention 基线训练（无蒸馏，直接硬标签训练）")
    print("  学生模型: BiLSTM+Attention (来自 03-bilstm)")
    print("  对比目的: 验证蒸馏是否带来性能提升")
    print("=" * 70)

    print("\n[1/4] 加载数据...")
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    vocab_size = len(conf.student_vocab)
    print(f"词表大小: {vocab_size}")

    print("\n[2/4] 初始化学生模型 (BiLSTM+Attention)...")
    student = StudentBiLSTM(vocab_size)
    student.to(conf.device)

    student_params = get_num_params(student)
    print(f"学生参数量: {student_params:,}")

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    save_dir = "../save_models"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    baseline_model_path = os.path.join(save_dir, "bilstm_baseline.pt")

    best_f1 = 0
    best_epoch = 0
    history = []

    print("\n" + "=" * 70)
    print("开始基线训练")
    print(f"  学习率: {conf.learning_rate}, batch_size: {conf.batch_size}")
    print("=" * 70)

    for epoch in range(conf.num_epochs):
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch + 1}/{conf.num_epochs}")
        print(f"{'─' * 70}")

        student.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        train_bar = tqdm(train_loader, desc="  [Train]")
        for batch in train_bar:
            _, _, student_input_ids, labels = batch
            student_input_ids = student_input_ids.to(conf.device)
            labels = labels.to(conf.device)

            outputs = student(student_input_ids)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        from sklearn.metrics import accuracy_score, f1_score
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="macro")

        dev_acc, dev_f1, _, _ = evaluate_student(student, dev_loader, conf.device)

        print(f"\n  训练集 - Loss: {avg_loss:.4f}  Acc: {train_acc:.4f}  Macro-F1: {train_f1:.4f}")
        print(f"  验证集 - Acc:  {dev_acc:.4f}  Macro-F1: {dev_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'dev_acc': dev_acc,
            'dev_f1': dev_f1
        })

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            torch.save(student.state_dict(), baseline_model_path)
            print(f"  ★ 最优模型已保存 (Dev F1={best_f1:.4f})")

    print(f"\n最佳模型: Epoch {best_epoch}, Dev F1={best_f1:.4f}")

    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    best_student = StudentBiLSTM(vocab_size)
    best_student.load_state_dict(torch.load(baseline_model_path, map_location=conf.device))
    best_student.to(conf.device)

    test_acc, test_f1, test_preds, test_labels = evaluate_student(
        best_student, test_loader, conf.device
    )

    print(f"\n测试集 Acc:  {test_acc:.4f}")
    print(f"测试集 Macro-F1: {test_f1:.4f}")

    print("\n分类报告:")
    report = classification_report(test_labels, test_preds, target_names=conf.class_list)
    print(report)

    with open("../logs/test_report_baseline.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BiLSTM基线训练测试报告（无蒸馏）\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"学生模型: BiLSTM+Attention\n")
        f.write(f"参数量: {student_params:,}\n")
        f.write(f"学习率: {conf.learning_rate}\n")
        f.write(f"Batch Size: {conf.batch_size}\n")
        f.write(f"最佳Epoch: {best_epoch}\n\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n\n")
        f.write(report)
        f.write("\n\n训练历史:\n")
        for h in history:
            f.write(f"Epoch {h['epoch']:2d}: "
                    f"train_loss={h['train_loss']:.4f}, "
                    f"train_acc={h['train_acc']:.4f}, "
                    f"dev_acc={h['dev_acc']:.4f}, "
                    f"dev_f1={h['dev_f1']:.4f}\n")

    print(f"\n测试报告已保存: ../logs/test_report_baseline.txt")
    print(f"基线模型已保存: {baseline_model_path}")
    print("\n基线训练完成！")
    return best_student


if __name__ == "__main__":
    train_baseline()
