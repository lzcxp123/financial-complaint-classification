import os
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn

from config import conf
from dataEDA_Processing import (
    load_dataset, build_vocab, load_vocab, get_data_loaders
)
from textcnn_model import TextCNN


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, data_loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(conf.device)
            batch_y = batch_y.to(conf.device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


def train_model():
    """训练主函数 - 遵循225原则"""
    set_seed(conf.seed)

    # ========== 2个初始化 ==========
    # 初始化1：数据
    print("=" * 60)
    print("TextCNN 模型训练")
    print("=" * 60)

    # 构建或加载词表
    if os.path.exists(conf.vocab_path):
        print(f"\n加载已有词表: {conf.vocab_path}")
        vocab = load_vocab(conf.vocab_path)
    else:
        train_texts, _ = load_dataset(conf.train_path)
        vocab = build_vocab(train_texts, min_freq=conf.min_freq, save_path=conf.vocab_path)

    print(f"词表大小: {len(vocab)}")

    # 获取DataLoader
    train_loader, dev_loader, test_loader = get_data_loaders(vocab)

    # 初始化2：模型+优化器+损失函数
    model = TextCNN(len(vocab))
    model.to(conf.device)
    print(f"\n模型设备: {conf.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 创建保存目录
    os.makedirs(os.path.dirname(conf.save_model_path), exist_ok=True)
    os.makedirs(conf.result_path, exist_ok=True)

    # ========== 2个遍历 ==========
    best_f1 = 0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(conf.num_epochs):
        # 遍历1：epoch循环
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{conf.num_epochs} [Train]")

        for batch_x, batch_y in train_bar:
            # 遍历2：batch循环
            batch_x = batch_x.to(conf.device)
            batch_y = batch_y.to(conf.device)

            # ========== 5个步骤 ==========
            # 第1步：前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 第2步：梯度归零
            optimizer.zero_grad()

            # 第3步：反向传播
            loss.backward()

            # 第4步：参数更新
            optimizer.step()

            # 第5步：记录损失
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        dev_loss, dev_acc, dev_f1, _, _ = evaluate(model, dev_loader, criterion)

        print(f"\nEpoch {epoch + 1}/{conf.num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev   Loss: {dev_loss:.4f}  Acc: {dev_acc:.4f}  Macro-F1: {dev_f1:.4f}")

        # 保存最优模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), conf.save_model_path)
            print(f"  ★ 最优模型已保存 (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  早停计数: {patience_counter}/{conf.patience}")
            if patience_counter >= conf.patience:
                print(f"\n早停触发！连续 {conf.patience} 个epoch无提升")
                break

    # ========== 测试集评估 ==========
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    model.load_state_dict(torch.load(conf.save_model_path))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)

    print(f"\n测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Acc:  {test_acc:.4f}")
    print(f"测试集 Macro-F1: {test_f1:.4f}")

    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=conf.class_list))

    # 保存测试结果
    report = classification_report(test_labels, test_preds, target_names=conf.class_list)
    with open(os.path.join(conf.result_path, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n\n")
        f.write(report)
    print(f"\n测试报告已保存: {os.path.join(conf.result_path, 'test_report.txt')}")

    print("\n训练完成！")
    return model, vocab


if __name__ == "__main__":
    train_model()
