# 任务目标：教学版BERT训练脚本（225原则）
# todo: 掌握深度学习训练的标准化流程

import torch
import torch.nn as nn
from demo_bert import DemoBertClassifier
from transformers import BertTokenizer


# ============================================================
# 225原则：2个初始化 + 2个遍历 + 5个步骤
# ============================================================

# ---------- 2个初始化 ----------
# 初始化1：模型
model = DemoBertClassifier(num_classes=9)

# 初始化2：优化器 + 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ---------- 2个遍历 ----------
for epoch in range(2):  # 遍历1：epoch遍历

    model.train()
    epoch_loss = 0

    # 模拟训练数据batch
    num_batches = 10
    for batch_idx in range(num_batches):  # 遍历2：batch遍历

        # 模拟一个batch
        batch_size = 8
        seq_len = 64
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 9, (batch_size,))

        # ---------- 5个步骤 ----------
        # 第1步：前向传播
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # 第2步：梯度归零
        optimizer.zero_grad()

        # 第3步：反向传播
        loss.backward()

        # 第4步：参数更新
        optimizer.step()

        # 第5步：记录损失
        epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / num_batches
    print(f"\nEpoch {epoch+1} 完成 | 平均Loss: {avg_loss:.4f}\n")

print("教学训练完成！")
print("\n225原则总结：")
print("  2个初始化: 模型 + 优化器/损失函数")
print("  2个遍历: epoch循环 + batch循环")
print("  5个步骤: 前向→损失→归零→反向→更新")
