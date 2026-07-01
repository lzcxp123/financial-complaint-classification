# 任务目标：教学版BERT分类模型构建
# 简化版，展示BERT分类的核心原理
# todo: 理解BERT模型结构和分类流程

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class DemoBertClassifier(nn.Module):
    """教学版BERT分类模型（简化版）

    BERT分类流程：
    1. 输入文本经过Tokenize得到token IDs
    2. BERT编码得到每个token的隐状态
    3. 取[CLS]位置的输出作为句子表示
    4. 通过分类层得到各类别logits
    5. softmax得到类别概率
    """

    def __init__(self, num_classes=9, pretrained_path="ProsusAI/finbert"):
        super().__init__()

        # 1. BERT编码器
        self.bert = BertModel.from_pretrained(pretrained_path)

        # 2. 分类层：768 -> num_classes
        self.classifier = nn.Linear(768, num_classes)

        print(f"BERT分类模型初始化完成")
        print(f"类别数: {num_classes}")

    def forward(self, input_ids, attention_mask):
        """前向传播

        Args:
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 取[CLS]位置的输出（pooled output）
        # 这是BERT预训练中[CLS]token学到的句子级别表示
        pooled_output = outputs[1]  # shape: [batch_size, 768]

        # 通过分类层
        logits = self.classifier(pooled_output)  # shape: [batch_size, num_classes]

        return logits

    def predict(self, input_ids, attention_mask):
        """预测函数"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            return pred, probs


if __name__ == "__main__":
    print("=" * 50)
    print("教学版BERT分类模型演示")
    print("=" * 50)

    # 初始化模型
    model = DemoBertClassifier(num_classes=9)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\n输入形状:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    # 前向传播
    logits = model(input_ids, attention_mask)
    print(f"\n输出形状: {logits.shape}")

    # 预测
    pred, probs = model.predict(input_ids, attention_mask)
    print(f"\n预测结果:")
    print(f"  类别ID: {pred.tolist()}")
    print(f"  类别概率: {probs.tolist()}")

    print("\n教学演示完成！")
