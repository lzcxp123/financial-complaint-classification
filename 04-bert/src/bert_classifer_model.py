import torch
import torch.nn as nn
from transformers import BertModel
from config import conf


class BertClassifier(nn.Module):
    """BERT文本分类模型（优化版）
    结构：BERT(预训练) → [CLS]向量 → Multi-Sample Dropout → Linear(分类层)
    优化点：
      - Multi-Sample Dropout (5个Dropout取平均)，提升泛化
      - 层衰减学习率（在train.py中实现）
      - 仅冻结前2层，大部分参数参与训练
    """

    def __init__(self):
        super(BertClassifier, self).__init__()

        # 第一步：加载预训练BERT模型
        self.bert = BertModel.from_pretrained(conf.bert_path)

        # 第二步：冻结BERT前2层（数据量大，大部分层参与训练）
        self._freeze_layers()

        # 第三步：Multi-Sample Dropout (提升泛化能力)
        self.dropout_num = 5
        self.dropouts = nn.ModuleList([
            nn.Dropout(conf.hidden_dropout_prob) for _ in range(self.dropout_num)
        ])

        # 第四步：分类层
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def _freeze_layers(self):
        """冻结指定层"""
        for name, param in self.bert.named_parameters():
            # 检查是否是需要冻结的层
            for freeze_layer in conf.freeze_layers:
                if freeze_layer in name:
                    param.requires_grad = False
                    break

    def forward(self, input_ids, attention_mask):
        """前向传播
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            out: [batch_size, num_classes]
        """
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # 取[CLS]位置的输出（pooled output）
        pooled = outputs[1]  # [batch_size, hidden_size]

        # Multi-Sample Dropout: 多个dropout取平均，提升泛化
        logits_sum = 0
        for dropout in self.dropouts:
            dropped = dropout(pooled)
            logits_sum += self.fc(dropped)
        out = logits_sum / len(self.dropouts)  # 平均

        return out


def get_num_params(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # 测试模型
    print("加载FinBERT模型...")
    model = BertClassifier()
    print("模型结构:")
    print(model)

    # 统计参数量
    total_params, trainable_params = get_num_params(model)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")

    # 测试前向传播
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, attention_mask)
    print(f"\n输入形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过！")
