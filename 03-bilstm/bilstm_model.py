import torch
import torch.nn as nn
import torch.nn.functional as F
from config import conf


class BiLSTM_Attention(nn.Module):
    """BiLSTM+Attention文本分类模型
    结构：Embedding → BiLSTM(2层) → Attention → Dropout → FC
    """

    def __init__(self, vocab_size):
        super(BiLSTM_Attention, self).__init__()

        # 第一步：词嵌入层
        self.embedding = nn.Embedding(vocab_size, conf.embed_dim, padding_idx=0)

        # 第二步：双向LSTM层
        self.lstm = nn.LSTM(
            input_size=conf.embed_dim,
            hidden_size=conf.hidden_size,
            num_layers=conf.num_layers,
            bidirectional=True,
            dropout=conf.lstm_dropout if conf.num_layers > 1 else 0.0,
            batch_first=True
        )

        # 第三步：Attention层（可学习的query向量方式）
        # 输入维度是 hidden_size * 2（双向拼接）
        self.attention = nn.Linear(conf.hidden_size * 2, 1)

        # 第四步：Dropout层
        self.dropout = nn.Dropout(conf.dropout)

        # 第五步：全连接层
        self.fc = nn.Linear(conf.hidden_size * 2, conf.num_classes)

    def forward(self, x):
        """前向传播
        Args:
            x: [batch_size, seq_len]
        Returns:
            out: [batch_size, num_classes]
        """
        # 第一步：词嵌入
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # 第二步：BiLSTM
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_size * 2]

        # 第三步：Attention
        # 计算attention权重
        attn_scores = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]

        # 加权求和得到上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_size * 2]

        # 第四步：Dropout
        context = self.dropout(context)

        # 第五步：全连接层输出
        out = self.fc(context)  # [batch_size, num_classes]

        return out


if __name__ == "__main__":
    # 测试模型
    vocab_size = 10000
    model = BiLSTM_Attention(vocab_size)
    print("模型结构:")
    print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randint(0, vocab_size, (32, 256))
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过！")
