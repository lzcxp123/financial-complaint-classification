import torch
import torch.nn as nn
import torch.nn.functional as F
from config import conf


class TextCNN(nn.Module):
    """TextCNN文本分类模型
    结构：Embedding → 多尺度Conv1d → ReLU → MaxPool → Dropout → FC
    """

    def __init__(self, vocab_size):
        super(TextCNN, self).__init__()

        # 第一步：词嵌入层
        self.embedding = nn.Embedding(vocab_size, conf.embed_dim, padding_idx=0)

        # 第二步：多尺度卷积层
        self.convs = nn.ModuleList()
        for filter_size in conf.filter_sizes:
            self.convs.append(
                nn.Conv1d(
                    in_channels=conf.embed_dim,
                    out_channels=conf.num_filters,
                    kernel_size=filter_size
                )
            )

        # 第三步：Dropout层
        self.dropout = nn.Dropout(conf.dropout)

        # 第四步：全连接层
        fc_input_dim = conf.num_filters * len(conf.filter_sizes)
        self.fc = nn.Linear(fc_input_dim, conf.num_classes)

    def conv_and_pool(self, x, conv):
        """卷积+池化"""
        # x: [batch_size, embed_dim, seq_len]
        out = conv(x)  # [batch_size, num_filters, seq_len - filter_size + 1]
        out = F.relu(out)
        out = F.max_pool1d(out, out.size(2))  # [batch_size, num_filters, 1]
        out = out.squeeze(2)  # [batch_size, num_filters]
        return out

    def forward(self, x):
        """前向传播
        Args:
            x: [batch_size, seq_len]
        Returns:
            out: [batch_size, num_classes]
        """
        # 第一步：词嵌入
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embed = embed.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len] 适配Conv1d

        # 第二步：多尺度卷积+池化
        conv_outs = []
        for conv in self.convs:
            conv_out = self.conv_and_pool(embed, conv)
            conv_outs.append(conv_out)

        # 拼接所有卷积结果
        out = torch.cat(conv_outs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]

        # 第三步：Dropout
        out = self.dropout(out)

        # 第四步：全连接层输出
        out = self.fc(out)  # [batch_size, num_classes]

        return out


if __name__ == "__main__":
    # 测试模型
    vocab_size = 10000
    model = TextCNN(vocab_size)
    print("模型结构:")
    print(model)

    # 测试前向传播
    x = torch.randint(0, vocab_size, (32, 256))
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("模型测试通过！")
