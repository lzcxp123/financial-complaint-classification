import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import conf


# =============================================================
# 教师模型：FinBERT（来自04-bert）
# =============================================================
class TeacherFinBERT(nn.Module):
    """教师模型：FinBERT文本分类器
    结构：BERT → [CLS] → Multi-Sample Dropout → FC
    """

    def __init__(self):
        super(TeacherFinBERT, self).__init__()
        self.bert = BertModel.from_pretrained(conf.teacher_bert_path)

        # Multi-Sample Dropout（与04-bert一致）
        self.dropout_num = 5
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(self.dropout_num)
        ])
        self.fc = nn.Linear(conf.teacher_hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled = outputs[1]

        # Multi-Sample Dropout
        logits_sum = 0
        for dropout in self.dropouts:
            dropped = dropout(pooled)
            logits_sum += self.fc(dropped)
        out = logits_sum / len(self.dropouts)

        return out


def load_teacher_model():
    """加载已训练的教师模型（FinBERT）"""
    teacher = TeacherFinBERT()
    teacher.load_state_dict(
        torch.load(conf.teacher_model_path, map_location=conf.device)
    )
    teacher.to(conf.device)
    teacher.eval()

    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"教师模型 (FinBERT) 加载完成")
    print(f"  参数量: {total_params:,}")

    return teacher


# =============================================================
# 学生模型：BiLSTM+Attention（来自03-bilstm）
# =============================================================
class StudentBiLSTM(nn.Module):
    """学生模型：BiLSTM+Attention文本分类器
    结构：Embedding → BiLSTM(2层) → Attention → Dropout → FC
    """

    def __init__(self, vocab_size):
        super(StudentBiLSTM, self).__init__()

        # 第一步：词嵌入层
        self.embedding = nn.Embedding(vocab_size, conf.student_embed_dim, padding_idx=0)

        # 第二步：双向LSTM层
        self.lstm = nn.LSTM(
            input_size=conf.student_embed_dim,
            hidden_size=conf.student_hidden_size,
            num_layers=conf.student_num_layers,
            bidirectional=True,
            dropout=conf.student_lstm_dropout if conf.student_num_layers > 1 else 0.0,
            batch_first=True
        )

        # 第三步：Attention层（可学习的query向量方式）
        self.attention = nn.Linear(conf.student_hidden_size * 2, 1)

        # 第四步：Dropout层
        self.dropout = nn.Dropout(conf.student_dropout)

        # 第五步：全连接层
        self.fc = nn.Linear(conf.student_hidden_size * 2, conf.num_classes)

    def forward(self, x):
        """前向传播
        Args:
            x: [batch_size, seq_len]
        Returns:
            out: [batch_size, num_classes]
        """
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)

        # Attention
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Dropout + FC
        context = self.dropout(context)
        out = self.fc(context)

        return out


def load_student_model(vocab_size, load_pretrained=True):
    """加载学生模型（BiLSTM+Attention）
    Args:
        vocab_size: 词表大小
        load_pretrained: 是否加载03-bilstm预训练权重
    """
    student = StudentBiLSTM(vocab_size)

    if load_pretrained and conf.student_model_path:
        try:
            student.load_state_dict(
                torch.load(conf.student_model_path, map_location=conf.device)
            )
            print(f"学生模型 (BiLSTM+Attention) 加载预训练权重成功")
        except Exception as e:
            print(f"学生模型预训练权重加载失败，将从头训练: {e}")
    else:
        print(f"学生模型 (BiLSTM+Attention) 随机初始化")

    student.to(conf.device)

    total_params = sum(p.numel() for p in student.parameters())
    print(f"  参数量: {total_params:,}")

    return student


# =============================================================
# 工具函数
# =============================================================
def get_num_params(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("蒸馏模型测试")
    print("=" * 60)

    # 教师模型测试
    print("\n[教师模型] FinBERT...")
    teacher = TeacherFinBERT()
    teacher_params = get_num_params(teacher)
    print(f"  参数量: {teacher_params:,}")

    batch_size = 2
    seq_len = 128
    t_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    t_attention_mask = torch.ones(batch_size, seq_len)
    t_output = teacher(t_input_ids, t_attention_mask)
    print(f"  输入: {t_input_ids.shape}")
    print(f"  输出: {t_output.shape}")

    # 学生模型测试
    print("\n[学生模型] BiLSTM+Attention...")
    vocab_size = 10000
    student = StudentBiLSTM(vocab_size)
    student_params = get_num_params(student)
    print(f"  参数量: {student_params:,}")
    print(f"  压缩比: {student_params/teacher_params*100:.1f}%")

    s_input = torch.randint(0, vocab_size, (batch_size, 256))
    s_output = student(s_input)
    print(f"  输入: {s_input.shape}")
    print(f"  输出: {s_output.shape}")

    print("\n模型测试通过！")
