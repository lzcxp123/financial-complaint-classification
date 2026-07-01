import torch
import os


class Config(object):
    def __init__(self):
        # 模型名称
        self.model_name = "bilstm_distilled_from_finbert"

        # 数据路径
        self.data_path = "../../01-data"
        self.train_path = os.path.join(self.data_path, "train.txt")
        self.dev_path = os.path.join(self.data_path, "dev.txt")
        self.test_path = os.path.join(self.data_path, "test.txt")
        self.class_path = os.path.join(self.data_path, "class.txt")

        # 类别列表
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]
        self.num_classes = len(self.class_list)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =====================================================
        # 教师模型配置（FinBERT，来自04-bert）
        # =====================================================
        self.teacher_model_path = "../../04-bert/save_models/finbert.pt"
        self.teacher_bert_path = "../../04-bert/ProsusAI/finbert"
        self.teacher_hidden_size = 768
        self.teacher_max_seq_length = 256

        # =====================================================
        # 学生模型配置（BiLSTM+Attention，来自03-bilstm）
        # =====================================================
        self.student_model_path = "../../03-bilstm/save_models/bilstm_attention.pt"
        self.student_vocab_path = "../../03-bilstm/data/vocab.pkl"
        self.student_pad_size = 256
        self.student_embed_dim = 128
        self.student_hidden_size = 128
        self.student_num_layers = 2
        self.student_dropout = 0.5
        self.student_lstm_dropout = 0.3

        # 蒸馏后学生模型保存路径
        self.model_save_path = "../save_models/bilstm_distilled.pt"

        # =====================================================
        # 训练超参数
        # =====================================================
        self.num_epochs = 1
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5

        # =====================================================
        # 蒸馏超参数
        # =====================================================
        self.alpha = 0.3       # 硬标签损失权重 (0.3*CE + 0.7*KL)
        self.temperature = 4   # 蒸馏温度

        # API端口
        self.api_port = 8005

        # 随机种子
        self.seed = 42


conf = Config()
