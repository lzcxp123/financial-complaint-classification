import torch
import os


class Config(object):
    def __init__(self):
        # 模型名称
        self.model_name = "finbert"

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

        # 模型保存路径
        self.model_save_path = "../save_models/finbert.pt"
        self.quantized_model_path = "../save_models/finbert_quantized.pt"

        # 训练超参数
        self.num_epochs = 15
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01

        # BERT模型配置
        self.bert_path = "../ProsusAI/finbert"
        self.max_seq_length = 256

        # 冻结参数：仅冻结embedding和前2层（数据量大，大部分层参与训练）
        self.freeze_layers = [
            'bert.embeddings',
            'bert.encoder.layer.0',
            'bert.encoder.layer.1',
        ]

        # 隐藏层大小
        self.hidden_size = 768

        # 梯度累积：等效 batch_size = 16 * 4 = 64
        self.gradient_accumulation_steps = 4

        # 层衰减学习率（顶层学习率高，底层低，稳定微调）
        self.layerwise_lr_decay = 0.85

        # 早停
        self.early_stop_patience = 5

        # Dropout
        self.hidden_dropout_prob = 0.1

        # API端口
        self.api_port = 8004

        # 随机种子
        self.seed = 42


conf = Config()
