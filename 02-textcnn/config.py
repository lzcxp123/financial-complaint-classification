import torch
import os


class Config(object):
    def __init__(self):
        # 模型名称
        self.model_name = "textcnn"

        # 数据路径
        self.data_path = "../01-data"
        self.train_path = os.path.join(self.data_path, "train.txt")
        self.dev_path = os.path.join(self.data_path, "dev.txt")
        self.test_path = os.path.join(self.data_path, "test.txt")
        self.class_path = os.path.join(self.data_path, "class.txt")

        # 类别列表
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]
        self.num_classes = len(self.class_list)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 保存路径
        self.save_model_path = os.path.join("./save_models", "textcnn.pt")
        self.vocab_path = os.path.join("./data", "vocab.pkl")
        self.result_path = "./result"

        # 超参数
        self.num_epochs = 20
        self.batch_size = 64
        self.pad_size = 256
        self.learning_rate = 1e-3
        self.embed_dim = 128
        self.num_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.dropout = 0.5
        self.min_freq = 2

        # 早停
        self.patience = 5

        # API端口
        self.api_port = 8002

        # 随机种子
        self.seed = 42


conf = Config()
