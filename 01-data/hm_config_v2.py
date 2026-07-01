import os


class Config(object):
    def __init__(self):
        # 原始数据路径
        self.raw_data_path = "../complaints.csv"

        # 数据目录
        self.data_dir = "./"

        # 输出文件路径
        self.train_path = os.path.join(self.data_dir, "train.txt")
        self.dev_path = os.path.join(self.data_dir, "dev.txt")
        self.test_path = os.path.join(self.data_dir, "test.txt")
        self.class_path = os.path.join(self.data_dir, "class.txt")
        self.stopwords_path = os.path.join(self.data_dir, "stopwords.txt")

        # -------------------------------------------------------
        # v2 核心改动: 类别合并映射
        # "Payday & short-term loans" 合并到 "Consumer loans"
        # 从 9 类缩减为 8 类, 彻底消灭 198 条的超小类
        # -------------------------------------------------------
        self.category_mapping = {
            # 信用报告类
            "Credit reporting": "Credit reporting",
            "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
            "Credit reporting or other personal consumer reports": "Credit reporting",

            # 支付卡类
            "Credit card": "Payment cards",
            "Prepaid card": "Payment cards",
            "Credit card or prepaid card": "Payment cards",

            # 抵押贷款
            "Mortgage": "Mortgage",

            # 债务催收
            "Debt collection": "Debt collection",
            "Debt or credit management": "Debt collection",

            # 银行服务
            "Bank account or service": "Banking services",
            "Checking or savings account": "Banking services",

            # 消费贷款 (合并了 Payday / Title loan / Student / Vehicle)
            "Student loan": "Consumer loans",
            "Vehicle loan or lease": "Consumer loans",
            "Consumer Loan": "Consumer loans",
            "Payday loan": "Consumer loans",
            "Payday loan, title loan, or personal loan": "Consumer loans",
            "Payday loan, title loan, personal loan, or advance loan": "Consumer loans",
            "Title loan": "Consumer loans",

            # 转账与虚拟货币
            "Money transfers": "Money transfers",
            "Money transfer, virtual currency, or money service": "Money transfers",
            "Virtual currency": "Money transfers",

            # 其他金融服务
            "Other financial service": "Other financial services",
            "Other financial services": "Other financial services",
        }

        # v2: 8 个大类
        self.major_categories = [
            "Credit reporting",
            "Payment cards",
            "Mortgage",
            "Debt collection",
            "Banking services",
            "Consumer loans",
            "Money transfers",
            "Other financial services",
        ]

        # 类别数量
        self.num_classes = len(self.major_categories)

        # -------------------------------------------------------
        # v2 核心改动: 均衡采样配置
        # 每类目标数量: 25000 条
        # 总量: 8 × 25000 = 200000 条 (20万)
        # -------------------------------------------------------
        self.n_per_class = 25000

        # 随机种子
        self.random_seed = 42

        # 训练/验证/测试集划分比例
        self.train_ratio = 0.8
        self.dev_ratio = 0.1
        self.test_ratio = 0.1


# 全局单例
conf = Config()
