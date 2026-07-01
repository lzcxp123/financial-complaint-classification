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

        # 类别合并映射：22个细粒度类别合并为9个大类
        # 键：原始product字段值，值：合并后的大类名称
        self.category_mapping = {
            # 信用报告类
            "Credit reporting": "Credit reporting",
            "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",

            # 支付卡类
            "Credit card": "Payment cards",
            "Prepaid card": "Payment cards",
            "Credit card or prepaid card": "Payment cards",

            # 抵押贷款
            "Mortgage": "Mortgage",

            # 债务催收
            "Debt collection": "Debt collection",

            # 银行服务
            "Bank account or service": "Banking services",
            "Checking or savings account": "Banking services",

            # 消费贷款
            "Student loan": "Consumer loans",
            "Vehicle loan or lease": "Consumer loans",
            "Consumer Loan": "Consumer loans",

            # 转账与虚拟货币
            "Money transfers": "Money transfers",
            "Money transfer, virtual currency, or money service": "Money transfers",
            "Virtual currency": "Money transfers",

            # 发薪日及短期贷款
            "Payday loan": "Payday & short-term loans",
            "Payday loan, title loan, or personal loan": "Payday & short-term loans",
            "Title loan": "Payday & short-term loans",

            # 其他金融服务
            "Other financial service": "Other financial services",
            "Other financial services": "Other financial services",
        }

        # 大类列表（按顺序，对应label_id）
        self.major_categories = [
            "Credit reporting",
            "Payment cards",
            "Mortgage",
            "Debt collection",
            "Banking services",
            "Consumer loans",
            "Money transfers",
            "Payday & short-term loans",
            "Other financial services",
        ]

        # 类别数量
        self.num_classes = len(self.major_categories)

        # 采样数量
        self.sample_n = 50000

        # 随机种子
        self.random_seed = 42

        # 训练/验证/测试集划分比例
        self.train_ratio = 0.8
        self.dev_ratio = 0.1
        self.test_ratio = 0.1


# 全局单例
conf = Config()
