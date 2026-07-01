import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from config import conf


def init_tokenizer():
    """初始化BertTokenizer"""
    conf.tokenizer = BertTokenizer.from_pretrained(conf.bert_path)


def load_raw_data(filepath):
    """读取 tab 分隔的 text\tlabel 文件"""
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                text, label = parts[0], int(parts[1])
                data.append((text, label))
    return data


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, label


def collate_fn(batch):
    """动态padding的collate函数"""
    texts, labels = zip(*batch)

    # 使用tokenizer进行编码（新版transformers用__call__方法）
    encoded = conf.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=conf.max_seq_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_mask, labels


def build_dataloader(train_path, dev_path, test_path, batch_size=16):
    """构建数据加载器"""
    # 确保tokenizer已初始化
    if not hasattr(conf, 'tokenizer') or conf.tokenizer is None:
        init_tokenizer()

    # 加载数据
    train_data = load_raw_data(train_path)
    dev_data = load_raw_data(dev_path)
    test_data = load_raw_data(test_path)

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 创建Dataset
    train_dataset = TextDataset(train_data, conf.tokenizer)
    dev_dataset = TextDataset(dev_data, conf.tokenizer)
    test_dataset = TextDataset(test_data, conf.tokenizer)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, dev_loader, test_loader


def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1, all_preds, all_labels


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """带warmup的线性学习率调度器"""
    from transformers import get_linear_schedule_with_warmup as linear_warmup
    return linear_warmup(optimizer, num_warmup_steps, num_training_steps)


if __name__ == "__main__":
    # 测试数据加载
    from config import conf

    print("测试数据加载...")
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    # 取一个batch测试
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        print(f"\nBatch形状:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  labels: {labels.shape}")
        break

    print("\n数据加载测试通过！")
