import re
import pickle
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from config import conf


def tokenize_en(text):
    """英文分词：使用正则分词，兼容无NLTK环境"""
    text = text.lower()
    # 正则分词：保留单词、数字、基本标点
    tokens = re.findall(r"[a-zA-Z0-9]+|[.,!?;:'\"()\-]", text)
    return tokens


def build_vocab(texts, min_freq=2, save_path=None):
    """第一步：从训练集构建词表"""
    print("=" * 60)
    print("第一步：构建词表")
    print("=" * 60)

    counter = Counter()
    for text in texts:
        tokens = tokenize_en(text)
        counter.update(tokens)

    # 过滤低频词
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    print(f"总词数: {len(counter)}")
    print(f"过滤低频词(min_freq={min_freq})后词表大小: {len(vocab)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"词表已保存: {save_path}")

    return vocab


def load_vocab(vocab_path):
    """加载词表"""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def process_text(text, vocab, pad_size=256):
    """第二步：文本处理 - 分词→转ID→截断/填充"""
    tokens = tokenize_en(text)

    # 截断或填充
    if len(tokens) >= pad_size:
        tokens = tokens[:pad_size]
    else:
        tokens = tokens + ["<PAD>"] * (pad_size - len(tokens))

    # 转为ID
    ids = [vocab.get(w, 1) for w in tokens]  # 1=<UNK>
    return ids


def load_dataset(file_path):
    """第三步：加载数据集文件"""
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return texts, labels


class TextDataset(Dataset):
    """第四步：自定义Dataset"""

    def __init__(self, texts, labels, vocab, pad_size=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.pad_size = pad_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        ids = process_text(text, self.vocab, self.pad_size)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_data_loaders(vocab):
    """第五步：获取DataLoader"""
    print("\n" + "=" * 60)
    print("第二步：加载数据并创建DataLoader")
    print("=" * 60)

    # 加载数据
    train_texts, train_labels = load_dataset(conf.train_path)
    dev_texts, dev_labels = load_dataset(conf.dev_path)
    test_texts, test_labels = load_dataset(conf.test_path)

    print(f"训练集: {len(train_texts)} 条")
    print(f"验证集: {len(dev_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")

    # 创建Dataset
    train_dataset = TextDataset(train_texts, train_labels, vocab, conf.pad_size)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocab, conf.pad_size)
    test_dataset = TextDataset(test_texts, test_labels, vocab, conf.pad_size)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=conf.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # 测试数据预处理
    print("测试数据预处理模块...")

    # 加载训练集文本构建词表
    train_texts, _ = load_dataset(conf.train_path)
    vocab = build_vocab(train_texts, min_freq=conf.min_freq, save_path=conf.vocab_path)

    # 测试文本处理
    test_text = "I have a problem with my credit card charge."
    ids = process_text(test_text, vocab, pad_size=conf.pad_size)
    print(f"\n测试文本: {test_text}")
    print(f"分词后ID(前20个): {ids[:20]}")
    print(f"序列长度: {len(ids)}")

    print("\n数据预处理模块测试完成！")
