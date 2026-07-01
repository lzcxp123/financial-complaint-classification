import re
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import conf


def tokenize_en(text):
    """英文分词：使用正则分词，与03-bilstm保持一致"""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+|[.,!?;:'\"()\-]", text)
    return tokens


def load_vocab(vocab_path):
    """加载词表"""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def init_teacher_tokenizer():
    """初始化教师模型的BertTokenizer"""
    conf.teacher_tokenizer = BertTokenizer.from_pretrained(conf.teacher_bert_path)


def init_student_vocab():
    """初始化学生模型的词表"""
    conf.student_vocab = load_vocab(conf.student_vocab_path)


def text_to_student_ids(text, vocab, pad_size):
    """将文本转为学生模型（BiLSTM）的输入ID序列"""
    tokens = tokenize_en(text)
    if len(tokens) >= pad_size:
        tokens = tokens[:pad_size]
    else:
        tokens = tokens + ["<PAD>"] * (pad_size - len(tokens))
    ids = [vocab.get(w, 1) for w in tokens]  # 1 = <UNK>
    return ids


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
    """文本数据集（保存原始文本，在collate_fn中编码）"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, label


def collate_fn(batch):
    """
    动态编码函数：
    - 教师模型：BertTokenizer编码
    - 学生模型：nltk分词+词表编码
    返回: teacher_input_ids, teacher_attention_mask, student_input_ids, labels
    """
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)

    # ========== 教师模型编码（BERT tokenizer）==========
    if not hasattr(conf, 'teacher_tokenizer') or conf.teacher_tokenizer is None:
        init_teacher_tokenizer()

    teacher_encoded = conf.teacher_tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=conf.teacher_max_seq_length,
        return_tensors='pt'
    )
    teacher_input_ids = teacher_encoded['input_ids']
    teacher_attention_mask = teacher_encoded['attention_mask']

    # ========== 学生模型编码（nltk分词 + 词表）==========
    if not hasattr(conf, 'student_vocab') or conf.student_vocab is None:
        init_student_vocab()

    student_ids_list = []
    for text in texts:
        ids = text_to_student_ids(text, conf.student_vocab, conf.student_pad_size)
        student_ids_list.append(ids)
    student_input_ids = torch.tensor(student_ids_list, dtype=torch.long)

    return teacher_input_ids, teacher_attention_mask, student_input_ids, labels


def build_dataloader(train_path, dev_path, test_path, batch_size=32):
    """构建数据加载器"""
    # 预加载 tokenizer 和 vocab
    init_teacher_tokenizer()
    init_student_vocab()

    # 加载数据
    train_data = load_raw_data(train_path)
    dev_data = load_raw_data(dev_path)
    test_data = load_raw_data(test_path)

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(dev_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 创建Dataset
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    return train_loader, dev_loader, test_loader


def evaluate_student(model, data_loader, device):
    """评估学生模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            _, _, student_input_ids, labels = batch
            student_input_ids = student_input_ids.to(device)
            labels = labels.to(device)

            outputs = model(student_input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, f1, all_preds, all_labels


def get_num_params(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


if __name__ == "__main__":
    print("测试数据加载...")

    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    # 取一个batch测试
    for batch in train_loader:
        teacher_input_ids, teacher_attention_mask, student_input_ids, labels = batch
        print(f"\nBatch形状:")
        print(f"  教师 - input_ids: {teacher_input_ids.shape}, attention_mask: {teacher_attention_mask.shape}")
        print(f"  学生 - input_ids: {student_input_ids.shape}")
        print(f"  labels: {labels.shape}")
        break

    print(f"\n词表大小: {len(conf.student_vocab)}")
    print("数据加载测试通过！")
