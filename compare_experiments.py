"""
金融消费者投诉文本分类 - 实验对比脚本
自动加载各模型的最优checkpoint，在统一测试集上评估并输出对比表格
适配项目：20万数据、8类、TextCNN/BiLSTM/FinBERT/量化/蒸馏BiLSTM(FinBERT→BiLSTM)

用法:
  python compare_experiments.py              # 快速模式（默认采样1000条，几分钟出结果）
  python compare_experiments.py --n 5000    # 评估前5000条
  python compare_experiments.py --full      # 完整测试集评估（慢）
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "01-data", "test.txt")
CLASS_PATH = os.path.join(BASE_DIR, "01-data", "class.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = device.type == "cuda"
print(f"使用设备: {device}")

# 加载类别列表
class_list = [line.strip() for line in open(CLASS_PATH, encoding="utf-8")]
num_classes = len(class_list)


# ==================== 工具函数 ====================
def load_test_data(max_samples=None):
    """加载测试集"""
    texts = []
    labels = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(int(parts[1]))
                if max_samples and len(texts) >= max_samples:
                    break
    return texts, labels


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    return acc, macro_f1, weighted_f1, per_class_f1


def tokenize_en(text):
    """英文分词（正则分词，与02-textcnn/03-bilstm保持一致）"""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+|[.,!?;:'\"()\-]", text)
    return tokens


def texts_to_ids_batch(texts, vocab, pad_size):
    """批量文本转id序列"""
    ids_list = []
    for text in texts:
        tokens = tokenize_en(text)
        ids = [vocab.get(t, 1) for t in tokens]
        if len(ids) < pad_size:
            ids = ids + [0] * (pad_size - len(ids))
        else:
            ids = ids[:pad_size]
        ids_list.append(ids)
    return ids_list


def _clear_module_cache(*module_names):
    """清理sys.modules中缓存的模块，避免同名模块冲突"""
    for name in module_names:
        if name in sys.modules:
            del sys.modules[name]


# ==================== 模型加载器 ====================
class TextCNNEvaluator:
    """TextCNN模型评估器"""
    name = "TextCNN"
    batch_size = 512

    def __init__(self):
        module_dir = os.path.join(BASE_DIR, "02-textcnn")
        sys.path.insert(0, module_dir)
        _clear_module_cache("config", "textcnn_model")

        old_cwd = os.getcwd()
        os.chdir(module_dir)
        try:
            from config import conf as textcnn_conf
            import pickle
            from textcnn_model import TextCNN

            self.pad_size = textcnn_conf.pad_size
        finally:
            os.chdir(old_cwd)

        vocab_path = os.path.join(BASE_DIR, "02-textcnn", "data", "vocab.pkl")
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.model = TextCNN(len(self.vocab))
        self.model.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "02-textcnn", "save_models", "textcnn.pt"),
            map_location=device
        ))
        self.model.to(device)
        self.model.eval()
        if IS_CUDA:
            self.model.half()

    def predict_batch(self, texts):
        ids_list = texts_to_ids_batch(texts, self.vocab, self.pad_size)
        input_tensor = torch.tensor(ids_list, dtype=torch.long).to(device)
        with torch.inference_mode():
            output = self.model(input_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()
        return preds


class BiLSTMEvaluator:
    """BiLSTM+Attention模型评估器"""
    name = "BiLSTM+Attention"
    batch_size = 512

    def __init__(self):
        module_dir = os.path.join(BASE_DIR, "03-bilstm")
        sys.path.insert(0, module_dir)
        _clear_module_cache("config", "bilstm_model")

        old_cwd = os.getcwd()
        os.chdir(module_dir)
        try:
            from config import conf as bilstm_conf
            import pickle
            from bilstm_model import BiLSTM_Attention

            self.pad_size = bilstm_conf.pad_size
        finally:
            os.chdir(old_cwd)

        vocab_path = os.path.join(BASE_DIR, "03-bilstm", "data", "vocab.pkl")
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.model = BiLSTM_Attention(len(self.vocab))
        self.model.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "03-bilstm", "save_models", "bilstm_attention.pt"),
            map_location=device
        ))
        self.model.to(device)
        self.model.eval()
        if IS_CUDA:
            self.model.half()

    def predict_batch(self, texts):
        ids_list = texts_to_ids_batch(texts, self.vocab, self.pad_size)
        input_tensor = torch.tensor(ids_list, dtype=torch.long).to(device)
        with torch.inference_mode():
            output = self.model(input_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()
        return preds


class FinBERTEvaluator:
    """FinBERT模型评估器"""
    name = "FinBERT"
    batch_size = 128 if IS_CUDA else 32

    def __init__(self):
        bert_path = os.path.join(BASE_DIR, "04-bert", "ProsusAI", "finbert")

        from transformers import BertTokenizer, BertModel

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = 256

        class BertClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained(bert_path)
                self.dropout_num = 5
                self.dropouts = nn.ModuleList([
                    nn.Dropout(0.1) for _ in range(self.dropout_num)
                ])
                self.fc = nn.Linear(768, num_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                pooled = outputs[1]
                logits_sum = 0
                for dropout in self.dropouts:
                    dropped = dropout(pooled)
                    logits_sum += self.fc(dropped)
                out = logits_sum / len(self.dropouts)
                return out

        self.model = BertClassifier()
        self.model.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "04-bert", "save_models", "finbert.pt"),
            map_location=device
        ))
        self.model.to(device)
        self.model.eval()
        # GPU上用FP16加速
        if IS_CUDA:
            self.model.half()

    def predict_batch(self, texts):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        if IS_CUDA:
            input_ids = input_ids
            attention_mask = attention_mask
        with torch.inference_mode():
            output = self.model(input_ids, attention_mask)
            preds = torch.argmax(output, dim=1).cpu().numpy()
        return preds


class FinBERTQuantizedEvaluator:
    """FinBERT量化模型评估器（仅CPU）"""
    name = "FinBERT-INT8"
    batch_size = 64

    def __init__(self):
        bert_path = os.path.join(BASE_DIR, "04-bert", "ProsusAI", "finbert")

        from transformers import BertTokenizer, BertModel

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_len = 256

        class BertClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained(bert_path)
                self.dropout_num = 5
                self.dropouts = nn.ModuleList([
                    nn.Dropout(0.1) for _ in range(self.dropout_num)
                ])
                self.fc = nn.Linear(768, num_classes)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                pooled = outputs[1]
                logits_sum = 0
                for dropout in self.dropouts:
                    dropped = dropout(pooled)
                    logits_sum += self.fc(dropped)
                out = logits_sum / len(self.dropouts)
                return out

        base_model = BertClassifier()
        self.model = torch.quantization.quantize_dynamic(base_model, {nn.Linear}, dtype=torch.qint8)
        self.model.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "04-bert", "save_models", "finbert_quantized.pt"),
            map_location='cpu'
        ))
        self.model.eval()
        self.cpu_device = torch.device('cpu')

    def predict_batch(self, texts):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.cpu_device)
        attention_mask = encoded['attention_mask'].to(self.cpu_device)
        with torch.inference_mode():
            output = self.model(input_ids, attention_mask)
            preds = torch.argmax(output, dim=1).cpu().numpy()
        return preds


class DistilledBiLSTMEvaluator:
    """蒸馏BiLSTM模型评估器（FinBERT→BiLSTM）"""
    name = "蒸馏BiLSTM"
    batch_size = 512

    def __init__(self):
        module_dir = os.path.join(BASE_DIR, "05-distill", "src")
        sys.path.insert(0, module_dir)
        _clear_module_cache("config", "distill_model")

        old_cwd = os.getcwd()
        os.chdir(module_dir)
        try:
            from config import conf as distill_conf
            from distill_model import StudentBiLSTM
            import pickle

            self.pad_size = distill_conf.student_pad_size
        finally:
            os.chdir(old_cwd)

        vocab_path = os.path.join(BASE_DIR, "03-bilstm", "data", "vocab.pkl")
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.model = StudentBiLSTM(len(self.vocab))
        self.model.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "05-distill", "save_models", "bilstm_distilled.pt"),
            map_location=device
        ))
        self.model.to(device)
        self.model.eval()
        if IS_CUDA:
            self.model.half()

    def predict_batch(self, texts):
        ids_list = texts_to_ids_batch(texts, self.vocab, self.pad_size)
        input_tensor = torch.tensor(ids_list, dtype=torch.long).to(device)
        with torch.inference_mode():
            output = self.model(input_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()
        return preds


# ==================== 主评估函数 ====================
def run_evaluation(max_samples=None):
    """运行所有模型评估"""
    print("=" * 80)
    print("金融消费者投诉文本分类 - 实验对比")
    print("=" * 80)

    # 加载测试数据
    print("\n[1/3] 加载测试数据...")
    test_texts, test_labels = load_test_data(max_samples)
    print(f"测试集大小: {len(test_texts)} 条" + ("（采样）" if max_samples else "（完整测试集）"))
    print(f"类别数量: {num_classes} 类")
    print(f"运行设备: {device}")
    print(f"精度: {'FP16 (半精度加速)' if IS_CUDA else 'FP32'}")

    # 定义所有评估器（模型名, 评估器类, 模型文件路径）
    model_configs = [
        ("TextCNN", TextCNNEvaluator, os.path.join(BASE_DIR, "02-textcnn", "save_models", "textcnn.pt")),
        ("BiLSTM+Attention", BiLSTMEvaluator, os.path.join(BASE_DIR, "03-bilstm", "save_models", "bilstm_attention.pt")),
        ("FinBERT", FinBERTEvaluator, os.path.join(BASE_DIR, "04-bert", "save_models", "finbert.pt")),
        ("FinBERT-INT8", FinBERTQuantizedEvaluator, os.path.join(BASE_DIR, "04-bert", "save_models", "finbert_quantized.pt")),
        ("蒸馏BiLSTM", DistilledBiLSTMEvaluator, os.path.join(BASE_DIR, "05-distill", "save_models", "bilstm_distilled.pt")),
    ]

    results = []
    all_preds = {}
    all_per_class_f1 = {}

    print("\n[2/3] 评估各模型...")
    for model_name, evaluator_cls, model_path in model_configs:
        if not os.path.exists(model_path):
            print(f"  ⚠️  {model_name}: 模型文件不存在，跳过")
            continue

        try:
            import time
            t0 = time.time()
            print(f"\n  >>> 评估 {model_name}...")
            evaluator = evaluator_cls()
            batch_size = evaluator_cls.batch_size

            all_preds_list = []
            total_batches = (len(test_texts) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(test_texts), batch_size), total=total_batches,
                         desc=f"  {model_name}", unit="batch"):
                batch_texts = test_texts[i:i + batch_size]
                preds = evaluator.predict_batch(batch_texts)
                all_preds_list.extend(preds)

            elapsed = time.time() - t0
            preds_array = np.array(all_preds_list)
            acc, macro_f1, weighted_f1, per_class_f1 = compute_metrics(test_labels, preds_array)

            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Macro-F1': macro_f1,
                'Weighted-F1': weighted_f1,
            })
            all_preds[model_name] = preds_array
            all_per_class_f1[model_name] = per_class_f1

            print(f"  ✅ {model_name}: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Weighted-F1={weighted_f1:.4f} ({elapsed:.1f}s)")

            # 清理模型释放内存
            del evaluator
            if IS_CUDA:
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  ❌ {model_name} 失败: {e}")
            traceback.print_exc()

    if not results:
        print("\n❌ 没有可评估的模型！请先训练至少一个模型。")
        return

    # 输出对比表格
    print("\n[3/3] 生成对比表格...")
    df = pd.DataFrame(results)
    df = df.sort_values('Macro-F1', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("模型性能对比（8类金融投诉分类）")
    print("=" * 80)
    print(df.to_string(index=False))

    # 保存CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n对比表格已保存: {csv_path}")

    # 保存Markdown
    md_path = os.path.join(OUTPUT_DIR, "model_comparison.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 模型性能对比\n\n")
        f.write("| Model | Accuracy | Macro-F1 | Weighted-F1 |\n")
        f.write("|-------|----------|----------|-------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Macro-F1']:.4f} | {row['Weighted-F1']:.4f} |\n")
        f.write(f"\n*共 {len(df)} 个模型，按 Macro-F1 降序排列*\n")
        f.write(f"*测试集: {len(test_texts)} 条*\n")
    print(f"Markdown表格已保存: {md_path}")

    # 保存每类F1
    per_class_df = pd.DataFrame(all_per_class_f1, index=class_list)
    per_class_csv = os.path.join(OUTPUT_DIR, "per_class_f1.csv")
    per_class_df.to_csv(per_class_csv)
    print(f"每类F1已保存: {per_class_csv}")

    # 保存最优模型的混淆矩阵
    best_model = df.iloc[0]['Model']
    if best_model in all_preds:
        cm = confusion_matrix(test_labels, all_preds[best_model])
        cm_df = pd.DataFrame(cm, index=class_list, columns=class_list)
        cm_csv = os.path.join(OUTPUT_DIR, f"confusion_matrix_{best_model}.csv")
        cm_df.to_csv(cm_csv)
        print(f"最优模型({best_model})混淆矩阵已保存: {cm_csv}")

    print("\n" + "=" * 80)
    print("实验对比完成！")
    print("下一步: python visualize.py 生成可视化图表")
    print("=" * 80)

    return df, all_preds, all_per_class_f1, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金融投诉文本分类 - 模型对比评估")
    parser.add_argument("--n", type=int, default=1000, help="采样N条测试数据（默认1000，快速出结果）")
    parser.add_argument("--full", action="store_true", help="评估完整测试集（慢）")
    args = parser.parse_args()

    max_samples = None if args.full else args.n
    run_evaluation(max_samples=max_samples)
