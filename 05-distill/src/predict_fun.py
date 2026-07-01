"""
蒸馏模型预测函数 (FinBERT -> BiLSTM+Attention)
单例模式加载模型和词表，返回预测结果
"""
import os
import sys
import torch
import pickle
import numpy as np
import re

sys.path.insert(0, os.path.dirname(__file__))
from config import conf
from distill_model import StudentBiLSTM
from utils import init_student_vocab

_model = None
_vocab = None


def _load_model():
    """加载蒸馏后的BiLSTM学生模型和词表"""
    global _model, _vocab

    if _model is None:
        # 加载词表
        init_student_vocab()
        _vocab = conf.student_vocab
        vocab_size = len(_vocab)

        # 加载模型
        if not os.path.exists(conf.model_save_path):
            raise FileNotFoundError(f"蒸馏模型文件不存在: {conf.model_save_path}")

        _model = StudentBiLSTM(vocab_size)
        _model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
        _model.to(conf.device)
        _model.eval()
        print(f"蒸馏模型已加载: {conf.model_save_path}")

    return _model, _vocab


def _tokenize_en(text):
    """英文分词（正则分词，不依赖nltk）"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _text_to_ids(text, vocab, pad_size):
    """文本转id序列"""
    tokens = _tokenize_en(text)
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) < pad_size:
        ids = ids + [0] * (pad_size - len(ids))
    else:
        ids = ids[:pad_size]
    return ids


def predict(text):
    """
    预测单条文本的类别

    Args:
        text: 待预测的英文投诉文本

    Returns:
        dict: {
            "predicted_class": 预测类别名,
            "predicted_id": 预测类别id,
            "confidence": 置信度,
            "top_classes": Top3类别列表,
            "model_type": 模型类型
        }
    """
    model, vocab = _load_model()

    # 第一步：文本编码
    input_ids = _text_to_ids(text, vocab, conf.pad_size)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(conf.device)

    # 第二步：前向传播
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

    # 第三步：构建结果
    top3_ids = torch.topk(probs, k=min(3, conf.num_classes), dim=1).indices[0].tolist()
    top_classes = []
    for idx in top3_ids:
        top_classes.append({
            "class_name": conf.class_list[idx],
            "probability": round(probs[0][idx].item(), 4)
        })

    return {
        "predicted_class": conf.class_list[pred_id],
        "predicted_id": pred_id,
        "confidence": round(confidence, 4),
        "top_classes": top_classes,
        "model_type": "Distilled BiLSTM (Teacher: FinBERT)"
    }


if __name__ == "__main__":
    # 测试
    test_text = "I have a problem with my credit card. There is an unauthorized charge of $500 on my account."
    result = predict(test_text)
    print(f"测试文本: {test_text}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']}")
    print("预测函数测试通过!")
