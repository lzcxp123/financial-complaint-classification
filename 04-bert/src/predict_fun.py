import torch
import torch.nn as nn
from transformers import BertTokenizer
from config import conf
from bert_classifer_model import BertClassifier


_model_full = None
_model_quantized = None
_tokenizer = None


def load_model(quantized=False):
    """加载模型和分词器（单例模式，分别缓存普通模型和量化模型）"""
    global _model_full, _model_quantized, _tokenizer

    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(conf.bert_path)

    if quantized:
        if _model_quantized is None:
            device = torch.device("cpu")
            _model_quantized = torch.quantization.quantize_dynamic(
                BertClassifier(),
                {nn.Linear},
                dtype=torch.qint8
            )
            state_dict = torch.load(conf.quantized_model_path, map_location=device)
            _model_quantized.load_state_dict(state_dict)
            _model_quantized.to(device)
            _model_quantized.eval()
            print(f"量化模型已加载: {conf.quantized_model_path}")
        return _model_quantized, _tokenizer
    else:
        if _model_full is None:
            device = conf.device
            if not torch.cuda.is_available():
                device = torch.device("cpu")
            _model_full = BertClassifier()
            state_dict = torch.load(conf.model_save_path, map_location=device)
            _model_full.load_state_dict(state_dict)
            _model_full.to(device)
            _model_full.eval()
            print(f"模型已加载: {conf.model_save_path}")
        return _model_full, _tokenizer


def predict(text, quantized=False):
    """单文本预测
    Args:
        text: 投诉文本
        quantized: 是否使用量化模型
    Returns:
        dict: 预测结果
    """
    model, tokenizer = load_model(quantized=quantized)

    # 第一步：文本编码
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=conf.max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # 第二步：模型预测
    device = torch.device("cpu") if quantized else conf.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prob = torch.softmax(output, dim=1)
        pred_id = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_id].item()

    # 第三步：构建返回结果
    predicted_class = conf.class_list[pred_id]
    probabilities = prob[0].cpu().numpy().tolist()

    # 各类别概率排序
    class_probs = []
    for i, cls_name in enumerate(conf.class_list):
        class_probs.append({
            "class_name": cls_name,
            "probability": probabilities[i]
        })
    class_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "predicted_id": pred_id,
        "model_type": "quantized" if quantized else "full",
        "top_classes": class_probs[:3]
    }


def predict_batch(texts, quantized=False):
    """批量预测
    Args:
        texts: 文本列表
        quantized: 是否使用量化模型
    Returns:
        list: 预测结果列表
    """
    model, tokenizer = load_model(quantized=quantized)

    # 批量编码
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=conf.max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    device = torch.device("cpu") if quantized else conf.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_ids = torch.argmax(probs, dim=1).cpu().numpy()

    results = []
    for i, text in enumerate(texts):
        pred_id = pred_ids[i]
        confidence = probs[i][pred_id].item()
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_class": conf.class_list[pred_id],
            "confidence": round(confidence, 4),
            "predicted_id": int(pred_id)
        })

    return results


if __name__ == "__main__":
    test_texts = [
        "I have a problem with my credit card. There is an unauthorized charge on my account.",
        "My mortgage payment was not applied correctly. I have been making payments on time.",
        "I am being harassed by debt collectors. They call me multiple times a day."
    ]

    print("注意: 请先运行 train.py 训练模型后再测试预测功能")
    print("\n预测函数使用示例:")
    for text in test_texts:
        print(f"\n文本: {text[:60]}...")
        print("结果: (需要先训练模型)")
