import torch
import os
from config import conf
from dataEDA_Processing import load_vocab, process_text
from bilstm_model import BiLSTM_Attention


_model = None
_vocab = None


def load_model():
    """加载模型和词表（单例模式）"""
    global _model, _vocab

    if _model is None or _vocab is None:
        # 加载词表
        if not os.path.exists(conf.vocab_path):
            raise FileNotFoundError(f"词表文件不存在: {conf.vocab_path}")
        _vocab = load_vocab(conf.vocab_path)

        # 加载模型
        if not os.path.exists(conf.save_model_path):
            raise FileNotFoundError(f"模型文件不存在: {conf.save_model_path}")

        _model = BiLSTM_Attention(len(_vocab))
        _model.load_state_dict(torch.load(conf.save_model_path, map_location=conf.device))
        _model.to(conf.device)
        _model.eval()
        print(f"模型已加载: {conf.save_model_path}")

    return _model, _vocab


def predict(text):
    """单文本预测
    Args:
        text: 投诉文本
    Returns:
        dict: {
            "predicted_class": 预测类别名,
            "confidence": 置信度,
            "predicted_id": 预测类别ID,
            "top_classes": Top3类别概率
        }
    """
    model, vocab = load_model()

    # 第一步：文本处理
    ids = process_text(text, vocab, pad_size=conf.pad_size)
    input_tensor = torch.tensor([ids], dtype=torch.long).to(conf.device)

    # 第二步：前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
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
        "top_classes": class_probs[:3]
    }


if __name__ == "__main__":
    # 测试预测
    test_texts = [
        "I have a problem with my credit card. There is an unauthorized charge on my account.",
        "My mortgage payment was not applied correctly. I have been making payments on time.",
        "I am being harassed by debt collectors. They call me multiple times a day."
    ]

    for text in test_texts:
        result = predict(text)
        print(f"\n文本: {text[:60]}...")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("Top 3:")
        for item in result['top_classes']:
            print(f"  {item['class_name']}: {item['probability']:.4f}")
