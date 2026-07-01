# 任务目标：教学版预测函数
# todo: 理解模型预测的完整流程

import torch
from transformers import BertTokenizer

# 假设的模型（实际使用时替换为真实模型）
# from demo-bert import DemoBertClassifier


def demo_predict(text, model, tokenizer, class_list):
    """教学版预测函数

    预测流程：
    1. 文本分词（Tokenize）
    2. 转为token IDs
    3. 模型前向传播
    4. softmax得到概率
    5. argmax得到预测类别

    Args:
        text: 输入文本
        model: BERT分类模型
        tokenizer: BERT分词器
        class_list: 类别名称列表

    Returns:
        dict: 预测结果
    """
    # 第一步：分词
    tokens = tokenizer.tokenize(text)
    print(f"分词结果: {tokens[:20]}...")

    # 第二步：转为token IDs
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    print(f"Token IDs: {input_ids.squeeze()[:30].tolist()}...")

    # 第三步：模型前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"模型logits: {logits.squeeze().tolist()}")

    # 第四步：softmax得到概率
    probs = torch.softmax(logits, dim=1).squeeze()
    print(f"各类别概率: {[f'{p:.3f}' for p in probs.tolist()]}")

    # 第五步：得到预测
    pred_id = torch.argmax(probs).item()
    pred_class = class_list[pred_id]
    confidence = probs[pred_id].item()

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "predicted_class": pred_class,
        "predicted_id": pred_id,
        "confidence": confidence,
        "all_probabilities": {cls: round(float(prob), 4) for cls, prob in zip(class_list, probs.tolist())}
    }


if __name__ == "__main__":
    # 模拟类别列表
    class_list = [
        "Credit reporting",
        "Payment cards",
        "Mortgage",
        "Debt collection",
        "Banking services",
        "Consumer loans",
        "Money transfers",
        "Payday & short-term loans",
        "Other financial services"
    ]

    # 测试文本
    test_text = "I have a problem with my credit card. There is an unauthorized charge on my account."

    print("=" * 50)
    print("教学版预测演示")
    print("=" * 50)
    print(f"输入文本: {test_text}\n")

    # 模拟预测结果（实际使用时调用真实模型）
    print("由于未加载真实模型，这里模拟预测结果...")
    print("\n完整预测流程（使用真实模型时）：")
    print("  1. tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')")
    print("  2. model = DemoBertClassifier()")
    print("  3. model.load_state_dict(torch.load('finbert.pt'))")
    print("  4. result = demo_predict(text, model, tokenizer, class_list)")

    print("\n教学演示完成！")
