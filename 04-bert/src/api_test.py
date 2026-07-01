import requests
import json

API_URL = "http://localhost:8004"


def test_health():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试 1: 健康检查")
    print("=" * 50)
    try:
        resp = requests.get(f"{API_URL}/health")
        print(f"状态码: {resp.status_code}")
        print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        assert resp.status_code == 200
        print("✅ 健康检查通过\n")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}\n")


def test_classes():
    """测试类别列表接口"""
    print("=" * 50)
    print("测试 2: 获取类别列表")
    print("=" * 50)
    try:
        resp = requests.get(f"{API_URL}/classes")
        print(f"状态码: {resp.status_code}")
        data = resp.json()
        print(f"类别数量: {data['num_classes']}")
        print(f"类别列表:")
        for i, cls in enumerate(data['classes']):
            print(f"  {i}. {cls}")
        assert resp.status_code == 200
        print("✅ 类别接口测试通过\n")
    except Exception as e:
        print(f"❌ 类别接口测试失败: {e}\n")


def test_model_info():
    """测试模型信息接口"""
    print("=" * 50)
    print("测试 3: 获取模型信息")
    print("=" * 50)
    try:
        resp = requests.get(f"{API_URL}/models")
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"模型名称: {data['model_name']}")
            print(f"BERT模型: {data['bert_path']}")
            print(f"类别数量: {data['num_classes']}")
            print(f"完整模型存在: {data['full_model_exists']}")
            print(f"量化模型存在: {data['quantized_model_exists']}")
            if data['full_model_exists']:
                print(f"完整模型大小: {data['full_model_size_mb']} MB")
            if data['quantized_model_exists']:
                print(f"量化模型大小: {data['quantized_model_size_mb']} MB")
        else:
            print(f"响应: {resp.json()}")
        print("✅ 模型信息接口测试通过\n")
    except Exception as e:
        print(f"❌ 模型信息接口测试失败: {e}\n")


def test_predict(quantized=False):
    """测试预测接口"""
    model_type = "量化" if quantized else "完整"
    print("=" * 50)
    print(f"测试 4: 预测接口 ({model_type}模型)")
    print("=" * 50)

    test_cases = [
        {
            "text": "I have a problem with my credit card. There is an unauthorized charge of $500 on my account that I did not make.",
            "expected": "Payment cards"
        },
        {
            "text": "My mortgage payment was not applied correctly. I have been making payments on time every month but the bank says I am behind.",
            "expected": "Mortgage"
        },
        {
            "text": "I am being harassed by debt collectors. They call me multiple times a day, even at work. I have asked them to stop but they continue.",
            "expected": "Debt collection"
        },
    ]

    all_pass = True
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"输入文本: {case['text'][:80]}...")
        print(f"期望类别: {case['expected']}")

        try:
            payload = {"text": case['text'], "quantized": quantized}
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            print(f"状态码: {resp.status_code}")

            if resp.status_code == 200:
                result = resp.json()
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.4f}")
                if 'model_type' in result:
                    print(f"模型类型: {result['model_type']}")
                print("Top 3:")
                for item in result['top_classes']:
                    print(f"  {item['class_name']}: {item['probability']:.4f}")
            else:
                print(f"错误: {resp.json()}")
                all_pass = False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            all_pass = False

    print("\n" + "=" * 50)
    if all_pass:
        print(f"✅ 所有预测测试用例请求成功 ({model_type}模型)")
    else:
        print(f"❌ 部分测试用例失败 ({model_type}模型)")
    print("=" * 50)
    return all_pass


def test_missing_text():
    """测试缺少text字段的情况"""
    print("\n" + "=" * 50)
    print("测试 5: 缺少 text 字段")
    print("=" * 50)
    try:
        resp = requests.post(f"{API_URL}/predict", json={})
        print(f"状态码: {resp.status_code}")
        print(f"响应: {resp.json()}")
        assert resp.status_code == 400
        print("✅ 错误处理测试通过\n")
    except Exception as e:
        print(f"❌ 测试失败: {e}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("FinBERT API 测试脚本")
    print(f"API 地址: {API_URL}")
    print("=" * 60)
    print("请确保 Flask API 已启动 (cd src && python api.py)\n")

    test_health()
    test_classes()
    test_model_info()

    # 测试完整模型预测
    try:
        test_predict(quantized=False)
    except Exception as e:
        print(f"\n⚠️ 完整模型预测测试失败: {e}")
        print("请确认模型已训练完成 (python train.py)")

    # 测试量化模型预测
    try:
        test_predict(quantized=True)
    except Exception as e:
        print(f"\n⚠️ 量化模型预测测试失败: {e}")
        print("请确认量化已完成 (python quantize.py)")

    test_missing_text()

    print("\n🎉 测试完成！")
