import requests
import json

API_URL = "http://localhost:8002"


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


def test_predict():
    """测试预测接口"""
    print("=" * 50)
    print("测试 3: 预测接口")
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
        {
            "text": "I opened a checking account last week and now there are hidden fees that I was not told about. The bank charged me overdraft fees incorrectly.",
            "expected": "Banking services"
        },
        {
            "text": "I have a student loan and the servicer is not applying my payments correctly. They keep saying I am delinquent but I pay every month.",
            "expected": "Consumer loans"
        },
    ]

    all_pass = True
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"输入文本: {case['text'][:80]}...")
        print(f"期望类别: {case['expected']}")

        try:
            resp = requests.post(f"{API_URL}/predict", json={"text": case['text']})
            print(f"状态码: {resp.status_code}")

            if resp.status_code == 200:
                result = resp.json()
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.4f}")
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
        print("✅ 所有预测测试用例请求成功")
    else:
        print("❌ 部分测试用例失败")
    print("=" * 50)


def test_missing_text():
    """测试缺少text字段的情况"""
    print("\n" + "=" * 50)
    print("测试 4: 缺少 text 字段")
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
    print("API 测试脚本")
    print(f"API 地址: {API_URL}")
    print("请确保 Flask API 已启动 (python api.py)\n")

    test_health()
    test_classes()
    test_predict()
    test_missing_text()

    print("\n🎉 测试完成！")
