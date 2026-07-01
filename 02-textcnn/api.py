from flask import Flask, request, jsonify
from predict_fun import predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_api():
    """预测API接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "请提供 text 字段"}), 400

        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "text 字段不能为空"}), 400

        result = predict(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "model": "textcnn"})


@app.route('/classes', methods=['GET'])
def get_classes():
    """获取类别列表"""
    from config import conf
    return jsonify({
        "num_classes": conf.num_classes,
        "classes": conf.class_list
    })


if __name__ == '__main__':
    print("TextCNN Flask API 启动中...")
    app.run(host='0.0.0.0', port=8002, debug=False)
