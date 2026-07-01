from flask import Flask, request, jsonify
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from predict_fun import predict
from config import conf

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

        # 检查是否使用量化模型
        quantized = data.get('quantized', False)

        result = predict(text, quantized=quantized)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": f"模型文件不存在，请先训练模型: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "model": "finbert"})


@app.route('/classes', methods=['GET'])
def get_classes():
    """获取类别列表"""
    return jsonify({
        "num_classes": conf.num_classes,
        "classes": conf.class_list
    })


@app.route('/models', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    import os
    model_path = conf.model_save_path
    quantized_path = conf.quantized_model_path

    info = {
        "model_name": conf.model_name,
        "bert_path": conf.bert_path,
        "num_classes": conf.num_classes,
        "full_model_exists": os.path.exists(model_path),
        "quantized_model_exists": os.path.exists(quantized_path),
        "full_model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2) if os.path.exists(model_path) else 0,
        "quantized_model_size_mb": round(os.path.getsize(quantized_path) / (1024 * 1024), 2) if os.path.exists(quantized_path) else 0
    }
    return jsonify(info)


if __name__ == '__main__':
    print("FinBERT Flask API 启动中...")
    print(f"API端口: {conf.api_port}")
    print(f"FinBERT模型: {conf.bert_path}")
    app.run(host='0.0.0.0', port=conf.api_port, debug=False)
