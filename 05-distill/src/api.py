"""
蒸馏模型 Flask API (FinBERT -> BiLSTM+Attention)
端口: 8005
"""
from flask import Flask, request, jsonify
import sys
import os

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

        result = predict(text)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": f"模型文件不存在，请先训练蒸馏模型: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"预测失败: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "model": "distilled_bilstm"})


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
    info = {
        "model_name": conf.model_name,
        "student_path": conf.student_path,
        "teacher_path": conf.teacher_path,
        "distill_alpha": conf.alpha,
        "distill_temperature": conf.temperature,
        "num_classes": conf.num_classes,
        "distilled_model_exists": os.path.exists(conf.model_save_path)
    }
    return jsonify(info)


if __name__ == '__main__':
    print("蒸馏BiLSTM Flask API 启动中...")
    print(f"API端口: {conf.api_port}")
    print(f"学生模型: BiLSTM+Attention")
    print(f"教师模型: FinBERT")
    app.run(host='0.0.0.0', port=conf.api_port, debug=False)
