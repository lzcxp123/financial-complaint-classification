"""
TextCNN 金融消费者投诉分类 - Streamlit前端
端口: 8002 | emoji: 📝
"""
import streamlit as st
import requests

CLASS_CN = {
    "Credit reporting": "信用报告",
    "Payment cards": "支付卡",
    "Mortgage": "抵押贷款",
    "Debt collection": "债务催收",
    "Banking services": "银行服务",
    "Consumer loans": "消费贷款",
    "Money transfers": "转账汇款",
    "Other financial services": "其他金融服务"
}

st.set_page_config(page_title="TextCNN 金融投诉分类", page_icon="📝", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #F0F4F8;
}
h1 {
    color: #1E3A5F !important;
    text-align: center;
    font-size: 1.8rem !important;
    font-weight: bold !important;
    padding-top: 0.5rem;
}
h2, h3 {
    color: #2E86AB !important;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1rem;
    margin-top: -10px;
    margin-bottom: 20px;
}
.card {
    background: #fff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.stTextArea textarea {
    border-radius: 10px !important;
    border: 2px solid #2E86AB !important;
    font-size: 0.95rem !important;
}
div.stButton > button:first-child {
    border-radius: 10px !important;
    height: 2.8rem;
    font-weight: 600;
}
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #2E86AB, #1E3A5F) !important;
    border-radius: 6px !important;
}
div[data-testid="stProgress"] > div > div {
    border-radius: 6px !important;
}
.result-title {
    font-size: 1.1rem;
    font-weight: bold;
    color: #1E3A5F;
    margin-bottom: 10px;
}
.model-info p {
    margin: 4px 0;
    color: #555;
}
.top3-item {
    margin: 8px 0;
}
.st-emotion-cache-1y4p8pa {
    max-width: 900px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
    <h1>📝 金融消费者投诉文本分类系统</h1>
    <div class="subtitle">TextCNN 卷积神经网络模型</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

API_URL = "http://localhost:8002/predict"

with st.expander("ℹ️ 模型简介", expanded=False):
    st.markdown("""
    <div class="model-info">
    <p><b>模型架构</b>：Embedding → 多尺度 Conv1d(3,4,5) → ReLU → MaxPool → Dropout → FC</p>
    <p><b>模型特点</b>：轻量快速，多尺度卷积捕捉不同粒度的文本特征，适合短文本分类</p>
    <p><b>参数量</b>：约 1M</p>
    <p><b>数据集</b>：CFPB Consumer Complaints，8类均衡采样，共20万条</p>
    </div>
    """, unsafe_allow_html=True)

text_input = st.text_area(
    "请输入投诉文本（英文）:",
    height=150,
    placeholder="示例1：I have a problem with my credit card. There is an unauthorized charge of $500...\n示例2：My mortgage payment was not applied correctly to my account...\n示例3：I am being harassed by debt collectors calling my work every day..."
)

col1, col2 = st.columns([1, 5])
with col1:
    predict_btn = st.button("🔍 开始预测", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ 清空", use_container_width=True)

if clear_btn:
    st.rerun()

if predict_btn:
    if not text_input or not text_input.strip():
        st.warning("⚠️ 请输入投诉文本后再预测")
    else:
        with st.spinner("预测中..."):
            try:
                resp = requests.post(API_URL, json={"text": text_input}, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()

                    st.markdown("""
                    <div class="card">
                    <div class="result-title">✅ 预测结果</div>
                    """, unsafe_allow_html=True)

                    class_cn = CLASS_CN.get(result["predicted_class"], "")
                    st.success(f"**预测类别：{result['predicted_class']}** ({class_cn})")
                    st.info(f"🎯 置信度：**{result['confidence']:.4f}** ({result['confidence']*100:.2f}%)")

                    st.markdown("### 🏆 Top 3 类别概率")
                    for i, item in enumerate(result["top_classes"], 1):
                        cn = CLASS_CN.get(item["class_name"], "")
                        st.markdown(f"<div class='top3-item'>{i}. **{item['class_name']}** ({cn}) — {item['probability']:.4f}</div>", unsafe_allow_html=True)
                        st.progress(item["probability"])

                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.error(f"❌ 预测失败: {resp.json().get('error', '未知错误')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请确认 Flask API 已启动")
                st.info("启动命令：`cd 02-textcnn && python api.py`")
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")

st.markdown("---")
st.caption("基于 TextCNN 的金融消费者投诉文本分类系统 | 数据集：CFPB Consumer Complaints | API端口: 8002")
