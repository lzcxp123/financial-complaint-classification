"""
FinBERT 金融消费者投诉分类 - Streamlit前端
端口: 8004 | emoji: 🤖
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

st.set_page_config(page_title="FinBERT 金融投诉分类", page_icon="🤖", layout="centered")

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
    <h1>🤖 金融消费者投诉文本分类系统</h1>
    <div class="subtitle">FinBERT 金融预训练语言模型</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

API_URL = "http://localhost:8004/predict"

with st.expander("ℹ️ 模型简介", expanded=False):
    st.markdown("""
    <div class="model-info">
    <p><b>模型架构</b>：BERT(预训练12层) → <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token → Dropout → Linear(分类层)</p>
    <p><b>模型特点</b>：基于 ProsusAI/finbert 金融领域预训练模型，对金融文本理解更深入</p>
    <p><b>参数量</b>：约 110M</p>
    <p><b>优化技术</b>：INT8动态量化（体积减半、推理加速，精度损失极小）</p>
    <p><b>数据集</b>：CFPB Consumer Complaints，8类均衡采样，共20万条</p>
    </div>
    """, unsafe_allow_html=True)

col_a, col_b = st.columns([1, 3])
with col_a:
    use_quantized = st.checkbox("INT8 量化", value=False)
with col_b:
    st.caption("量化模型体积减半、推理加速，精度损失极小")

text_input = st.text_area(
    "请输入投诉文本（英文）:",
    height=150,
    placeholder="示例1：There are multiple incorrect items on my credit report that I have disputed...\n示例2：I tried to transfer money online but the transaction failed and funds were lost...\n示例3：My mortgage company is refusing to modify my loan despite financial hardship..."
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
                payload = {"text": text_input, "quantized": use_quantized}
                resp = requests.post(API_URL, json=payload, timeout=60)

                if resp.status_code == 200:
                    result = resp.json()

                    st.markdown("""
                    <div class="card">
                    <div class="result-title">✅ 预测结果</div>
                    """, unsafe_allow_html=True)

                    if "model_type" in result:
                        st.caption(f"当前使用的模型: {result['model_type']}")

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
                    error_msg = resp.json().get("error", "未知错误")
                    if "模型文件不存在" in error_msg:
                        st.error(f"❌ {error_msg}")
                        st.info("请先在 src 目录下运行: `python train.py`")
                    else:
                        st.error(f"❌ 预测失败: {error_msg}")

            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请确认 Flask API 已启动")
                st.info("启动命令：`cd 04-bert/src && python api.py`")
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")

st.markdown("---")
with st.expander("📊 模型详细信息"):
    try:
        resp = requests.get("http://localhost:8004/models", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            col_left, col_right = st.columns(2)
            with col_left:
                st.write(f"**模型名称**: {info['model_name']}")
                st.write(f"**BERT模型**: {info['bert_path']}")
                st.write(f"**类别数量**: {info['num_classes']}")
            with col_right:
                if info.get("full_model_exists"):
                    st.write(f"**完整模型大小**: {info.get('full_model_size_mb', 0)} MB")
                else:
                    st.write("**完整模型**: 未训练")
                if info.get("quantized_model_exists"):
                    st.write(f"**量化模型大小**: {info.get('quantized_model_size_mb', 0)} MB")
                else:
                    st.write("**量化模型**: 未生成")
        else:
            st.info("无法获取模型信息，请确认API已启动")
    except:
        st.info("请先启动Flask API服务")

st.caption("基于 FinBERT (ProsusAI/finbert) 的金融消费者投诉文本分类系统 | 数据集：CFPB Consumer Complaints | API端口: 8004")
