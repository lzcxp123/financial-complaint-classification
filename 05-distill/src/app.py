"""
知识蒸馏 金融消费者投诉分类 - Streamlit前端
端口: 8005 | emoji: 🎓
教师模型: FinBERT | 学生模型: BiLSTM+Attention
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

st.set_page_config(page_title="知识蒸馏 金融投诉分类", page_icon="🎓", layout="centered")

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
.distill-flow {
    text-align: center;
    padding: 15px;
    font-size: 1.5rem;
    color: #2E86AB;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
    <h1>🎓 金融消费者投诉文本分类系统</h1>
    <div class="subtitle">知识蒸馏模型（FinBERT → BiLSTM+Attention）</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

API_URL = "http://localhost:8005/predict"

st.info("""
**知识蒸馏原理**：
- 🧑‍🏫 教师模型 (FinBERT): 金融预训练大模型，参数量约110M，精度高
- 🎓 学生模型 (BiLSTM+Attention): 轻量级模型，参数量约5M，速度快
- 📐 蒸馏损失：L = 0.3×CE(硬标签) + 0.7×KL(软标签/T)×T²
- 🌡️ 蒸馏温度：T=4
- ✅ 效果：学生模型在保持轻量快速的同时，接近教师模型的精度
""")

text_input = st.text_area(
    "请输入投诉文本（英文）:",
    height=150,
    placeholder="示例1：I am receiving multiple calls a day from a debt collection agency...\n示例2：My bank has placed a hold on my account without notification...\n示例3：I have errors on my credit report that have not been corrected after dispute..."
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
                payload = {"text": text_input}
                resp = requests.post(API_URL, json=payload, timeout=30)

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
                        st.info("请按顺序运行:")
                        st.code("cd 05-distill/src")
                        st.code("python train_distill.py  # 蒸馏训练")
                    else:
                        st.error(f"❌ 预测失败: {error_msg}")

            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请确认 Flask API 已启动")
                st.info("启动命令：`cd 05-distill/src && python api.py`")
            except Exception as e:
                st.error(f"❌ 发生错误: {str(e)}")

st.markdown("---")
with st.expander("📊 教师与学生模型对比"):
    try:
        resp = requests.get("http://localhost:8005/models", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            st.write(f"**模型名称**: {info['model_name']}")
            st.write(f"**教师模型**: {info['teacher_path']}")
            st.write(f"**学生模型**: {info['student_path']}")
            st.write(f"**蒸馏配置**: alpha={info['distill_alpha']}, temperature={info['distill_temperature']}")
            st.write(f"**类别数量**: {info['num_classes']}")
            st.write(f"**蒸馏模型存在**: {'是 ✅' if info['distilled_model_exists'] else '否 ❌'}")
        else:
            st.info("无法获取模型信息，请确认API已启动")
    except:
        st.info("请先启动Flask API服务")

st.caption("基于 知识蒸馏 (FinBERT→BiLSTM) 的金融消费者投诉文本分类系统 | 数据集：CFPB Consumer Complaints | API端口: 8005")
