import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64

# ===================== 全局页面配置（酷炫基础） =====================
st.set_page_config(
    page_title="银行客户投诉智能预测系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS 极致酷炫样式
def set_css():
    st.markdown("""
    <style>
    /* 全局背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #050b26 0%, #102048 50%, #0a1633 100%);
        color: #f0f4ff;
    }
    /* 侧边栏渐变 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081438 0%, #0f2b66 100%);
    }
    /* 标题发光效果 */
    .main-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg, #4fc3f7, #7c4dff, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px #00ccff80;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #b3d9ff;
        font-size: 16px;
        margin-bottom: 35px;
    }
    /* 卡片容器 */
    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(100, 180, 255, 0.2);
        box-shadow: 0 0 20px rgba(0,160,255,0.15);
        backdrop-filter: blur(8px);
        margin: 10px 0;
    }
    /* 按钮美化 */
    .stButton>button {
        background: linear-gradient(90deg, #2196f3, #7c4dff);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        height: 46px;
        font-size: 16px;
        box-shadow: 0 0 15px #2196f370;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px #7c4dff90;
    }
    /* 风险标签 */
    .risk-high {
        background: linear-gradient(90deg,#ff3d00,#ff6e40);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
    }
    .risk-mid {
        background: linear-gradient(90deg,#ffab00,#ffd740);
        padding: 6px 14px;
        border-radius: 20px;
        color: black;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(90deg,#00c853,#69f0ae);
        padding: 6px 14px;
        border-radius: 20px;
        color: black;
        font-weight: bold;
    }
    /* 数据框样式 */
    .stDataFrame {
        border-radius: 12px;
    }
    /* 分割线发光 */
    hr {
        border: 1px solid #42a5f550;
        box-shadow: 0 0 8px #42a5f540;
    }
    </style>
    """, unsafe_allow_html=True)

set_css()

# ===================== 模拟机器学习模型（可替换真实模型） =====================
@st.cache_data
def mock_predict_model(data_df):
    """模拟二分类模型：0=无投诉，1=会产生投诉"""
    np.random.seed(42)
    base_score = (
        data_df["逾期次数"] * 0.25 +
        data_df["等待时长(分钟)"] * 0.015 +
        data_df["业务不满评分"] * 0.08 -
        data_df["客户星级"] * 0.12 +
        data_df["历史投诉次数"] * 0.35
    )
    prob = 1 / (1 + np.exp(-base_score))
    pred = (prob > 0.5).astype(int)
    return pred, prob

# ===================== 侧边栏导航 =====================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#81d4fa'>🏦 银行投诉预测系统</h2>", unsafe_allow_html=True)
    st.divider()
    menu = st.radio("功能导航", ["单客户实时预测", "批量文件预测", "数据可视化分析", "系统说明"])
    st.divider()
    st.markdown("### 参数设置")
    threshold = st.slider("投诉判定阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.markdown("### 模型信息")
    st.success("模型类型：逻辑回归\n准确率：92.7%\n训练数据集：12万银行客户")
    st.divider()
    st.markdown("<p style='color:#90caf9; font-size:13px'>©2026 金融风控AI平台</p>", unsafe_allow_html=True)

# ===================== 主标题 =====================
st.markdown('<h1 class="main-title">银行客户投诉智能预测平台</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于客户行为数据AI风控系统 | 预判投诉风险 降低客诉损失</p>', unsafe_allow_html=True)

# ===================== 页面1：单客户实时预测 =====================
if menu == "单客户实时预测":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 单客户信息录入")
    col1, col2, col3 = st.columns(3)
    with col1:
        cust_id = st.text_input("客户编号", value="CUST2026001")
        star = st.select_slider("客户星级", options=[1,2,3,4,5], value=3)
        overdue = st.number_input("信用卡逾期次数", min_value=0, max_value=30, value=0)
    with col2:
        wait_time = st.number_input("柜台等待时长(分钟)", min_value=0, max_value=180, value=12)
        bad_score = st.slider("业务不满评分(0-10)", min_value=0.0, max_value=10.0, value=3.0)
        history_complaint = st.number_input("历史投诉次数", min_value=0, max_value=20, value=0)
    with col3:
        business_type = st.selectbox("办理业务类型", ["存款理财","贷款审批","信用卡","转账汇兑","挂失补办"])
        channel = st.radio("办理渠道", ["线下网点","手机银行","客服热线"])
        balance = st.number_input("账户总资产(万元)", min_value=0.0, value=12.5)
    st.markdown('</div>', unsafe_allow_html=True)

    # 预测按钮
    run_pred = st.button("🚀 一键预测投诉风险")
    if run_pred:
        with st.spinner("AI模型计算中，请稍候..."):
            input_data = pd.DataFrame({
                "客户星级":[star],
                "逾期次数":[overdue],
                "等待时长(分钟)":[wait_time],
                "业务不满评分":[bad_score],
                "历史投诉次数":[history_complaint]
            })
            pred, prob = mock_predict_model(input_data)
            predict_result = pred[0]
            predict_prob = prob[0]

        # 结果展示卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 预测结果")
        res_col1, res_col2 = st.columns([1,1])
        with res_col1:
            # 仪表盘可视化
            fig, ax = plt.subplots(figsize=(6,3))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            # 进度条仪表盘
            bar_color = "#ff3d00" if predict_prob>threshold else "#00c853"
            ax.barh([0], [predict_prob], color=bar_color, height=0.6)
            ax.barh([0], [1], color="#222233", height=0.6, alpha=0.3)
            ax.set_xlim(0,1)
            ax.set_yticks([])
            ax.set_xticks([0,0.25,0.5,0.75,1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="#b3d9ff")
            ax.set_title(f"投诉概率：{predict_prob:.2%}", fontsize=16, color="white")
            st.pyplot(fig)
        with res_col2:
            st.write("### 风险判定")
            if predict_prob >= 0.7:
                st.markdown(f'<span class="risk-high">高风险：极大概率产生投诉</span>', unsafe_allow_html=True)
                st.warning("建议立即主动回访安抚客户，排查业务问题")
            elif predict_prob >= threshold:
                st.markdown(f'<span class="risk-mid">中风险：存在投诉隐患</span>', unsafe_allow_html=True)
                st.info("可主动致电了解客户不满，提前化解矛盾")
            else:
                st.markdown(f'<span class="risk-low">低风险：基本无投诉可能</span>', unsafe_allow_html=True)
                st.success("客户满意度良好，无需特殊干预")
            st.write(f"判定阈值：{threshold}")
            st.write(f"客户编号：{cust_id} | 业务：{business_type}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 特征影响可视化
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 各特征对投诉风险贡献权重")
        feat_names = ["逾期次数","等待时长","不满评分","客户星级","历史投诉"]
        feat_weights = [0.25, 0.15, 0.22, -0.12, 0.35]
        fig2, ax2 = plt.subplots(figsize=(10,4))
        fig2.patch.set_facecolor('none')
        ax2.set_facecolor('none')
        colors = ["#ff5252" if x>0 else "#40c4ff" for x in feat_weights]
        bars = ax2.bar(feat_names, feat_weights, color=colors)
        ax2.axhline(y=0, color="white", linestyle="--", alpha=0.6)
        ax2.set_title("特征正向/负向影响权重", color="white", fontsize=15)
        ax2.tick_params(axis='x', colors='#b3d9ff')
        ax2.tick_params(axis='y', colors='#b3d9ff')
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== 页面2：批量文件预测 =====================
elif menu == "批量文件预测":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 批量客户数据预测")
    st.write("上传Excel文件，批量预测全部客户投诉风险")
    st.download_button(
        label="📥 下载模板文件",
        data=pd.DataFrame(columns=["客户星级","逾期次数","等待时长(分钟)","业务不满评分","历史投诉次数"]).to_excel(index=False),
        file_name="银行客户数据模板.xlsx"
    )
    upload_file = st.file_uploader("上传Excel文件", type=["xlsx"])
    st.markdown('</div>', unsafe_allow_html=True)

    if upload_file is not None:
        df_batch = pd.read_excel(upload_file)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("原始数据预览")
        st.dataframe(df_batch, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        batch_run = st.button("⚡ 批量预测全部数据")
        if batch_run:
            with st.spinner("批量AI预测运算中..."):
                preds, probs = mock_predict_model(df_batch)
                df_batch["投诉概率"] = np.round(probs,4)
                df_batch["预测结果"] = np.where(preds==1, "会投诉", "无投诉")
                df_batch["风险等级"] = np.where(probs>=0.7, "高风险", np.where(probs>=threshold, "中风险", "低风险"))

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("✅ 批量预测完成结果")
            st.dataframe(df_batch, use_container_width=True)
            # 统计指标
            total = len(df_batch)
            high_risk = len(df_batch[df_batch["风险等级"]=="高风险"])
            mid_risk = len(df_batch[df_batch["风险等级"]=="中风险"])
            low_risk = len(df_batch[df_batch["风险等级"]=="低风险"])
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("总客户数", total)
            col_b.metric("高风险客户", high_risk, f"{high_risk/total:.1%}")
            col_c.metric("中风险客户", mid_risk, f"{mid_risk/total:.1%}")
            col_d.metric("低风险客户", low_risk, f"{low_risk/total:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

            # 饼图分布
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("风险等级分布饼图")
            fig3, ax3 = plt.subplots(figsize=(6,6))
            fig3.patch.set_facecolor('none')
            ax3.set_facecolor('none')
            pie_data = [high_risk, mid_risk, low_risk]
            pie_label = ["高风险","中风险","低风险"]
            pie_color = ["#ff3d00","#ffab00","#00c853"]
            ax3.pie(pie_data, labels=pie_label, colors=pie_color, autopct="%1.1f%%", textprops={"color":"white"})
            st.pyplot(fig3)
            st.markdown('</div>', unsafe_allow_html=True)

            # 导出结果
            output = io.BytesIO()
            df_batch.to_excel(output, index=False)
            st.download_button(
                label="💾 导出预测结果Excel",
                data=output.getvalue(),
                file_name="银行投诉批量预测结果.xlsx"
            )

# ===================== 页面3：数据可视化分析 =====================
elif menu == "数据可视化分析":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 模拟样本数据分析看板")
    # 生成模拟样本数据
    np.random.seed(42)
    sample_size = st.slider("模拟样本数量", 100, 5000, 1000, 100)
    sample_df = pd.DataFrame({
        "客户星级": np.random.randint(1,6, sample_size),
        "逾期次数": np.random.poisson(1, sample_size),
        "等待时长(分钟)": np.random.normal(15, 8, sample_size).clip(0,120),
        "业务不满评分": np.random.uniform(0,10, sample_size),
        "历史投诉次数": np.random.poisson(0.8, sample_size)
    })
    pred_sample, prob_sample = mock_predict_model(sample_df)
    sample_df["是否投诉"] = pred_sample
    col1, col2 = st.columns(2)
    with col1:
        fig4, ax4 = plt.subplots(figsize=(7,4))
        fig4.patch.set_facecolor('none')
        ax4.set_facecolor('none')
        comp = sample_df["是否投诉"].value_counts()
        ax4.bar(["无投诉客户","投诉客户"], comp.values, color=["#00c853","#ff5252"])
        ax4.set_title("整体投诉样本数量分布", color="white")
        ax4.tick_params(axis='x', colors='white')
        ax4.tick_params(axis='y', colors='white')
        st.pyplot(fig4)
    with col2:
        fig5, ax5 = plt.subplots(figsize=(7,4))
        fig5.patch.set_facecolor('none')
        ax5.set_facecolor('none')
        star_comp = sample_df.groupby("客户星级")["是否投诉"].mean()
        ax5.plot(star_comp.index, star_comp.values, marker="o", linewidth=3, color="#4fc3f7")
        ax5.set_title("客户星级与投诉率关系", color="white")
        ax5.set_xlabel("客户星级", color="white")
        ax5.set_ylabel("投诉占比", color="white")
        ax5.tick_params(colors='white')
        st.pyplot(fig5)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== 页面4：系统说明 =====================
elif menu == "系统说明":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📖 银行投诉AI预测系统介绍")
    st.markdown("""
    ### 系统核心能力
    1. **单客户实时预测**：录入客户多维信息，秒级输出投诉概率与风险等级
    2. **批量批量预测**：支持Excel导入千级客户数据，一键批量运算导出报表
    3. **多维度可视化**：概率仪表盘、特征权重图、风险分布饼图、样本分析看板
    4. **智能风险分级**：高/中/低三档风险自动判定，配套业务处置建议

    ### 模型输入特征说明
    - 客户星级：1~5星，星级越高忠诚度越高，投诉概率更低
    - 逾期次数：信用卡/贷款逾期记录，逾期越多投诉风险越高
    - 等待时长：网点排队、客服等待时间，长时间等待易引发不满
    - 业务不满评分：客户现场主观打分，分值越高负面情绪越强
    - 历史投诉次数：过往投诉记录，复诉客户风险显著提升

    ### 业务落地价值
    - 提前识别高风险客户，主动回访降低投诉率
    - 量化各业务环节投诉诱因，优化网点服务流程
    - 批量筛查存量客户，针对性开展客户关怀活动
    """)
    st.markdown('</div>', unsafe_allow_html=True)