"""
银行消费者投诉智能分类系统 - Streamlit 前端
风格：简洁大气、银行企业级深蓝金色主题
兼容项目现有 Flask API 接口
"""

import streamlit as st
import requests
import os

# ================================================================
# 页面基础配置
# ================================================================

st.set_page_config(
    page_title="银行消费者投诉智能分类系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# 全局样式 - 银行深蓝金色主题
# ================================================================

# 颜色方案
NAVY = "#0A1628"
ROYAL_BLUE = "#1B3A5C"
STEEL_BLUE = "#2E5D8A"
GOLD = "#C9A84C"
LIGHT_GOLD = "#E8D5A0"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F7FA"
MEDIUM_GRAY = "#B0BEC5"
DARK_TEXT = "#1A2332"
SUCCESS_GREEN = "#1B7A3D"
ERROR_RED = "#C62828"
WARN_ORANGE = "#E65100"

# 自定义 CSS
custom_css = """
<style>
    /* 全局字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif;
    }

    /* 隐藏 Streamlit 默认顶部菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 大标题区域 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0A1628;
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #5A6B7F;
        font-weight: 400;
        letter-spacing: 1px;
    }

    /* 金色分割线 */
    .gold-divider {
        height: 3px;
        background: linear-gradient(to right, transparent, #C9A84C, transparent);
        border: none;
        margin: 1rem 0 1.5rem 0;
    }

    /* 模型选择卡片 */
    .model-card {
        background: linear-gradient(135deg, #0A1628, #1B3A5C);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(10, 22, 40, 0.3);
        transition: all 0.3s ease;
    }

    .model-card:hover {
        box-shadow: 0 6px 20px rgba(10, 22, 40, 0.5);
        transform: translateY(-2px);
    }

    .model-card h3 {
        color: #C9A84C;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }

    .model-card p {
        color: #B0BEC5;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.5;
    }

    /* 结果卡片 */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.8rem 2rem;
        box-shadow: 0 2px 12px rgba(10, 22, 40, 0.08);
        border: 1px solid #E8EAF0;
    }

    .result-card h3 {
        color: #0A1628;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }

    /* 置信度仪表盘 */
    .confidence-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #C9A84C;
        text-align: center;
        line-height: 1;
    }

    .confidence-label {
        font-size: 0.85rem;
        color: #5A6B7F;
        text-align: center;
        margin-top: 0.3rem;
    }

    /* 概率条样式 */
    .prob-bar-container {
        margin-bottom: 0.6rem;
    }

    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }

    .prob-bar-label-name {
        color: #1A2332;
        font-weight: 500;
    }

    .prob-bar-label-value {
        color: #5A6B7F;
        font-weight: 400;
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #1B3A5C 100%);
    }

    [data-testid="stSidebar"] * {
        color: #E8EAF0 !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.92rem;
        padding: 0.5rem 0;
    }

    /* 隐藏默认 Radio 的边框圆点，用卡片替代 */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Banner 图片 */
    .banner-image {
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(10, 22, 40, 0.15);
        margin-bottom: 1rem;
    }

    /* 信息提示框 */
    .info-box {
        background: #EBF2FA;
        border-left: 4px solid #C9A84C;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #1A2332;
        line-height: 1.6;
    }

    /* 类别标签 */
    .category-tag {
        display: inline-block;
        background: linear-gradient(135deg, #0A1628, #1B3A5C);
        color: #C9A84C;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* 底部版权 */
    .footer-text {
        text-align: center;
        font-size: 0.78rem;
        color: #8899AA;
        padding: 1.5rem 0 1rem 0;
    }

    /* 示例选择卡片 */
    .example-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0A1628;
        margin-bottom: 0.8rem;
    }

    .example-card {
        background: white;
        border: 1px solid #E8EAF0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        border-left: 3px solid #C9A84C;
    }

    .example-card:hover {
        box-shadow: 0 2px 8px rgba(10, 22, 40, 0.1);
    }

    .example-card-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1B3A5C;
        margin-bottom: 0.3rem;
    }

    .example-card-preview {
        font-size: 0.8rem;
        color: #5A6B7F;
        line-height: 1.5;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .example-category-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0A1628, #1B3A5C);
        color: #C9A84C;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ================================================================
# Banner 图片
# ================================================================

def _img_to_base64(img_path):
    """将图片转为 base64 编码"""
    import base64
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
banner_path = os.path.join(current_dir, "bank_banner.jpg")

if os.path.exists(banner_path):
    st.markdown(
        f'<img src="data:image/jpeg;base64,{_img_to_base64(banner_path)}" class="banner-image">',
        unsafe_allow_html=True
    )
else:
    # 备用：纯色 Banner
    st.markdown(
        '<div style="background: linear-gradient(135deg, #0A1628, #1B3A5C, #2E5D8A); '
        'height: 120px; border-radius: 12px; margin-bottom: 1rem; '
        'display: flex; align-items: center; justify-content: center;">'
        '<span style="color: #C9A84C; font-size: 1.8rem; font-weight: 700; letter-spacing: 3px;">'
        '🏦 银行消费者投诉智能分类系统</span></div>',
        unsafe_allow_html=True
    )


# ================================================================
# 标题区域
# ================================================================

st.markdown(
    '<div class="main-title">银行消费者投诉智能分类系统</div>'
    '<div class="sub-title">Financial Consumer Complaint Intelligent Classification System</div>'
    '<hr class="gold-divider">',
    unsafe_allow_html=True
)

# ================================================================
# 侧边栏 - 模型选择
# ================================================================

with st.sidebar:
    st.markdown(
        '<div style="margin-bottom: 1.5rem;">'
        '<div style="font-size: 1.1rem; font-weight: 600; color: #C9A84C; '
        'margin-bottom: 0.8rem; letter-spacing: 1px;">📊 模型选择</div>'
        '<div style="font-size: 0.82rem; color: #8899AA; line-height: 1.6;">'
        '选择用于分类的模型，不同模型在精度和速度之间各有取舍</div></div>',
        unsafe_allow_html=True
    )

    model_options = {
        "TextCNN": {
            "port": 8002,
            "desc": "轻量快速，适合实时高并发场景",
            "params": "~1M",
            "icon": "⚡"
        },
        "BiLSTM+Attention": {
            "port": 8003,
            "desc": "长文本建模能力优秀",
            "params": "~5M",
            "icon": "🔗"
        },
        "FinBERT": {
            "port": 8004,
            "desc": "金融预训练模型，精度最高",
            "params": "110M",
            "icon": "🤖"
        },
        "DistilBERT+KD": {
            "port": 8005,
            "desc": "知识蒸馏，精度与速度平衡",
            "params": "66M",
            "icon": "🎓"
        }
    }

    selected_model = st.radio(
        "",
        list(model_options.keys()),
        index=2,  # 默认 FinBERT
        label_visibility="collapsed"
    )

    # 显示选中模型信息
    model_info = model_options[selected_model]
    st.markdown(
        f'<div class="model-card">'
        f'<h3>{model_info["icon"]} {selected_model}</h3>'
        f'<p>{model_info["desc"]}</p>'
        f'<p style="margin-top: 0.5rem; color: #C9A84C;">参数量: {model_info["params"]}  |  端口: {model_info["port"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    # FinBERT 量化选项
    if selected_model == "FinBERT":
        use_quantized = st.checkbox("使用 INT8 量化模型（加速推理）", value=False)
    else:
        use_quantized = False

    st.markdown("---")

    # 支持类别列表
    st.markdown(
        '<div style="font-size: 0.88rem; color: #8899AA; margin-top: 0.5rem;">'
        '<div style="color: #C9A84C; font-weight: 600; margin-bottom: 0.5rem;">📋 支持类别</div>'
        '<div style="line-height: 2;">'
        '1. Credit reporting<br>'
        '2. Payment cards<br>'
        '3. Mortgage<br>'
        '4. Debt collection<br>'
        '5. Banking services<br>'
        '6. Consumer loans<br>'
        '7. Money transfers<br>'
        '8. Other financial services'
        '</div></div>',
        unsafe_allow_html=True
    )


# ================================================================
# 示例数据 - 来自真实训练集
# ================================================================

EXAMPLES = [
    {
        "title": "信用卡逾期误报",
        "category": "Credit reporting",
        "text": (
            "The existence of a derogatory rating on my account is causing me significant concern. "
            "I am deeply worried about its potential impact on my credit. It has already resulted in "
            "the denial of a recent loan application and an increase in the interest rates on my "
            "existing credit accounts. I want to emphasize the severe financial and emotional distress "
            "that this negative rating has caused me and will continue to cause until it is resolved."
        )
    },
    {
        "title": "信用卡奖励未兑现",
        "category": "Payment cards",
        "text": (
            "I applied for a credit card with Chase called Chase Freedom Unlimited. I specifically "
            "chose to apply with them because of the promotion they had to get $200.00 cash back "
            "after spending $500.00 in the first 3 months of having the credit card. I was able to "
            "meet that requirement probably within the first month of the card opening. I noticed "
            "however that they did not apply the promotion to my account even after the fact. They "
            "told me the promotion was never in my account to begin with. I feel like it was a click "
            "bait and I was mislead with multiple different information."
        )
    },
    {
        "title": "贷款止赎与沟通问题",
        "category": "Mortgage",
        "text": (
            "We were approved for forbearance due to our hardship from Covid-19 as I haven't been "
            "working but my husband has. We were supposed to start paying payments again but they "
            "offered us an extension through forbearance. They failed to send us a notice that we were "
            "denied an extension and our house went into foreclosure. Citizens filed legal action "
            "against us. We need help fighting to save our home. Citizens is in the wrong. They "
            "have admitted that and now lied about modifying our loan and not communicated with us."
        )
    },
    {
        "title": "虚假债务催收",
        "category": "Debt collection",
        "text": (
            "I received a collection notice in the amount of $75.00 for a violation of toll crossing. "
            "It stated I crossed the bridge without paying. We have never owned this vehicle or "
            "crossed the bridge on this date. I called the company and she stated she would take "
            "care of the account and delete the charges. I have also sent a followup letter to the "
            "company in question. As a senior citizen and retired I would like to file a formal "
            "complaint against this company who are clearly trying to bilk people out of their "
            "hard earned money."
        )
    },
    {
        "title": "银行账户冻结",
        "category": "Banking services",
        "text": (
            "I attempted to utilize my debit card via my smart watch to make a purchase at a store "
            "for a total of $170.00. Both times were declined. My account was flagged for unusual "
            "activity. I received texts and emails stating that my account had been restricted and "
            "that I could not withdraw or use my debit card. I contacted customer support but they "
            "refused to lift the restrictions even though I provided all information they requested. "
            "Considering I have around $7000.00 in the account and am currently on vacation, I need "
            "this resolved quite expeditiously."
        )
    },
    {
        "title": "学生贷款高额还款",
        "category": "Consumer loans",
        "text": (
            "I am part of the many Americans that grew up in the middle class. Because they were "
            "middle class, I didn't qualify for federal loans, therefore private loans it was. I "
            "began my career immediately after graduating from college, but unfortunately a year "
            "later the position was eliminated due to budget cuts. I called Navient to discuss an "
            "income based repayment option. They told me income based repayment was not available "
            "to me due to having private loans and my only option was forbearance. Between my "
            "husband and I, we pay $2000.00 a month in student loans."
        )
    },
    {
        "title": "支付平台资金被扣",
        "category": "Money transfers",
        "text": (
            "I am filing a complaint against Cash App (Block, Inc.) due to inadequate customer "
            "service and unfair practices, which violate the Consumer Financial Protection Act. "
            "Specifically, Cash App failed to take timely and effective measures to prevent and "
            "address fraud on their platform, leaving my account vulnerable and unprotected. "
            "Furthermore, their dispute resolution process was unfair and deceptive, as they did "
            "not comply with error resolution requirements under the Electronic Fund Transfer Act. "
            "These actions have caused significant inconvenience and financial loss."
        )
    },
    {
        "title": "汇款服务冻结资金",
        "category": "Money transfers",
        "text": (
            "Attempted to send money using Moneygram. They accepted the transfer and placed a "
            "debit of $1500.00 against my bank account. Received email stating transaction "
            "complete. 2 hours later received email stating that my account was fraud and banned "
            "me from using their service. The transaction was legitimate and an attempt to help "
            "my family overseas. They are now holding my money for up to 30 days with no "
            "explanation. I have attempted to contact them but they will not return money, the "
            "fee or explain why they deemed me a fraud."
        )
    },
    {
        "title": "信用报告欺诈条目",
        "category": "Credit reporting",
        "text": (
            "I am urging you to take action on these inaccurate and fraudulent accounts on my "
            "credit report. It is clear that I have no association with them and they are causing "
            "significant harm to my credit score. On my credit report, you have shown incorrect "
            "accounts that should not be there at all. This is not only unjust to me, but it's also "
            "concerning, because I've never done or made any of the things you accuse me of. "
            "Please remove them promptly and ensure that they do not reappear."
        )
    },
    {
        "title": "抵押贷款利率问题",
        "category": "Mortgage",
        "text": (
            "My payment with Amerisave increased to $1300.00, an increase of around $400.00. "
            "Due to increase, it has been difficult to make payment, no explanation provided for "
            "increase. Further, a payment was made in error on my account, the amount of $2600.00. "
            "A payment delay in amount of $2000.00 caused payment to return. I request Amerisave "
            "allow me time to catch up on past due payments including this month and next month."
        )
    },
    {
        "title": "身份盗窃信用报告",
        "category": "Credit reporting",
        "text": (
            "I am a victim of identity theft. The information listed below, which appears on my "
            "credit report, is the result of that. Multiple unauthorized accounts and fraudulent "
            "charges have appeared on my report due to identity theft. I did not open, authorize, "
            "or make any of these fraudulent accounts. They are solely the result of identity "
            "theft and fraud. Pursuant to my rights under the FCRA, I demand the immediate "
            "removal of all these fraudulent items from my credit report."
        )
    },
    {
        "title": "高额贷款利息争议",
        "category": "Other financial services",
        "text": (
            "I received several emails soliciting a Personal Loan via this Internet Lender. The "
            "female representative ushered me through the Loan signing process, claiming that she "
            "needed to get this done quickly. I complained that the monthly payment was very high "
            "at $250.00 and she responded saying this is an expensive form of borrowing. I was "
            "never told that this would be a bi-weekly payment and not monthly. This is totally "
            "unacceptable. I am paying back 68% of the $750.00 Loan in less than every 30 days."
        )
    },
]

# ================================================================
# 初始化 session_state 与按钮逻辑（必须放在 widget 之前）
# ================================================================

if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

# 示例选择回调
def _use_example(idx):
    st.session_state.complaint_text = EXAMPLES[idx]["text"]

# 清空回调
def _clear_text():
    st.session_state.complaint_text = ""

# ================================================================
# 主区域 - 输入与预测
# ================================================================

# 两列布局：输入区 + 信息区
col_input, col_info = st.columns([2, 1])

with col_input:
    st.markdown(
        '<h3 style="color: #0A1628; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.8rem;">'
        '📝 投诉文本输入</h3>',
        unsafe_allow_html=True
    )

    text_input = st.text_area(
        "请输入投诉文本（英文）:",
        height=180,
        placeholder="例如：I have a problem with my credit card. There is an unauthorized charge of $500 on my account that I did not make. I have contacted the bank multiple times but they have not resolved the issue...",
        label_visibility="collapsed",
        value=st.session_state.complaint_text
    )

    # 双列按钮
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        predict_btn = st.button("🔍 开始分类", type="primary", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("✕ 清空", use_container_width=True, on_click=_clear_text)

with col_info:
    st.markdown(
        '<h3 style="color: #0A1628; font-size: 1.05rem; font-weight: 600; margin-bottom: 0.8rem;">'
        'ℹ️ 系统说明</h3>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info-box">'
        '本系统基于深度学习模型，对金融消费者投诉文本进行自动分类。<br><br>'
        '<b>使用步骤：</b><br>'
        '1. 在左侧选择分类模型<br>'
        '2. 输入投诉文本或选择下方示例<br>'
        '3. 点击「开始分类」<br><br>'
        '<b>提示：</b>请确保对应的 Flask API 服务已启动'
        '</div>',
        unsafe_allow_html=True
    )

# ================================================================
# 示例选择区域
# ================================================================

with st.expander("📌 点击选择示例投诉文本（来自真实训练数据）", expanded=False):
    # 用 Streamlit 按钮按 4 列展示
    for row_start in range(0, len(EXAMPLES), 4):
        row_examples = EXAMPLES[row_start:row_start + 4]
        cols = st.columns(4)
        for col_i, example in enumerate(row_examples):
            with cols[col_i]:
                preview = example["text"][:70] + "..." if len(example["text"]) > 70 else example["text"]
                st.markdown(
                    f'<div class="example-card">'
                    f'<div class="example-card-title">'
                    f'<span class="example-category-badge">{example["category"]}</span>'
                    f'{example["title"]}'
                    f'</div>'
                    f'<div class="example-card-preview">{preview}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.button(
                    "使用此示例",
                    key=f"ex_{row_start + col_i}",
                    use_container_width=True,
                    on_click=_use_example,
                    args=(row_start + col_i,)
                )

# ================================================================
# 预测逻辑
# ================================================================

if predict_btn:
    if not text_input or not text_input.strip():
        st.markdown(
            '<div style="background: #FFF3E0; border-left: 4px solid #E65100; '
            'border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.9rem; color: #1A2332;">'
            '⚠️ 请输入投诉文本后再进行分类</div>',
            unsafe_allow_html=True
        )
    else:
        api_url = f"http://localhost:{model_info['port']}/predict"

        with st.spinner(f"正在使用 {selected_model} 模型进行分类..."):
            try:
                payload = {"text": text_input}
                if selected_model == "FinBERT":
                    payload["quantized"] = use_quantized

                resp = requests.post(api_url, json=payload, timeout=60)

                if resp.status_code == 200:
                    result = resp.json()

                    # ========== 预测结果展示 ==========
                    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

                    # 顶部：类别 + 置信度
                    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])

                    with top_col1:
                        st.markdown(
                            '<div class="result-card">'
                            '<h3>🎯 分类结果</h3>'
                            f'<div class="category-tag">'
                            f'{result["predicted_class"]}'
                            f'</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    with top_col2:
                        conf_value = result["confidence"]
                        if conf_value >= 0.9:
                            conf_color = "#1B7A3D"
                        elif conf_value >= 0.7:
                            conf_color = "#C9A84C"
                        else:
                            conf_color = "#E65100"

                        st.markdown(
                            '<div class="result-card" style="text-align: center;">'
                            '<h3>📊 置信度</h3>'
                            f'<div class="confidence-value" style="color: {conf_color};">{conf_value:.2%}</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    with top_col3:
                        model_type = result.get("model_type", selected_model)
                        st.markdown(
                            '<div class="result-card" style="text-align: center;">'
                            '<h3>🧠 模型</h3>'
                            f'<div style="color: #1B3A5C; font-size: 1rem; font-weight: 600; margin-top: 0.5rem;">'
                            f'{selected_model}'
                            f'</div>'
                            f'<div style="color: #8899AA; font-size: 0.78rem; margin-top: 0.2rem;">'
                            f'{model_type}'
                            f'</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    # ========== 概率分布 ==========
                    st.markdown("<br>", unsafe_allow_html=True)

                    if "top_classes" in result:
                        st.markdown(
                            '<div class="result-card">'
                            '<h3>📈 各类别概率分布</h3>',
                            unsafe_allow_html=True
                        )

                        # 使用 Streamlit 原生 progress bar
                        for i, item in enumerate(result["top_classes"]):
                            prob = item["probability"]
                            # 根据概率设置颜色
                            if prob >= 0.7:
                                bar_color = "#1B7A3D"
                            elif prob >= 0.3:
                                bar_color = "#C9A84C"
                            else:
                                bar_color = "#2E5D8A"

                            st.markdown(
                                f'<div class="prob-bar-container">'
                                f'<div class="prob-bar-label">'
                                f'<span class="prob-bar-label-name">{i+1}. {item["class_name"]}</span>'
                                f'<span class="prob-bar-label-value">{prob:.2%}</span>'
                                f'</div></div>',
                                unsafe_allow_html=True
                            )
                            st.progress(prob)

                        st.markdown('</div>', unsafe_allow_html=True)

                else:
                    error_msg = resp.json().get("error", "未知错误")
                    st.markdown(
                        f'<div style="background: #FFEBEE; border-left: 4px solid #C62828; '
                        f'border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.9rem; color: #1A2332;">'
                        f'❌ 分类失败: {error_msg}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    if "模型文件不存在" in error_msg:
                        st.markdown(
                            '<div class="info-box" style="margin-top: 0.5rem;">'
                            '请先训练对应模型后再进行预测。'
                            '</div>',
                            unsafe_allow_html=True
                        )

            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<div style="background: #FFEBEE; border-left: 4px solid #C62828; '
                    'border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.9rem; color: #1A2332;">'
                    f'❌ 无法连接到 API 服务（端口 {model_info["port"]}）<br>'
                    f'请确认 Flask API 已启动: <code style="background: #E8EAF0; padding: 2px 6px; '
                    f'border-radius: 4px; font-size: 0.82rem;">python api.py</code>'
                    '</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.markdown(
                    f'<div style="background: #FFEBEE; border-left: 4px solid #C62828; '
                    f'border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.9rem; color: #1A2332;">'
                    f'❌ 发生错误: {str(e)}'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ================================================================
# 底部
# ================================================================

st.markdown(
    '<hr class="gold-divider">'
    '<div class="footer-text">'
    '基于多模型融合的金融消费者投诉文本智能分类系统 | '
    'TextCNN / BiLSTM+Attention / FinBERT / DistilBERT+KD'
    '</div>',
    unsafe_allow_html=True
)
