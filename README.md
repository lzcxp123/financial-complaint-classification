# 基于多模型融合的金融消费者投诉文本智能分类系统

## 项目简介

本项目构建了一个面向金融消费者投诉文本的多模型智能分类系统，涵盖从传统深度学习模型（TextCNN、BiLSTM+Attention）到预训练大模型（FinBERT）的完整实验对比，并引入模型优化技术（INT8动态量化、知识蒸馏）实现推理加速和模型压缩。

**项目特色**：
- 🎯 **多模型对比**：TextCNN / BiLSTM+Attention / FinBERT / 蒸馏BiLSTM
- ⚡ **模型优化**：INT8动态量化 + FinBERT→BiLSTM知识蒸馏
- 📊 **大规模数据**：20万条均衡采样数据，8类金融投诉
- 🌐 **可视化界面**：统一CSS美化的Streamlit前端 + Flask API后端
- 🚀 **一键启动**：run.py统一启动脚本

---

## 目录结构

```
financial-complaint-classification/
├── run.py                      # 统一启动脚本（交互式菜单/命令行）
│
├── 01-data/                    # 数据探索与预处理
│   ├── hm_config_v2.py         # 配置类（8类、每类25000条）
│   ├── dataEDA_v2.py           # 数据EDA + 预处理
│   ├── train.txt               # 训练集（160,000条）
│   ├── dev.txt                 # 验证集（20,000条）
│   ├── test.txt                # 测试集（20,000条）
│   └── class.txt               # 8个类别列表
│
├── 02-textcnn/                 # TextCNN 模型
│   ├── config.py               # 配置（端口8002）
│   ├── textcnn_model.py        # 模型定义
│   ├── dataEDA_Processing.py   # 数据处理
│   ├── train.py                # 训练脚本
│   ├── test.py                 # 测试集评估
│   ├── predict_fun.py          # 预测函数
│   ├── api.py                  # Flask API (端口: 8002)
│   ├── app.py                  # Streamlit前端（美化版）
│   ├── data/vocab.pkl          # 词表
│   ├── save_models/            # 模型保存
│   └── result/                 # 测试结果
│
├── 03-bilstm/                  # BiLSTM+Attention 模型
│   ├── config.py               # 配置（端口8003）
│   ├── bilstm_model.py         # 模型定义
│   ├── dataEDA_Processing.py   # 数据处理
│   ├── train.py                # 训练脚本
│   ├── test.py                 # 测试集评估
│   ├── predict_fun.py          # 预测函数
│   ├── api.py                  # Flask API (端口: 8003)
│   ├── app.py                  # Streamlit前端（美化版）
│   ├── data/vocab.pkl          # 词表
│   ├── save_models/            # 模型保存
│   └── result/                 # 测试结果
│
├── 04-bert/                    # FinBERT 预训练模型
│   ├── src/
│   │   ├── config.py           # 配置（端口8004）
│   │   ├── bert_classifer_model.py  # 模型定义
│   │   ├── utils.py            # 数据处理
│   │   ├── train.py            # 训练脚本（层衰减LR+Multi-Sample Dropout）
│   │   ├── quantize.py         # INT8动态量化
│   │   ├── test.py             # 测试集评估
│   │   ├── predict_fun.py      # 预测函数
│   │   ├── api.py              # Flask API (端口: 8004)
│   │   ├── app.py              # Streamlit前端（量化复选框）
│   │   └── demo_*.py           # 教学演示脚本
│   ├── ProsusAI/finbert/       # FinBERT预训练权重
│   ├── save_models/            # 模型保存
│   └── logs/                   # 训练日志
│
├── 05-distill/                 # 知识蒸馏（FinBERT → BiLSTM）
│   ├── src/
│   │   ├── config.py           # 配置（端口8005）
│   │   ├── distill_model.py    # 教师模型(TeacherFinBERT) + 学生模型(StudentBiLSTM)
│   │   ├── distill_trainer.py  # 蒸馏训练器
│   │   ├── utils.py            # 双编码数据处理
│   │   ├── train_distill.py    # 蒸馏训练脚本
│   │   ├── test.py             # 测试集评估
│   │   ├── predict_fun.py      # 预测函数
│   │   ├── api.py              # Flask API (端口: 8005)
│   │   └── app.py              # Streamlit前端（蒸馏原理展示）
│   ├── save_models/            # 蒸馏模型保存
│   └── logs/                   # 训练日志
│
├── outputs/                    # 实验结果输出
│   ├── model_comparison.csv    # 模型对比表格
│   ├── model_comparison_bar.png    # 模型对比柱状图
│   ├── per_class_f1_heatmap.png    # 每类F1热力图
│   └── confusion_matrix_*.png  # 混淆矩阵
│
├── compare_experiments.py      # 多模型实验对比脚本
├── visualize.py                # 可视化图表生成
├── 答辩PPT大纲.md               # 答辩PPT大纲
└── README.md                   # 项目说明
```

---

## 环境安装

### 基础环境

```bash
# Python 3.10+
conda create -n fin-complaint python=3.10
conda activate fin-complaint

# 核心依赖
pip install pandas numpy scikit-learn tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers

# 部署相关
pip install flask streamlit requests

# 可视化
pip install matplotlib seaborn

# 数据处理
pip install beautifulsoup4 nltk
```

### 快速安装

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 第一步：数据准备

将 `complaints.csv` (CFPB数据集) 放在项目根目录下。

```bash
cd 01-data
python dataEDA_v2.py
```

**输出文件**：
- `train.txt`（160,000条） / `dev.txt`（20,000条） / `test.txt`（20,000条）
- `class.txt`（8个类别）

### 第二步：模型训练

#### TextCNN

```bash
cd ../02-textcnn
python train.py          # 训练
python test.py           # 测试评估
```

#### BiLSTM+Attention

```bash
cd ../03-bilstm
python train.py          # 训练
python test.py           # 测试评估
```

#### FinBERT（含优化训练策略）

```bash
cd ../04-bert/src
python train.py          # 训练（层衰减LR + Multi-Sample Dropout）
python quantize.py       # INT8量化
python test.py           # 测试评估
```

#### 知识蒸馏（FinBERT → BiLSTM）

```bash
cd ../../05-distill/src
python train_distill.py  # 蒸馏训练
python test.py           # 测试评估
```

### 第三步：实验对比

```bash
cd ../..
python compare_experiments.py    # 多模型对比评估
python visualize.py              # 生成可视化图表
```

结果输出到 `outputs/` 目录。

### 第四步：启动服务

#### 方式一：统一启动脚本

```bash
# 交互式菜单
python run.py

# 同时启动所有前端
python run.py --all

# 启动指定模型前端
python run.py --model textcnn    # textcnn / bilstm / finbert / distill
```

#### 方式二：分别启动

**Flask API**：

```bash
# TextCNN
cd 02-textcnn && python api.py        # 端口 8002

# BiLSTM
cd ../03-bilstm && python api.py      # 端口 8003

# FinBERT
cd ../04-bert/src && python api.py    # 端口 8004

# 蒸馏BiLSTM
cd ../../05-distill/src && python api.py  # 端口 8005
```

**Streamlit 前端**：

```bash
# 方式一：使用统一启动脚本
python run.py --all

# 方式二：分别启动
streamlit run 02-textcnn/app.py --server.port 8501
streamlit run 03-bilstm/app.py --server.port 8502
streamlit run 04-bert/src/app.py --server.port 8503
streamlit run 05-distill/src/app.py --server.port 8504
```

---

## 端口分配

| 模块 | Flask API 端口 | Streamlit 端口 | 特点 |
|------|---------------|---------------|------|
| TextCNN | 8002 | 8501 | 轻量快速 |
| BiLSTM+Attention | 8003 | 8502 | 长文本建模 |
| FinBERT | 8004 | 8503 | 金融预训练 + INT8量化复选框 |
| 蒸馏BiLSTM | 8005 | 8504 | FinBERT→BiLSTM蒸馏 |

---

## 系统架构

```
┌─────────────────┐    HTTP请求     ┌─────────────────┐    推理     ┌───────────┐
│   Streamlit     │ ──────────────> │   Flask API     │ ─────────> │   Model   │
│   前端界面       │ <────────────── │   后端服务       │ <───────── │   模型    │
│  (统一CSS美化)   │    JSON响应     │  (4个独立端口)   │    结果    │  (4种模型) │
└─────────────────┘                 └─────────────────┘            └───────────┘
```

### 前端界面特色

- **统一CSS美化**：深蓝 `#1E3A5F` + 亮蓝 `#2E86AB` 配色方案
- **卡片风格**：圆角10px + 阴影 + 白色背景
- **类别中文映射**：显示英文类别名 + 中文翻译
- **Top 3概率条**：渐变色进度条展示前3个预测类别

### API 接口说明

#### 1. 预测接口

```
POST /predict
Content-Type: application/json

请求体：
{
    "text": "I have a problem with my credit card..."
}

响应：
{
    "predicted_class": "Payment cards",
    "confidence": 0.9523,
    "predicted_id": 1,
    "top_classes": [
        {"class_name": "Payment cards", "probability": 0.9523},
        {"class_name": "Banking services", "probability": 0.0210},
        {"class_name": "Credit reporting", "probability": 0.0152}
    ],
    "model_type": "FinBERT"  // 或 "Distilled BiLSTM"
}
```

#### 2. 健康检查

```
GET /health
响应: {"status": "ok", "model": "textcnn"}
```

#### 3. 类别列表

```
GET /classes
响应: {"num_classes": 8, "classes": [...]}
```

#### 4. 模型信息（FinBERT/蒸馏模块）

```
GET /models
响应: {"model_name": "...", "bert_path": "...", "quantized_model_exists": true, ...}
```

---

## 模型对比

| 模型 | 参数量 | 推理速度 | Macro-F1（预估） | 特点 |
|------|--------|----------|-----------------|------|
| TextCNN | ~1M | ⚡⚡⚡⚡⚡ | ~0.75 | 轻量快速，多尺度卷积 |
| BiLSTM+Attention | ~5M | ⚡⚡⚡ | ~0.78 | 双向LSTM+Attention |
| FinBERT | 110M | ⚡ | **~0.87** | 金融预训练，精度最高 |
| FinBERT-INT8 | 55M | ⚡⚡ | ~0.86 | 量化压缩，体积减半 |
| 蒸馏BiLSTM | ~5M | ⚡⚡⚡⚡ | ~0.82 | 教师FinBERT→学生BiLSTM |

> 注：具体数值需运行 `compare_experiments.py` 后确定

---

## 技术栈

- **深度学习框架**：PyTorch 2.x
- **预训练模型**：HuggingFace Transformers (FinBERT)
- **数据处理**：pandas, scikit-learn, nltk, beautifulsoup4
- **后端服务**：Flask
- **前端界面**：Streamlit（统一CSS美化）
- **可视化**：matplotlib, seaborn

---

## 数据集

使用 **CFPB Consumer Complaints Dataset**，包含约170万条金融消费者投诉文本。

### 数据预处理流程

```
原始数据(170万条) → 过滤空文本 → 文本清洗 → 类别合并(22→8类) → 均衡采样(每类25000条) → 划分数据集
```

### 8个分类类别

| 类别（英文） | 类别（中文） | 合并来源 |
|-------------|-------------|---------|
| Credit reporting | 信用报告 | Credit reporting + Credit repair services |
| Payment cards | 支付卡 | Credit card + Prepaid card |
| Mortgage | 抵押贷款 | Mortgage |
| Debt collection | 债务催收 | Debt collection |
| Banking services | 银行服务 | Checking/savings account + Bank account/service |
| Consumer loans | 消费贷款 | Student loan + Vehicle loan + Consumer loan + Payday loan + Title loan |
| Money transfers | 转账汇款 | Money transfer + Virtual currency |
| Other financial services | 其他金融服务 | Other financial service |

### 数据划分（20万条总量）

- **训练集**：160,000条（80%）
- **验证集**：20,000条（10%）
- **测试集**：20,000条（10%）

---

## FinBERT 训练优化策略

为达到 **90%+ 训练准确率**，采用以下优化策略：

| 优化手段 | 配置 | 效果 |
|---------|------|------|
| **最大序列长度** | 256（原128） | 捕捉更多上下文 |
| **冻结层数** | 前2层（原6层） | 可训练参数量提升到~85% |
| **层衰减学习率** | lr = 3e-5 × 0.85^depth | 顶层快、底层稳 |
| **等效Batch Size** | 32（16×2累积） | 梯度更稳定 |
| **Multi-Sample Dropout** | 5个Dropout取平均 | 减少过拟合 |
| **训练监控** | train_acc + dev_acc + 早停(patience=5) | 防止过拟合 |

---

## 知识蒸馏架构

```
教师模型 (Teacher)          学生模型 (Student)
─────────────────          ───────────────────
  FinBERT (110M)            BiLSTM+Attention (~5M)
  冻结不训练                  需要蒸馏训练
       │                          │
       └───────软标签────────────┘
           (KL散度 + CE损失)
```

**蒸馏损失**：
```
L = α × CE(student, labels) + (1-α) × KL(student/T, teacher/T) × T²
α = 0.3（30%硬标签 + 70%软标签）
T = 4（蒸馏温度）
```

---

## 参考论文与资源

- [FinBERT: A Pretrained Language Model for Financial Communications](https://arxiv.org/abs/1908.10063)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [CFPB Consumer Complaints Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

---

## License

MIT License