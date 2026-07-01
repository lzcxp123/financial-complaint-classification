import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from hm_config_v2 import conf


# ---------------------------------------------------------------
# 以下函数与 v1 完全相同，此处重复仅为保持文件独立可运行
# ---------------------------------------------------------------

def load_data(csv_path):
    """第一步：加载CSV数据，只保留投诉叙述非空的记录"""
    print("=" * 60)
    print("第一步：加载原始数据")
    print("=" * 60)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"原始数据总量: {len(df)} 条")

    df = df.dropna(subset=['Consumer complaint narrative'])
    df = df[df['Consumer complaint narrative'].str.strip() != '']
    print(f"去除空文本后数据量: {len(df)} 条")

    df = df.rename(columns={
        'Consumer complaint narrative': 'text',
        'Product': 'product_raw'
    })

    return df


def eda_analysis(df, title="数据EDA分析"):
    """第二步：数据探索分析（EDA）"""
    print("\n" + "=" * 60)
    print(f"第二步：{title}")
    print("=" * 60)

    if 'product_raw' in df.columns:
        label_col = 'product_raw'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        label_col = None

    if label_col:
        label_counts = Counter(df[label_col])
        print(f"\n标签种类数: {len(label_counts)}")
        print("\n各类别数量:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count} 条 ({count / len(df) * 100:.2f}%)")

    df['text_len_char'] = df['text'].apply(lambda x: len(str(x)))
    print(f"\n文本长度统计（字符数）:")
    print(f"  平均长度: {df['text_len_char'].mean():.1f}")
    print(f"  最大长度: {df['text_len_char'].max()}")
    print(f"  最小长度: {df['text_len_char'].min()}")
    print(f"  中位数长度: {df['text_len_char'].median():.1f}")

    df['text_len_word'] = df['text'].apply(lambda x: len(str(x).split()))
    print(f"\n文本长度统计（词数）:")
    print(f"  平均词数: {df['text_len_word'].mean():.1f}")
    print(f"  最大词数: {df['text_len_word'].max()}")
    print(f"  最小词数: {df['text_len_word'].min()}")
    print(f"  中位数词数: {df['text_len_word'].median():.1f}")

    return df


def clean_text(text):
    """第三步：文本清洗"""
    text = str(text)
    text = re.sub(r'\bX{4,}\b', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def merge_categories(df):
    """第四步：类别合并（v2: 8类）"""
    print("\n" + "=" * 60)
    print("第四步：类别合并 (v2: Payday loans -> Consumer loans)")
    print("=" * 60)

    df['label'] = df['product_raw'].map(conf.category_mapping)

    unmatched = df[df['label'].isna()]['product_raw'].unique()
    if len(unmatched) > 0:
        print(f"\n警告：以下 {len(unmatched)} 个类别未匹配，将归入 'Other financial services':")
        for cat in unmatched:
            print(f"  - {cat}")
        df['label'] = df['label'].fillna('Other financial services')

    label_counts = Counter(df['label'])
    print(f"\n合并后类别数: {len(label_counts)}")
    print("\n合并后各类别数量:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} 条 ({count / len(df) * 100:.2f}%)")

    return df


# ---------------------------------------------------------------
# v2 核心：均衡采样 —— 下采样多数类 + 过采样少数类
# ---------------------------------------------------------------

def balanced_sampling(df, n_per_class=25000, random_seed=42):
    """
    方案 C：下采样多数类 + 过采样少数类 到统一数量

    逻辑:
      - 类别样本数 >= n_per_class → 随机下采样（不放回）
      - 类别样本数  <  n_per_class → 随机过采样（放回）
      - 最终每类 n_per_class 条
    """
    print("\n" + "=" * 60)
    print("第五步：均衡采样 (Balanced Sampling)")
    print("=" * 60)

    label_counts = Counter(df['label'])
    print(f"\n采样前数据量: {len(df)} 条")
    print(f"目标每类: {n_per_class} 条")
    print(f"预期总数据: {len(conf.major_categories)} × {n_per_class} = {len(conf.major_categories) * n_per_class} 条\n")

    print(f"{'类别':<30} {'原始数量':>8} {'操作':>12} {'采样后':>8}")
    print("-" * 62)

    np.random.seed(random_seed)
    balanced_dfs = []

    for label in conf.major_categories:
        label_df = df[df['label'] == label]
        orig_count = len(label_df)

        if orig_count == 0:
            print(f"{label:<30} {orig_count:>8} {'无数据':>12} {0:>8}")
            continue

        if orig_count >= n_per_class:
            # ===== 下采样（不放回）=====
            sampled = label_df.sample(n=n_per_class, random_state=random_seed)
            op = "下采样"
        else:
            # ===== 过采样（放回，控制倍数）=====
            # 当原始样本极少时（如 198 条合并后不再是极端情况），
            # 过采样倍数合理（< 15x），不会导致严重过拟合
            sampled = label_df.sample(
                n=n_per_class,
                random_state=random_seed,
                replace=True  # 放回采样，允许重复
            )
            op = "过采样"

        balanced_dfs.append(sampled)
        print(f"{label:<30} {orig_count:>8} {op:>12} {n_per_class:>8}")

    # 合并所有类别
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    sampled_counts = Counter(df_balanced['label'])
    print(f"\n采样后总数据量: {len(df_balanced)} 条")
    print("\n采样后各类别数量:")
    for label in conf.major_categories:
        count = sampled_counts.get(label, 0)
        pct = count / len(df_balanced) * 100
        print(f"  {label}: {count} 条 ({pct:.1f}%)")

    # 验证均匀性
    counts_list = [sampled_counts.get(l, 0) for l in conf.major_categories]
    if len(set(counts_list)) == 1:
        print(f"\n✅ 类别完全均衡！每类 {counts_list[0]} 条")
    else:
        print(f"\n⚠️ 类别不完全均衡: {counts_list}")

    return df_balanced


def split_and_save(df, conf_obj):
    """第六步：划分数据集并保存"""
    print("\n" + "=" * 60)
    print("第六步：划分数据集并保存")
    print("=" * 60)

    label_to_id = {label: idx for idx, label in enumerate(conf_obj.major_categories)}
    df['label_id'] = df['label'].map(label_to_id)

    # 分层划分
    train_df, temp_df = train_test_split(
        df,
        test_size=(conf_obj.dev_ratio + conf_obj.test_ratio),
        random_state=conf_obj.random_seed,
        stratify=df['label_id']
    )

    dev_ratio_in_temp = conf_obj.dev_ratio / (conf_obj.dev_ratio + conf_obj.test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_ratio_in_temp),
        random_state=conf_obj.random_seed,
        stratify=temp_df['label_id']
    )

    print(f"\n训练集数量: {len(train_df)} 条 ({len(train_df) / len(df) * 100:.2f}%)")
    print(f"验证集数量: {len(dev_df)} 条 ({len(dev_df) / len(df) * 100:.2f}%)")
    print(f"测试集数量: {len(test_df)} 条 ({len(test_df) / len(df) * 100:.2f}%)")

    # 验证每个集合的类别分布
    for name, subset in [("训练集", train_df), ("验证集", dev_df), ("测试集", test_df)]:
        counts = Counter(subset['label_id'])
        print(f"\n{name}各类别分布:")
        for i, label in enumerate(conf_obj.major_categories):
            print(f"  [{i}] {label}: {counts.get(i, 0)} 条")

    def save_dataset(data_df, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in data_df.iterrows():
                text = row['text'].replace('\t', ' ').replace('\n', ' ')
                f.write(f"{text}\t{row['label_id']}\n")
        print(f"已保存: {file_path}")

    save_dataset(train_df, conf_obj.train_path)
    save_dataset(dev_df, conf_obj.dev_path)
    save_dataset(test_df, conf_obj.test_path)

    with open(conf_obj.class_path, 'w', encoding='utf-8') as f:
        for label in conf_obj.major_categories:
            f.write(f"{label}\n")
    print(f"已保存: {conf_obj.class_path}")

    return train_df, dev_df, test_df


# ===============================================================
# 主流程
# ===============================================================
if __name__ == '__main__':
    # 第一步：加载数据
    df_raw = load_data(conf.raw_data_path)

    # 第二步：原始数据EDA分析
    df_raw = eda_analysis(df_raw, title="原始数据EDA分析")

    # 第三步：文本清洗
    print("\n" + "=" * 60)
    print("第三步：文本清洗")
    print("=" * 60)
    print(f"\n清洗前数据量: {len(df_raw)} 条")
    df_raw['text'] = df_raw['text'].apply(clean_text)
    df_raw = df_raw[df_raw['text'].str.strip() != '']
    print(f"清洗后数据量: {len(df_raw)} 条")

    # 第四步：类别合并 (v2: 8类, Payday -> Consumer loans)
    df_merged = merge_categories(df_raw)

    # 第五步：均衡采样 (v2 新方案)
    df_sampled = balanced_sampling(
        df_merged,
        n_per_class=conf.n_per_class,   # 每类 3000 条
        random_seed=conf.random_seed
    )

    # 第六步：划分数据集并保存
    train_df, dev_df, test_df = split_and_save(df_sampled, conf)

    print("\n" + "=" * 60)
    print("均衡采样数据预处理完成！")
    print(f"输出文件: {conf.train_path}, {conf.dev_path}, "
          f"{conf.test_path}, {conf.class_path}")
    print(f"模型训练时 num_classes = {conf.num_classes}")
    print("=" * 60)
