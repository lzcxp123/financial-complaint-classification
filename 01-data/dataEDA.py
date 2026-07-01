import re
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from hm_config import conf


def load_data(csv_path):
    """第一步：加载CSV数据，只保留投诉叙述非空的记录"""
    print("=" * 60)
    print("第一步：加载原始数据")
    print("=" * 60)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"原始数据总量: {len(df)} 条")

    # 只保留Consumer complaint narrative非空的记录
    df = df.dropna(subset=['Consumer complaint narrative'])
    df = df[df['Consumer complaint narrative'].str.strip() != '']
    print(f"去除空文本后数据量: {len(df)} 条")

    # 重命名列名，方便后续处理
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

    # 统计标签分布
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

    # 统计文本长度（字符数）
    df['text_len_char'] = df['text'].apply(lambda x: len(str(x)))
    print(f"\n文本长度统计（字符数）:")
    print(f"  平均长度: {df['text_len_char'].mean():.1f}")
    print(f"  最大长度: {df['text_len_char'].max()}")
    print(f"  最小长度: {df['text_len_char'].min()}")
    print(f"  中位数长度: {df['text_len_char'].median():.1f}")

    # 统计文本长度（词数）
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

    # 去除XXXX脱敏占位符
    text = re.sub(r'\bX{4,}\b', ' ', text)

    # 去除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)

    # 去除URL
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空格
    text = text.strip()

    return text


def merge_categories(df):
    """第四步：类别合并"""
    print("\n" + "=" * 60)
    print("第四步：类别合并")
    print("=" * 60)

    # 应用类别映射
    df['label'] = df['product_raw'].map(conf.category_mapping)

    # 统计映射后未匹配的类别
    unmatched = df[df['label'].isna()]['product_raw'].unique()
    if len(unmatched) > 0:
        print(f"\n警告：以下 {len(unmatched)} 个类别未匹配，将归入 'Other financial services':")
        for cat in unmatched:
            print(f"  - {cat}")
        df['label'] = df['label'].fillna('Other financial services')

    # 统计合并后的类别分布
    label_counts = Counter(df['label'])
    print(f"\n合并后类别数: {len(label_counts)}")
    print("\n合并后各类别数量:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} 条 ({count / len(df) * 100:.2f}%)")

    return df


def stratified_sampling(df, n=200000):
    """第五步：分层随机采样"""
    print("\n" + "=" * 60)
    print("第五步：分层随机采样")
    print("=" * 60)

    print(f"\n采样前数据量: {len(df)} 条")
    print(f"目标采样数: {n} 条")

    if len(df) <= n:
        print("数据量小于等于目标采样数，无需采样")
        return df

    # 按类别分层采样
    label_counts = Counter(df['label'])
    # 每个类别的采样比例
    sample_ratio = n / len(df)

    sampled_dfs = []
    for label, count in label_counts.items():
        label_df = df[df['label'] == label]
        n_sample = max(1, int(count * sample_ratio))
        if n_sample >= len(label_df):
            sampled_dfs.append(label_df)
        else:
            sampled_dfs.append(
                label_df.sample(n=n_sample, random_state=conf.random_seed)
            )

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    # 打乱顺序
    df_sampled = df_sampled.sample(frac=1, random_state=conf.random_seed).reset_index(drop=True)

    print(f"\n采样后数据量: {len(df_sampled)} 条")

    # 统计采样后的类别分布
    sampled_counts = Counter(df_sampled['label'])
    print("\n采样后各类别数量:")
    for label in conf.major_categories:
        count = sampled_counts.get(label, 0)
        print(f"  {label}: {count} 条 ({count / len(df_sampled) * 100:.2f}%)")

    return df_sampled


def split_and_save(df, conf_obj):
    """第六步：划分数据集并保存"""
    print("\n" + "=" * 60)
    print("第六步：划分数据集并保存")
    print("=" * 60)

    # 将标签转换为id
    label_to_id = {label: idx for idx, label in enumerate(conf_obj.major_categories)}
    df['label_id'] = df['label'].map(label_to_id)

    # 第一次划分：训练集 vs 临时集（验证+测试）
    train_df, temp_df = train_test_split(
        df,
        test_size=(conf_obj.dev_ratio + conf_obj.test_ratio),
        random_state=conf_obj.random_seed,
        stratify=df['label_id']
    )

    # 第二次划分：验证集 vs 测试集
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

    # 保存为tab分隔的txt文件
    def save_dataset(data_df, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in data_df.iterrows():
                text = row['text'].replace('\t', ' ').replace('\n', ' ')
                f.write(f"{text}\t{row['label_id']}\n")
        print(f"已保存: {file_path}")

    save_dataset(train_df, conf_obj.train_path)
    save_dataset(dev_df, conf_obj.dev_path)
    save_dataset(test_df, conf_obj.test_path)

    # 保存class.txt
    with open(conf_obj.class_path, 'w', encoding='utf-8') as f:
        for label in conf_obj.major_categories:
            f.write(f"{label}\n")
    print(f"已保存: {conf_obj.class_path}")

    return train_df, dev_df, test_df


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
    # 去除清洗后为空的文本
    df_raw = df_raw[df_raw['text'].str.strip() != '']
    print(f"清洗后数据量: {len(df_raw)} 条")

    # 第四步：类别合并
    df_merged = merge_categories(df_raw)

    # 第五步：分层采样
    df_sampled = stratified_sampling(df_merged, n=conf.sample_n)

    # 第六步：划分数据集并保存
    train_df, dev_df, test_df = split_and_save(df_sampled, conf)

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
