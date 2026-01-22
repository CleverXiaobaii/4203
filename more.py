# @title 用已训练模型处理全部数据（修复正则替换错误）
import numpy as np
import pandas as pd
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 下载必要的NLTK资源（如果未下载）
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# ===================== 1. 加载已训练的模型和向量器 =====================
print("正在加载模型和向量器...")
try:
    lda_model = joblib.load('lda_topic_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ 模型和向量器加载成功！")
except FileNotFoundError as e:
    print(f"❌ 找不到模型文件：{e}")
    print("请确保 lda_topic_model.pkl 和 tfidf_vectorizer.pkl 在当前目录")
    exit()

# ===================== 2. 加载全部原始数据 =====================
print("\n正在加载全部原始数据...")
datafile = 'abcnews-date-text.csv'  # 完整原始数据文件
try:
    # 加载全部数据（不限制50K）
    raw_data = pd.read_csv(datafile, parse_dates=[0])  # 保留publish_date为datetime类型
    print(f"✅ 原始数据加载成功，共 {len(raw_data):,} 条数据")
except FileNotFoundError:
    print(f"❌ 找不到原始数据文件：{datafile}")
    print("请确保 abcnews-date-text.csv 在当前目录")
    exit()

# ===================== 3. 数据预处理（修复正则替换错误，用pandas str方法） =====================
print("\n正在预处理数据...")

# 方法1：直接用pandas str方法（推荐，效率高，支持regex）
df_full = raw_data.copy()

# 1. 小写转换（pandas str.lower()）
df_full['headline_text'] = df_full['headline_text'].str.lower()

# 2. 移除标点（pandas str.replace()，支持regex=True）
df_full['headline_text'] = df_full['headline_text'].str.replace(r'[^\w\s]', '', regex=True)

# 3. 处理可能的NaN值（如果标题为空）
df_full = df_full.dropna(subset=['headline_text'])
df_full = df_full[df_full['headline_text'].str.strip() != '']  # 移除空字符串标题

# 特征工程（和之前一致）
df_full['word_count'] = df_full['headline_text'].str.split().str.len()
df_full['char_count'] = df_full['headline_text'].str.len()
df_full['year'] = df_full['publish_date'].dt.year
df_full['month'] = df_full['publish_date'].dt.month

# 去重
df_full = df_full.drop_duplicates(subset=['headline_text']).reset_index(drop=True)
print(f"预处理完成，去重后剩余 {len(df_full):,} 条数据")

# ===================== 4. 分批次预测主题（避免内存溢出） =====================
print("\n正在预测全部数据的LDA主题...")
# 分批次处理（120万条数据一次性处理可能内存不足，每批1万条）
batch_size = 10000
df_full['lda_topic'] = -1  # 初始化主题列
total_batches = len(df_full) // batch_size + 1

for batch_idx in range(total_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(df_full))
    batch_text = df_full.iloc[start:end]['headline_text']

    # TF-IDF向量化
    X_batch_tfidf = tfidf_vectorizer.transform(batch_text)

    # 主题预测
    batch_topics = lda_model.transform(X_batch_tfidf).argmax(axis=1)

    # 赋值到原数据框
    df_full.iloc[start:end, df_full.columns.get_loc('lda_topic')] = batch_topics

    # 打印进度
    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        print(f"已完成 {batch_idx + 1}/{total_batches} 批次（{end:,}/{len(df_full):,} 条）")

print("✅ 全部数据主题预测完成")

# ===================== 5. 分批次计算情感分数 =====================
print("\n正在计算全部数据的情感分数...")
analyzer = SentimentIntensityAnalyzer()


# 分批次计算情感分数（避免一次性处理压力）
def calculate_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return pd.Series([scores['compound'], scores['pos'], scores['neg'], scores['neu']])


# 初始化情感列
df_full[['sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']] = 0.0

for batch_idx in range(total_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(df_full))
    batch_text = df_full.iloc[start:end]['headline_text']

    # 计算情感分数
    batch_sentiments = batch_text.apply(calculate_sentiment)

    # 赋值到原数据框
    df_full.iloc[start:end, df_full.columns.get_loc('sentiment_compound')] = batch_sentiments[0].values
    df_full.iloc[start:end, df_full.columns.get_loc('sentiment_pos')] = batch_sentiments[1].values
    df_full.iloc[start:end, df_full.columns.get_loc('sentiment_neg')] = batch_sentiments[2].values
    df_full.iloc[start:end, df_full.columns.get_loc('sentiment_neu')] = batch_sentiments[3].values

    # 打印进度
    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
        print(f"已完成 {batch_idx + 1}/{total_batches} 批次（{end:,}/{len(df_full):,} 条）")

# 添加情感极性标签
df_full['sentiment_polarity'] = df_full['sentiment_compound'].apply(
    lambda x: '正向' if x > 0.05 else ('负向' if x < -0.05 else '中性')
)
print("✅ 全部数据情感分数计算完成")

# ===================== 6. 保存处理后的完整数据 =====================
print("\n正在保存完整数据...")
output_pickle = 'full_data_with_topic_sentiment.pkl'
output_csv = 'full_data_with_topic_sentiment.csv'

# 保存为pickle（推荐，保留数据类型，加载更快）
df_full.to_pickle(output_pickle)
# 保存为csv（可选，数据量大可能需要几分钟）
# df_full.to_csv(output_csv, index=False, encoding='utf-8')  # 如需CSV格式，取消注释

print(f"✅ 完整数据保存完成！")
print(f"📁 Pickle格式（推荐可视化使用）：{output_pickle}")
# print(f"📁 CSV格式（方便查看）：{output_csv}")  # 如需CSV格式，取消注释
print(f"📈 数据量：{len(df_full):,} 条")

# ===================== 7. 验证数据格式 =====================
print("\n=== 数据格式验证 ===")
required_cols = [
    'publish_date', 'headline_text', 'word_count', 'char_count', 'year', 'month',
    'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
    'lda_topic', 'sentiment_polarity'
]
missing_cols = [col for col in required_cols if col not in df_full.columns]
if not missing_cols:
    print("✅ 所有必要列都存在，可直接对接可视化脚本！")
else:
    print(f"⚠️  缺少列：{missing_cols}")

# 输出主题分布预览
print(f"\n=== 全部数据主题分布预览 ===")
topic_dist = df_full['lda_topic'].value_counts().sort_index()
for topic_id, count in topic_dist.items():
    print(f"主题 {topic_id}: {count:,} 条 ({count / len(df_full) * 100:.1f}%)")
