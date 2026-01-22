# @title
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm
from collections import Counter
import ast

# 方法一：设置Matplotlib无GUI后端（必须在import plt之前）
import matplotlib

matplotlib.use('Agg')  # 无GUI后端，只保存图片不显示
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
# from textblob import TextBlob
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

output_notebook()

from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import joblib

# %matplotlib inline

nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon')

# ===================== 统一加载数据，全程使用同一个DataFrame =====================
# 优先加载已处理数据，无则加载原始数据并处理
datafile = 'abcnews-date-text.csv'
try:
    # 尝试加载最终处理数据（如果之前运行过）
    df = pd.read_pickle('processed_eda_sentiment_df.pkl')
    print("Loaded processed DataFrame with sentiment from pickle.")
    # 补全可能缺失的列
    if 'word_count' not in df.columns:
        df['word_count'] = df['headline_text'].str.split().str.len()
    if 'char_count' not in df.columns:
        df['char_count'] = df['headline_text'].str.len()
    if 'year' not in df.columns:
        df['year'] = df['publish_date'].dt.year
    if 'month' not in df.columns:
        df['month'] = df['publish_date'].dt.month
except FileNotFoundError:
    # 加载原始数据并处理
    print("Loading raw data and processing...")
    raw_data = pd.read_csv(datafile, parse_dates=[0])  # 移除废弃参数
    df = raw_data.head(50000).copy()

    # 基础清洗
    df['headline_text'] = df['headline_text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # 文本长度特征
    df['word_count'] = df['headline_text'].str.split().str.len()  # 词数
    df['char_count'] = df['headline_text'].str.len()  # 字符数

    # 时间特征
    df['year'] = df['publish_date'].dt.year
    df['month'] = df['publish_date'].dt.month

    print(f"Dataset loaded and processed: {len(df)} headlines")

# 保存中间数据（仅一次）
df.to_pickle('processed_eda_df.pkl')
print("Intermediate DataFrame saved as 'processed_eda_df.pkl'")

# ===================== 数据基本统计 =====================
print("\n=== 數據基本統計 (Basic Dataset Stats) ===")
print(f"總 headlines 數: {len(df):,}")
print(f"平均詞數: {df['word_count'].mean():.2f} (std: {df['word_count'].std():.2f})")
print(f"平均字符數: {df['char_count'].mean():.2f} (std: {df['char_count'].std():.2f})")
print(f"詞數範圍: {df['word_count'].min()} - {df['word_count'].max()}")
print(f"字符數範圍: {df['char_count'].min()} - {df['char_count'].max()}")

# 词数/字符数分布可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['word_count'].hist(bins=20, ax=axes[0], edgecolor='black', color='skyblue')
axes[0].set_title('Word Count Distribution')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Frequency')

df['char_count'].hist(bins=20, ax=axes[1], edgecolor='black', color='lightgreen')
axes[1].set_title('Character Count Distribution')
axes[1].set_xlabel('Number of Characters')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('word_char_distribution.png', dpi=150, bbox_inches='tight')
print("图片已保存：word_char_distribution.png")

# ===================== 时间趋势分析 =====================
reindexed_data = df['headline_text'].copy()
reindexed_data.index = df['publish_date']

monthly_counts = reindexed_data.resample('M').count()
yearly_counts = reindexed_data.resample('A').count()
daily_counts = reindexed_data.resample('D').count()

fig, ax = plt.subplots(3, figsize=(18, 16))
ax[0].plot(daily_counts);
ax[0].set_title('Daily Counts');
ax[1].plot(monthly_counts);
ax[1].set_title('Monthly Counts');
ax[2].plot(yearly_counts);
ax[2].set_title('Yearly Counts');
plt.tight_layout()
plt.savefig('temporal_trends.png', dpi=150, bbox_inches='tight')
print("图片已保存：temporal_trends.png")

# ===================== 情感分析 =====================
# 安装并导入VADER
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 关键修复：提前定义analyzer，确保全局可用
analyzer = SentimentIntensityAnalyzer()

# 计算情感分数（确保只计算一次）
if 'sentiment_compound' not in df.columns:
    df['sentiment_compound'] = df['headline_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_pos'] = df['headline_text'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    df['sentiment_neg'] = df['headline_text'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
    df['sentiment_neu'] = df['headline_text'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
    # 保存包含情感分数的数据
    df.to_pickle('processed_eda_sentiment_df.pkl')
    print("DataFrame with sentiment saved as 'processed_eda_sentiment_df.pkl'")

# 情感统计
print("\n=== 情感基線統計 (Sentiment Baseline Stats) ===")
print(df['sentiment_compound'].describe())
print(f"平均情感分數: {df['sentiment_compound'].mean():.3f} (負面偏多?)")
print(f"正向比例 (>0.05): {(df['sentiment_compound'] > 0.05).mean():.1%}")
print(f"中性比例 (-0.05~0.05): {((df['sentiment_compound'] >= -0.05) & (df['sentiment_compound'] <= 0.05)).mean():.1%}")
print(f"負向比例 (<-0.05): {(df['sentiment_compound'] < -0.05).mean():.1%}")

# 情感分布可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['sentiment_compound'].hist(bins=30, edgecolor='black', color='lightblue', alpha=0.7)
plt.title('Sentiment Compound Score Distribution')
plt.xlabel('Compound Score (-1 to +1)')
plt.ylabel('Frequency')
plt.axvline(df['sentiment_compound'].mean(), color='red', linestyle='--',
            label=f'Mean: {df["sentiment_compound"].mean():.3f}')
plt.legend()

# 情感极性饼图
plt.subplot(1, 2, 2)
polarity_counts = df['sentiment_compound'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')).value_counts()
plt.pie(polarity_counts.values, labels=polarity_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Polarity Proportions')
plt.axis('equal')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
print("图片已保存：sentiment_distribution.png")

# ===================== 正负向高频词分析 =====================
positive_words = Counter()
negative_words = Counter()
stop_words = set(stopwords.words('english'))

for headline in df['headline_text']:
    words = headline.split()
    scores = analyzer.polarity_scores(headline)  # 现在analyzer一定已定义
    if scores['compound'] > 0.05:
        pos_words = [w for w in words if w not in stop_words and len(w) > 2]
        positive_words.update(pos_words)
    elif scores['compound'] < -0.05:
        neg_words = [w for w in words if w not in stop_words and len(w) > 2]
        negative_words.update(neg_words)

# 输出Top10正负向词
top_pos = positive_words.most_common(10)
top_neg = negative_words.most_common(10)

print(f"\n=== Top 10 Positive Words (from positive headlines) ===")
for word, count in top_pos:
    print(f"{word}: {count}")

print(f"\n=== Top 10 Negative Words (from negative headlines) ===")
for word, count in top_neg:
    print(f"{word}: {count}")

# 可视化正负向高频词
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pos_df = pd.DataFrame(top_pos, columns=['word', 'count'])
pos_df.plot.bar(x='word', y='count', ax=axes[0], color='green')
axes[0].set_title('Top Positive Words')
axes[0].set_xlabel('Words')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)

neg_df = pd.DataFrame(top_neg, columns=['word', 'count'])
neg_df.plot.bar(x='word', y='count', ax=axes[1], color='red')
axes[1].set_title('Top Negative Words')
axes[1].set_xlabel('Words')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('top_sentiment_words.png', dpi=150, bbox_inches='tight')
print("图片已保存：top_sentiment_words.png")

# ===================== 情感示例 =====================
print("\n=== 示例 Headlines (Negative vs Positive) ===")
neg_examples = df[df['sentiment_compound'] < -0.05]['headline_text'].head(3).tolist()
pos_examples = df[df['sentiment_compound'] > 0.05]['headline_text'].head(3).tolist()

print("Negative Examples:")
for ex in neg_examples:
    print(f"  - {ex} (score: {analyzer.polarity_scores(ex)['compound']:.3f})")

print("\nPositive Examples:")
for ex in pos_examples:
    print(f"  - {ex} (score: {analyzer.polarity_scores(ex)['compound']:.3f})")

# ===================== 数据质量检查 =====================
print("\n=== 缺失值检查 (Missing Values Check) ===")
missing_stats = df.isnull().sum()
print(missing_stats)
print(f"總缺失值: {missing_stats.sum()}")
print(f"缺失比例: {missing_stats.sum() / len(df) * 100:.2f}%")

# 缺失值热力图（如有缺失）
if missing_stats.sum() > 0:
    plt.figure(figsize=(8, 4))
    sb.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('missing_values_heatmap.png', dpi=150, bbox_inches='tight')
    print("图片已保存：missing_values_heatmap.png")
else:
    print("No missing values found - good data quality!")

# 去重（确保数据唯一性）
df = df.drop_duplicates(subset=['headline_text']).reset_index(drop=True)
print(f"去重后数据量：{len(df)} 条 headlines")

# ===================== TF-IDF 向量化 =====================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # 单字+双字短语
    stop_words='english',
    min_df=2,  # 忽略出现次数<2的词
    max_df=0.95  # 忽略出现频率>95%的词
)

X_tfidf = vectorizer.fit_transform(df['headline_text'])
feature_names = vectorizer.get_feature_names_out()

print("\n=== TF-IDF 基本統計 (TF-IDF Basic Stats) ===")
print(f"TF-IDF Matrix Shape: {X_tfidf.shape} (docs x features)")
print(f"Sparsity (non-zero %): {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]) * 100:.2f}%")
print(f"Vocabulary Size: {len(feature_names):,}")

# Top20 TF-IDF 术语
mean_tf_idf = np.asarray(X_tfidf.mean(axis=0)).flatten()
top_indices = mean_tf_idf.argsort()[-20:][::-1]
top_terms = [feature_names[i] for i in top_indices]
top_scores = mean_tf_idf[top_indices]

print(f"\n=== Top 20 TF-IDF Terms (Term Importance) ===")
for term, score in zip(top_terms, top_scores):
    print(f"{term}: {score:.4f}")

# 可视化Top20 TF-IDF术语
top_df = pd.DataFrame({'term': top_terms, 'score': top_scores})
plt.figure(figsize=(12, 6))
top_df.plot.bar(x='term', y='score', color='blue')
plt.title('Top 20 TF-IDF Terms (Quantifying Term Importance)')
plt.xlabel('Terms (Unigrams + Bigrams)')
plt.ylabel('Mean TF-IDF Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_tfidf_terms.png', dpi=150, bbox_inches='tight')
print("图片已保存：top_tfidf_terms.png")

# ===================== LDA 主题聚类 =====================
n_topics = 10  # 10个主题
lda = LDA(
    n_components=n_topics,
    random_state=42,
    max_iter=10,
    evaluate_every=-1
)

X_lda = lda.fit_transform(X_tfidf)
df['lda_topic'] = X_lda.argmax(axis=1)  # 为每条数据分配最可能的主题

# 输出每个主题的关键词（帮助理解主题含义）
print(f"\n=== LDA 主题关键词 (Top 5 words per topic) ===")
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]  # 每个主题Top5词
    print(f"主题 {topic_idx}: {' | '.join(top_words)}")

# ===================== 保存最终数据（包含LDA聚类和情感分数） =====================
# 确认所有必要列存在
required_columns = [
    'publish_date', 'headline_text', 'word_count', 'char_count', 'year', 'month',
    'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
    'lda_topic'
]

# 补全可能缺失的列（防御性编程）
for col in required_columns:
    if col not in df.columns:
        if col == 'word_count':
            df[col] = df['headline_text'].str.split().str.len()
        elif col == 'char_count':
            df[col] = df['headline_text'].str.len()
        elif col in ['year', 'month']:
            df[col] = df['publish_date'].dt.__getattribute__(col)
        print(f"⚠️  补全缺失列：{col}")

# 打印保存的列信息
print("\n=== 最终保存的列信息 ===")
for col in required_columns:
    print(f"✅ {col} (数据类型: {df[col].dtype})")

# 双格式保存
# 1. Pickle格式（保留完整数据类型，推荐后续分析使用）
df.to_pickle('final_data_with_topic_sentiment.pkl')
# 2. CSV格式（通用性强，方便查看分享）
df.to_csv('final_data_with_topic_sentiment.csv', index=False, encoding='utf-8')

# 保存LDA模型和TF-IDF向量器（方便后续复用）
joblib.dump(lda, 'lda_topic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 输出保存结果
print(f"\n📊 数据保存完成！")
print(f"📁 主数据文件：")
print(f"   - Pickle: final_data_with_topic_sentiment.pkl (保留datetime等类型)")
print(f"   - CSV: final_data_with_topic_sentiment.csv (通用格式)")
print(f"📁 模型文件：")
print(f"   - LDA模型: lda_topic_model.pkl")
print(f"   - TF-IDF向量器: tfidf_vectorizer.pkl")
print(f"📈 最终数据量：{len(df)} 条 headlines")

# 输出LDA主题分布
print(f"\n=== LDA 主题分布 ===")
topic_dist = df['lda_topic'].value_counts().sort_index()
for topic_id, count in topic_dist.items():
    print(f"主题 {topic_id}: {count} 条 ({count / len(df) * 100:.1f}%)")