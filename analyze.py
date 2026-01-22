# @title 数据集可视化分析（基于 LDA 主题 + 情感分数）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from collections import Counter

# 继续使用无GUI后端（保存图片）
import matplotlib

matplotlib.use('Agg')

# 设置中文字体（避免中文乱码，Windows系统）- 保留以兼容可能的中文关键词，不影响英文标签
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # SimHei=黑体，兼容中文和英文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 1. 加载数据和必要模型 =====================
# 加载最终数据集
df = pd.read_pickle('full_data_with_topic_sentiment.pkl')
print(f"✅ 数据集加载成功，共 {len(df)} 条数据")

# 加载 LDA 模型和 TF-IDF 向量器（获取主题关键词）
import joblib

lda = joblib.load('lda_topic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
feature_names = vectorizer.get_feature_names_out()

# 预定义情感极性标签（沿用之前的划分标准）
df['sentiment_polarity'] = df['sentiment_compound'].apply(
    lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
)

# 获取每个主题的Top10关键词（用于图表标注）
topic_keywords = {}
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]  # Top10词
    topic_keywords[topic_idx] = ', '.join(top_words[:5])  # 每个主题显示前5个关键词

print(f"\n=== 主题关键词对照表 ===")
for topic_id, keywords in topic_keywords.items():
    print(f"主题 {topic_id}: {keywords}")


# ===================== 2. 可视化函数定义（模块化） =====================
def plot_topic_distribution():
    """1. LDA 主题分布（饼图 + 条形图）"""
    topic_counts = df['lda_topic'].value_counts().sort_index()
    topic_labels = [f"topic {i}\n({topic_keywords[i]})" for i in topic_counts.index]

    # 子图：饼图（占比）+ 条形图（数量）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
    wedges, texts, autotexts = ax1.pie(
        topic_counts.values, labels=topic_labels, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 9}
    )
    ax1.set_title('LDA Topic Distribution Ratio', fontsize=14, fontweight='bold')

    # 条形图
    bars = ax2.bar(topic_counts.index, topic_counts.values, color=colors)
    ax2.set_title('Number of Samples per Topic', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Topic ID', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_xticks(topic_counts.index)
    ax2.set_xticklabels([f"Topic {i}" for i in topic_counts.index])

    # 在条形图上添加数值标签
    for bar, count in zip(bars, topic_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 50,
                 f'{count:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('1_topic_distribution.png', dpi=150, bbox_inches='tight')
    print("📊 主题分布图已保存：1_topic_distribution.png")


def plot_topic_sentiment_boxplot():
    """2. 各主题情感分数分布（箱线图）"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # 按主题分组的情感分数箱线图
    box_data = [df[df['lda_topic'] == i]['sentiment_compound'].values for i in range(10)]
    box_plot = ax.boxplot(
        box_data, labels=[f"Topic {i}" for i in range(10)],
        patch_artist=True, showfliers=False  # 隐藏异常值，更清晰
    )

    # 设置箱线图颜色
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 10))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    # 添加参考线（0分：中性基准）
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Neutral Baseline (0 Score)')

    ax.set_title('Sentiment Score Distribution by Topic (Compound Score: -1 Negative ~ +1 Positive)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Topic ID', fontsize=12)
    ax.set_ylabel('Sentiment Compound Score', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('2_topic_sentiment_boxplot.png', dpi=150, bbox_inches='tight')
    print("📊 主题-情感箱线图已保存：2_topic_sentiment_boxplot.png")


def plot_topic_sentiment_heatmap():
    """3. 主题-情感极性交叉热力图（统计各主题的情感极性占比）"""
    # 构建交叉表：主题 × 情感极性（使用英文标签）
    cross_tab = pd.crosstab(df['lda_topic'], df['sentiment_polarity'], normalize='index') * 100  # 按主题归一化（百分比）
    cross_tab = cross_tab[['positive', 'neutral', 'negative']]  # 调整列顺序为英文

    # 绘制热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns_heatmap = sb.heatmap(
        cross_tab, annot=True, fmt='.1f', cmap='RdYlGn_r',
        ax=ax, cbar_kws={'label': 'Percentage (%)'},
        annot_kws={'fontsize': 10}
    )

    ax.set_title('Sentiment Polarity Distribution Heatmap by Topic', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sentiment Polarity', fontsize=12)
    ax.set_ylabel('Topic ID', fontsize=12)
    ax.set_yticklabels([f"Topic {i}" for i in cross_tab.index], rotation=0)

    plt.tight_layout()
    plt.savefig('3_topic_sentiment_heatmap.png', dpi=150, bbox_inches='tight')
    print("📊 主题-情感热力图已保存：3_topic_sentiment_heatmap.png")


def plot_temporal_topic_trend():
    """4. 主题时间趋势（按年/月统计各主题发布数量）"""
    # 按年-主题统计数量
    yearly_topic = df.groupby(['year', 'lda_topic']).size().unstack(fill_value=0)

    # 绘制堆叠面积图
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    yearly_topic.plot.area(
        ax=ax, stacked=True, color=colors, alpha=0.7,
        linewidth=1
    )

    ax.set_title('Annual Publication Trend by Topic', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Publications', fontsize=12)
    ax.legend(title='Topic ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('4_temporal_topic_trend.png', dpi=150, bbox_inches='tight')
    print("📊 主题时间趋势图已保存：4_temporal_topic_trend.png")


def plot_temporal_sentiment_trend():
    """5. 情感时间趋势（按年统计平均情感分数）"""
    # 按年统计平均情感分数
    yearly_sentiment = df.groupby('year')['sentiment_compound'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 绘制带误差条的折线图
    ax.errorbar(
        yearly_sentiment['year'], yearly_sentiment['mean'],
        yerr=yearly_sentiment['std'] / np.sqrt(len(df) / len(yearly_sentiment)),  # 标准误
        fmt='o-', linewidth=2, markersize=6, color='darkblue',
        ecolor='lightblue', capsize=5, label='Average Sentiment Score'
    )

    # 添加中性基准线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Neutral Baseline')

    ax.set_title('Annual Average Sentiment Score Trend (Error Bars = Standard Error)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Sentiment Compound Score', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('5_temporal_sentiment_trend.png', dpi=150, bbox_inches='tight')
    print("📊 情感时间趋势图已保存：5_temporal_sentiment_trend.png")


def plot_topic_wordcloud():
    """6. 各主题关键词词云（每个主题生成一个词云）"""
    # 创建2×5的子图布局
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()  # 展平为一维数组，方便循环

    colors_list = list(mcolors.TABLEAU_COLORS.values())  # 颜色列表

    for topic_idx, ax in enumerate(axes):
        # 获取当前主题的所有文本
        topic_text = ' '.join(df[df['lda_topic'] == topic_idx]['headline_text'].tolist())

        # 生成词云
        wordcloud = WordCloud(
            width=400, height=300,
            background_color='white',
            max_words=50,
            colormap='viridis',
            stopwords=set(stopwords.words('english')),
            font_path=None  # 英文无需指定字体
        ).generate(topic_text)

        # 显示词云
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {topic_idx}\n{topic_keywords[topic_idx]}', fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig('6_topic_wordcloud.png', dpi=150, bbox_inches='tight')
    print("📊 主题词云图已保存：6_topic_wordcloud.png")


def plot_sentiment_wordcloud():
    """7. 情感极性关键词词云（正向/负向/中性对比）"""
    # 按情感极性分组文本
    pos_text = ' '.join(df[df['sentiment_polarity'] == 'positive']['headline_text'].tolist())
    neg_text = ' '.join(df[df['sentiment_polarity'] == 'negative']['headline_text'].tolist())
    neu_text = ' '.join(df[df['sentiment_polarity'] == 'neutral']['headline_text'].tolist())

    # 创建1×3的子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 定义词云参数
    wc_params = {
        'width': 500, 'height': 400,
        'background_color': 'white',
        'max_words': 80,
        'stopwords': set(stopwords.words('english')),
        'font_path': None
    }

    # 正向词云（绿色系）
    WordCloud(colormap='Greens', **wc_params).generate(pos_text).to_image()
    ax1.imshow(WordCloud(colormap='Greens', **wc_params).generate(pos_text), interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title('Positive Sentiment Keywords', fontsize=12, fontweight='bold')

    # 中性词云（灰色系）
    ax2.imshow(WordCloud(colormap='Greys', **wc_params).generate(neu_text), interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title('Neutral Sentiment Keywords', fontsize=12, fontweight='bold')

    # 负向词云（红色系）
    ax3.imshow(WordCloud(colormap='Reds', **wc_params).generate(neg_text), interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Negative Sentiment Keywords', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('7_sentiment_wordcloud.png', dpi=150, bbox_inches='tight')
    print("📊 情感关键词词云图已保存：7_sentiment_wordcloud.png")


def plot_topic_sentiment_histogram():
    """8. 各主题情感分数直方图（对比分布差异）"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))

    for topic_idx, ax in enumerate(axes):
        # 获取当前主题的情感分数
        sentiment_scores = df[df['lda_topic'] == topic_idx]['sentiment_compound']

        # 绘制直方图
        ax.hist(
            sentiment_scores, bins=20, color=colors[topic_idx],
            alpha=0.7, edgecolor='black', linewidth=0.5
        )

        # 添加均值线
        mean_score = sentiment_scores.mean()
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=1,
                   label=f'Mean: {mean_score:.3f}')

        ax.set_title(f'Topic {topic_idx}', fontsize=10)
        ax.set_xlabel('Sentiment Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('8_topic_sentiment_histogram.png', dpi=150, bbox_inches='tight')
    print("📊 主题情感分数直方图已保存：8_topic_sentiment_histogram.png")


# ===================== 3. 执行所有可视化 =====================
if __name__ == "__main__":
    # 导入必要的额外库（之前代码中已下载stopwords）
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords', quiet=True)

    # 依次执行可视化函数
    plot_topic_distribution()
    plot_topic_sentiment_boxplot()
    plot_topic_sentiment_heatmap()
    plot_temporal_topic_trend()
    plot_temporal_sentiment_trend()
    plot_topic_wordcloud()
    plot_sentiment_wordcloud()
    plot_topic_sentiment_histogram()

    print("\n🎉 所有可视化分析完成！共生成 8 张图表：")
    print("1. 1_topic_distribution.png - Topic Distribution (Pie + Bar Chart)")
    print("2. 2_topic_sentiment_boxplot.png - Topic-Sentiment Boxplot")
    print("3. 3_topic_sentiment_heatmap.png - Topic-Sentiment Heatmap")
    print("4. 4_temporal_topic_trend.png - Temporal Topic Trend Chart")
    print("5. 5_temporal_sentiment_trend.png - Temporal Sentiment Trend Chart")
    print("6. 6_topic_wordcloud.png - Topic Wordclouds")
    print("7. 7_sentiment_wordcloud.png - Sentiment Polarity Wordclouds")
    print("8. 8_topic_sentiment_histogram.png - Topic Sentiment Score Histograms")