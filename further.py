# @title Advanced Analysis: RoBERTa Sentiment, Anomaly Detection, Sensationalism Scoring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import re
import joblib

# 设置可视化样式（移除中文依赖，全部使用英文）
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ===================== 加载基础数据和模型 =====================
# 加载数据集（你的数据有 49,831 条，是子集，不影响）
df = pd.read_pickle('final_data_with_topic_sentiment.pkl')
print(f"✅ 基础数据集加载成功，共 {len(df):,} 条数据")


# ---------------------- 强制生成 sentiment_polarity 列（关键修复）----------------------
# 直接根据 sentiment_compound 计算，覆盖或新增该列
def get_sentiment_polarity(compound):
    if compound > 0.05:
        return 'Positive'  # 改为英文，避免后续处理中文
    elif compound < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# 强制生成列（不管之前有没有）
df['sentiment_polarity'] = df['sentiment_compound'].apply(get_sentiment_polarity)

# 打印列名确认，让你看到该列已存在
print(f"✅ 数据集当前列名：{df.columns.tolist()}")
print(f"✅ sentiment_polarity 列生成成功，分布：\n{df['sentiment_polarity'].value_counts()}")

# 加载 LDA 模型（用于主题匹配）
try:
    lda = joblib.load('lda_topic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ LDA 模型和 TF-IDF 向量器加载成功")
except FileNotFoundError as e:
    print(f"⚠️  未找到模型文件：{e}，主题相关功能可能受影响")


# ===================== 1. Advanced Sentiment Model: VADER vs RoBERTa =====================
def generate_annotation_template(n_samples=150):
    """
    生成手动标注模板（CSV文件），用于对比 VADER 和 RoBERTa 准确性
    随机抽取 150 条数据，涵盖不同主题和 VADER 情感极性
    """
    try:
        # 分层抽样：添加 group_keys=False 消除 DeprecationWarning
        sample_df = df.groupby(['lda_topic', 'sentiment_polarity'], dropna=True, group_keys=False).apply(
            lambda x: x.sample(min(5, len(x)), random_state=42)
        ).reset_index(drop=True)
    except:
        # 若分层抽样失败（比如某些主题+情感组合为空），改用简单随机抽样（确保能生成模板）
        print("⚠️  分层抽样失败，改用简单随机抽样生成模板")
        sample_df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    # 补充到目标样本数
    if len(sample_df) < n_samples:
        remaining = n_samples - len(sample_df)
        补充_samples = df.drop(sample_df.index).sample(remaining, random_state=42)
        sample_df = pd.concat([sample_df, 补充_samples], ignore_index=True)

    # 生成标注模板（字段名改为英文，方便标注）
    annotation_template = sample_df[['headline_text', 'lda_topic', 'sentiment_compound', 'sentiment_polarity']].copy()
    annotation_template['manual_sentiment'] = ''  # 填写：1=Positive, 0=Neutral, -1=Negative
    annotation_template['notes'] = ''  # 可选：记录讽刺、混合情感（英文备注）

    annotation_template.to_csv('sentiment_annotation_template.csv', index=False, encoding='utf-8')
    print(f"📋 手动标注模板已生成：sentiment_annotation_template.csv")
    print(f"❕ 操作说明：打开 CSV 文件，在 manual_sentiment 列填写真实情感（1=Positive, 0=Neutral, -1=Negative）")


def compare_vader_roberta(annotated_csv='sentiment_annotation_template.csv'):
    """
    对比 VADER 和 RoBERTa 的情感预测准确性
    输入：填写完成的手动标注 CSV 文件
    输出：准确率对比、混淆矩阵、细微情感捕捉案例
    """
    # 加载手动标注数据
    try:
        annotated_df = pd.read_csv(annotated_csv)
        # 过滤未标注数据和无效标注（仅保留 1/0/-1）
        annotated_df = annotated_df[
            (annotated_df['manual_sentiment'].notna()) &
            (annotated_df['manual_sentiment'].isin([1, 0, -1]))
            ].reset_index(drop=True)
        print(f"✅ 加载标注数据成功，共 {len(annotated_df)} 条有效标注")
    except FileNotFoundError:
        print(f"❌ 未找到标注文件：{annotated_csv}")
        print("请确认文件路径正确，且已填写 manual_sentiment 列")
        return

    # ---------------------- VADER 预测结果（基于已有的 compound 分数）----------------------
    def vader_to_label(compound):
        """将 VADER 的 compound 分数转换为 1/0/-1 标签（与手动标注一致）"""
        if compound > 0.05:
            return 1
        elif compound < -0.05:
            return -1
        else:
            return 0

    annotated_df['vader_label'] = annotated_df['sentiment_compound'].apply(vader_to_label)

    # ---------------------- 三分类 RoBERTa 模型（关键修改：支持 Positive/Neutral/Negative）----------------------
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # 设置设备（优先 GPU，无则用 CPU；GPU 运行 150 条样本约 1-2 分钟）
    device = 0 if torch.cuda.is_available() else -1
    print(f"⚙️ RoBERTa 运行设备：{'GPU' if device == 0 else 'CPU'}（三分类模型，GPU 可提速 10x+）")

    # 三分类预训练模型（专门适配情感三分类，无需近似计算中性概率）
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # 约 470MB，首次运行自动下载
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        roberta_model = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=tokenizer,
            device=device,
            top_k=None  # 替代 deprecated 的 return_all_scores=True，消除警告
        )
        print("✅ 三分类 RoBERTa 模型加载成功")
    except Exception as e:
        print(f"❌ RoBERTa 模型加载失败：{e}")
        print("解决方案：1. 检查网络连接（首次下载约 470MB）；2. 升级 transformers 库：pip install -U transformers")
        return

    # 修正 RoBERTa 预测逻辑（适配三分类输出，直接获取模型的中性标签）
    def roberta_predict(text):
        results = roberta_model(text)[0]
        # 构建标签-概率映射（适配模型的小写标签输出）
        score_dict = {}
        for res in results:
            label = res['label'].strip()  # 关键修改：取消 .upper()，保留原始小写标签
            score_dict[label] = res['score']

        # 取概率最大的标签转换为 1/0/-1（匹配小写标签）
        max_label = max(score_dict, key=score_dict.get)
        if max_label == 'positive':  # 小写标签
            return 1, score_dict['positive']
        elif max_label == 'neutral':  # 小写标签
            return 0, score_dict['neutral']
        elif max_label == 'negative':  # 小写标签
            return -1, score_dict['negative']
        else:
            # 异常情况默认中性
            return 0, 0.0

    # 批量预测（避免重复调用模型，提升效率）
    print("🔄 正在用 RoBERTa 预测情感...（150 条样本约 1-2 分钟，GPU 更快）")
    roberta_labels = []
    roberta_confidences = []
    for text in annotated_df['headline_text'].tolist():
        label, conf = roberta_predict(text)
        roberta_labels.append(label)
        roberta_confidences.append(conf)

    annotated_df['roberta_label'] = roberta_labels
    annotated_df['roberta_confidence'] = roberta_confidences

    # ---------------------- 模型性能对比（核心结果）----------------------
    # 计算准确率
    vader_acc = accuracy_score(annotated_df['manual_sentiment'], annotated_df['vader_label'])
    roberta_acc = accuracy_score(annotated_df['manual_sentiment'], annotated_df['roberta_label'])

    print("\n" + "=" * 50)
    print("=== VADER vs RoBERTa 情感预测准确率对比 ===")
    print(f"VADER 准确率：{vader_acc:.2%}")
    print(f"RoBERTa 准确率：{roberta_acc:.2%}")
    print(f"RoBERTa 相对提升：{((roberta_acc - vader_acc) / vader_acc * 100):.1f}%")
    print("=" * 50)

    # 生成分类报告（添加 zero_division=0 消除未预测类别的警告）
    print("\n=== VADER 分类报告 ===")
    print(classification_report(
        annotated_df['manual_sentiment'],
        annotated_df['vader_label'],
        target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
        zero_division=0  # 消除未预测类别的警告
    ))

    print("=== RoBERTa 分类报告 ===")
    print(classification_report(
        annotated_df['manual_sentiment'],
        annotated_df['roberta_label'],
        target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
        zero_division=0  # 消除未预测类别的警告
    ))

    # ---------------------- 混淆矩阵可视化（学术报告必备图）----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # VADER 混淆矩阵
    vader_cm = confusion_matrix(annotated_df['manual_sentiment'], annotated_df['vader_label'])
    sb.heatmap(
        vader_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
        xticklabels=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
        yticklabels=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    )
    ax1.set_title(f'VADER Confusion Matrix (Accuracy: {vader_acc:.2%})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=10)
    ax1.set_ylabel('True Label', fontsize=10)

    # RoBERTa 混淆矩阵
    roberta_cm = confusion_matrix(annotated_df['manual_sentiment'], annotated_df['roberta_label'])
    sb.heatmap(
        roberta_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
        xticklabels=['Negative (-1)', 'Neutral (0)', 'Positive (1)'],
        yticklabels=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    )
    ax2.set_title(f'RoBERTa Confusion Matrix (Accuracy: {roberta_acc:.2%})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=10)
    ax2.set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig('vader_roberta_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n📊 混淆矩阵图已保存：vader_roberta_confusion_matrix.png（直接插入学术报告）")

    # ---------------------- 细微情感捕捉案例（体现创新性的关键）----------------------
    # 筛选 RoBERTa 正确但 VADER 错误的案例（重点分析讽刺、混合情感）
    correct_roberta_incorrect_vader = annotated_df[
        (annotated_df['roberta_label'] == annotated_df['manual_sentiment']) &
        (annotated_df['vader_label'] != annotated_df['manual_sentiment'])
        ].head(8)  # 取前 8 个典型案例

    print("\n" + "=" * 60)
    print("=== RoBERTa 捕捉细微情感的典型案例（VADER 误判）===")
    print("注：这些案例体现了预训练语言模型对语义的深层理解优势")
    print("=" * 60)

    for idx, row in correct_roberta_incorrect_vader.iterrows():
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        print(f"\n📝 新闻标题：{row['headline_text']}")
        print(f"📌 真实情感（手动标注）：{row['manual_sentiment']}（{sentiment_map[row['manual_sentiment']]}）")
        print(f"❌ VADER 预测：{row['vader_label']}（Compound 分数：{row['sentiment_compound']:.3f}）")
        print(f"✅ RoBERTa 预测：{row['roberta_label']}（置信度：{row['roberta_confidence']:.3f}）")
        # 智能判断情感类型
        text_lower = row['headline_text'].lower()
        if 'won' in text_lower and ('lose' in text_lower or 'defeat' in text_lower):
            reason = '讽刺（Sarcasm）'
        elif 'but' in text_lower or 'however' in text_lower:
            reason = '混合情感（Mixed Sentiment）'
        elif row['manual_sentiment'] == 0 and (row['vader_label'] == 1 or row['vader_label'] == -1):
            reason = '中性识别（Neutral Recognition）'
        else:
            reason = '语境依赖（Context-Dependent）'
        print(f"💡 原因分析：{reason}")

    # ---------------------- 保存完整结果（方便后续引用）----------------------
    annotated_df.to_csv('vader_roberta_comparison_results.csv', index=False, encoding='utf-8')
    print("\n📄 完整对比结果已保存：vader_roberta_comparison_results.csv（包含所有预测标签和置信度）")


# ===================== 2. Topic–Sentiment Anomaly Detection =====================
def topic_sentiment_anomaly_detection():
    """主题-情感异常检测（无需修改，已适配你的列）"""
    # 按主题计算情感统计量
    topic_sentiment_stats = df.groupby('lda_topic')['sentiment_compound'].agg(['mean', 'std']).reset_index()
    topic_sentiment_stats.columns = ['lda_topic', 'topic_sentiment_mean', 'topic_sentiment_std']

    df_with_stats = df.merge(topic_sentiment_stats, on='lda_topic', how='left')
    df_with_stats['z_score'] = (df_with_stats['sentiment_compound'] - df_with_stats['topic_sentiment_mean']) / \
                               df_with_stats['topic_sentiment_std'].replace(0, 0.001)  # 避免除零
    df_with_stats['is_anomaly'] = abs(df_with_stats['z_score']) > 2

    # 统计结果
    anomaly_count = df_with_stats['is_anomaly'].sum()
    anomaly_ratio = anomaly_count / len(df_with_stats)
    print(f"\n=== 主题-情感异常检测结果 ===")
    print(f"异常值总数：{anomaly_count:,} 条")
    print(f"异常值比例：{anomaly_ratio:.2%}")

    # 可视化（所有中文改为英文，消除字体警告）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    anomaly_by_topic = df_with_stats[df_with_stats['is_anomaly']].groupby('lda_topic').size().sort_values(
        ascending=False)
    anomaly_by_topic.plot(kind='bar', color='orange', ax=ax1)
    ax1.set_title('Number of Sentiment Anomalies by Topic', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Topic ID', fontsize=10)
    ax1.set_ylabel('Number of Anomalies', fontsize=10)
    ax1.tick_params(axis='x', rotation=0)

    ax2.hist(
        df_with_stats[~df_with_stats['is_anomaly']]['sentiment_compound'],
        bins=50, alpha=0.5, label='Normal', color='blue', density=True
    )
    ax2.hist(
        df_with_stats[df_with_stats['is_anomaly']]['sentiment_compound'],
        bins=50, alpha=0.7, label='Anomalous', color='red', density=True
    )
    ax2.set_title('Sentiment Score Distribution: Normal vs Anomalous', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sentiment Compound Score', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('topic_sentiment_anomaly_distribution.png', dpi=150, bbox_inches='tight')
    print("📊 异常值分布可视化已保存")

    # 保存结果
    df_with_stats.to_pickle('full_data_with_anomaly_detection.pkl')
    df_with_stats[['headline_text', 'lda_topic', 'sentiment_compound', 'z_score', 'is_anomaly']].to_csv(
        'sentiment_anomaly_results.csv', index=False, encoding='utf-8'
    )


# ===================== 3. Sensationalism Scoring =====================
def calculate_sensationalism_score():
    """耸人听闻程度评分（无需修改，已适配你的列）"""
    df_sens = df.copy()

    # 特征1：极端情感
    df_sens['extreme_sentiment'] = df_sens['sentiment_compound'].apply(lambda x: x*x)#1 if abs(x) > 0.5 else 0)

    # 特征2：全大写词比例
    def uppercase_ratio(text):
        words = text.split()
        if len(words) == 0:
            return 0
        uppercase_words = [word for word in words if word.isupper() and len(word) > 1]
        return len(uppercase_words) / len(words)

    df_sens['uppercase_ratio'] = df_sens['headline_text'].apply(uppercase_ratio)

    # 特征3：标点计数
    df_sens['exclamation_count'] = df_sens['headline_text'].apply(lambda x: x.count('!'))
    df_sens['question_count'] = df_sens['headline_text'].apply(lambda x: x.count('?'))
    max_punct = df_sens[['exclamation_count', 'question_count']].max().max()
    df_sens['punctuation_score'] = (df_sens['exclamation_count'] + df_sens['question_count']) / (max_punct + 1)

    # 特征4：点击诱饵短语
    clickbait_phrases = [
        "you won't believe", "shocking", "at risk", "breaking", "exclusive",
        "must see", "never before", "secret", "revealed", "how to",
        "this is why", "what happened next", "unbelievable", "terrifying",
        "urgent", "alert", "don't miss", "viral", "explosive"
    ]

    def clickbait_match(text):
        text_lower = text.lower()
        match_count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
        return min(match_count / 3, 1)

    df_sens['clickbait_score'] = df_sens['headline_text'].apply(clickbait_match)

    # 计算最终分数
    weights = {'extreme_sentiment': 1, 'uppercase_ratio': 0, 'punctuation_score': 0, 'clickbait_score': 0}
    df_sens['sensationalism_score'] = (
            df_sens['extreme_sentiment'] * weights['extreme_sentiment'] +
            df_sens['uppercase_ratio'] * weights['uppercase_ratio'] +
            df_sens['punctuation_score'] * weights['punctuation_score'] +
            df_sens['clickbait_score'] * weights['clickbait_score']
    )

    # 归一化
    scaler = MinMaxScaler()
    df_sens['sensationalism_score'] = scaler.fit_transform(df_sens[['sensationalism_score']]).flatten()

    # 结果分析
    print(f"\n=== 耸人听闻程度评分结果 ===")
    print(f"分数范围：{df_sens['sensationalism_score'].min():.3f} - {df_sens['sensationalism_score'].max():.3f}")
    print(f"平均分数：{df_sens['sensationalism_score'].mean():.3f}")

    # 可视化（所有中文改为英文，消除字体警告）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.hist(df_sens['sensationalism_score'], bins=50, color='purple', alpha=0.7)
    ax1.set_title('Sensationalism Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Score (0=Low Sensationalism, 1=High)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)

    topic_sens_score = df_sens.groupby('lda_topic')['sensationalism_score'].mean().sort_values(ascending=False)
    topic_sens_score.plot(kind='bar', color='darkred', ax=ax2)
    ax2.set_title('Average Sensationalism Score by Topic', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Topic ID', fontsize=10)
    ax2.set_ylabel('Average Score', fontsize=10)
    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('sensationalism_scoring_results.png', dpi=150, bbox_inches='tight')
    print("📊 耸人听闻分数可视化已保存")

    # 保存结果
    df_sens.to_pickle('full_data_with_sensationalism_score.pkl')
    df_sens[['headline_text', 'lda_topic', 'sentiment_compound', 'sensationalism_score']].to_csv(
        'sensationalism_scoring_results.csv', index=False, encoding='utf-8'
    )


# ===================== 4. 新增：Topic-Level Sensationalism vs Fake News Correlation（提速版）=====================
def topic_sensationalism_fake_news_correlation(
    fake_news_csv='fake_news.csv',  # 假新闻CSV（仅含title列）
    real_news_csv='real_news.csv',  # 真实新闻CSV（仅含title列）
    threshold=65,  # 相似度阈值（0-100）
    top_n=3  # 每个标题只匹配前3个最相似的标注（减少计算量）
):
    """
    分析每个主题的耸人听闻程度与假新闻比例的相关性（提速版）
    核心优化：用rapidfuzz替代fuzzywuzzy，倒排索引缩小比对范围，批量匹配
    """
    print("\n" + "=" * 70)
    print("=== 主题级耸人听闻程度 vs 假新闻比例 相关性分析（提速版）===")
    print("=" * 70)

    # 1. 加载带耸人听闻分数的数据集（复用之前的计算结果）
    try:
        df_sens = pd.read_pickle('full_data_with_sensationalism_score.pkl')
        print("✅ 成功加载带耸人听闻分数的数据集")
    except FileNotFoundError:
        print("⚠️  未找到耸人听闻分数数据，正在重新计算...")
        df_sens = calculate_sensationalism_score()

    # 定义标题清洗函数（统一逻辑，保留关键词）
    def clean_title(title):
        if pd.isna(title):
            return ""
        # 清洗：去前后空格、小写、只保留字母/数字/空格（去除无意义符号）
        title = str(title).strip().lower()
        title = re.sub(r'[^\w\s]', '', title)  # 只保留单词和空格
        title = re.sub(r'\s+', ' ', title)     # 合并多个空格为一个
        return title

    # 2. 加载并处理假新闻+真实新闻数据
    def load_news_data(csv_path, is_fake_label):
        try:
            df = pd.read_csv(csv_path)
            title_cols = [col for col in df.columns if col.strip().lower() == 'title']
            if not title_cols:
                raise ValueError(f"CSV文件需包含'title'列，当前列名：{df.columns.tolist()}")
            df = df.rename(columns={title_cols[0]: 'headline_text'}).reset_index(drop=True)
            df['is_fake'] = is_fake_label
            df = df.drop_duplicates(subset=['headline_text']).reset_index(drop=True)
            df['headline_text_clean'] = df['headline_text'].apply(clean_title)
            # 过滤空标题（避免无效计算）
            df = df[df['headline_text_clean'] != ""].reset_index(drop=True)
            print(f"✅ 加载{'假新闻' if is_fake_label == 1 else '真实新闻'}数据成功，共 {len(df)} 条有效标题")
            return df[['headline_text', 'headline_text_clean', 'is_fake']]
        except Exception as e:
            raise Exception(f"{'假新闻' if is_fake_label == 1 else '真实新闻'}加载失败：{e}")

    try:
        fake_df = load_news_data(fake_news_csv, is_fake_label=1)
        real_df = load_news_data(real_news_csv, is_fake_label=0)
        fake_news_df = pd.concat([fake_df, real_df], ignore_index=True)
        print(f"✅ 合并后共 {len(fake_news_df)} 条有效标注（假新闻：{len(fake_df)} 条，真实新闻：{len(real_df)} 条）")
    except Exception as e:
        print(f"❌ 标注数据加载失败：{e}")
        return

    # 3. 主数据集清洗+去重
    df_sens['headline_text_clean'] = df_sens['headline_text'].apply(clean_title)
    df_sens = df_sens[df_sens['headline_text_clean'] != ""].reset_index(drop=True)
    df_sens = df_sens.drop_duplicates(subset=['headline_text_clean']).reset_index(drop=True)
    print(f"✅ 主数据集清洗完成：共 {len(df_sens)} 条去重后有效标题")

    # 4. 关键提速：安装并使用rapidfuzz（比fuzzywuzzy快10-100倍）
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        print("⚠️  未安装高效匹配库，正在自动安装（仅需一次）...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz"])
        from rapidfuzz import process, fuzz

    # 5. 核心优化：倒排索引（缩小比对范围，避免全量匹配）
    def build_inverted_index(text_list):
        """构建倒排索引：关键词→包含该关键词的文本索引"""
        inverted_index = {}
        for idx, text in enumerate(text_list):
            words = text.split()  # 按空格分词
            for word in words:
                if len(word) < 2:  # 过滤单字符关键词（无意义）
                    continue
                if word not in inverted_index:
                    inverted_index[word] = set()
                inverted_index[word].add(idx)
        return inverted_index

    # 为标注集构建倒排索引（基于清洗后的标题关键词）
    target_texts = fake_news_df['headline_text_clean'].tolist()
    target_is_fake = fake_news_df['is_fake'].tolist()
    inverted_index = build_inverted_index(target_texts)
    print(f"✅ 倒排索引构建完成：共 {len(inverted_index)} 个关键词")

    # 6. 批量模糊匹配（只在相关标注中比对，大幅提速）
    def batch_fuzzy_match(main_texts, target_texts, target_is_fake, inverted_index, threshold=80, top_n=3):
        print(f"🔍 批量模糊匹配（阈值：{threshold}，每个标题匹配前{top_n}个候选）...")
        match_is_fake = []
        batch_size = 1000  # 分批处理，避免内存占用过大

        for i in range(0, len(main_texts), batch_size):
            batch_texts = main_texts[i:i+batch_size]
            # 每批输出进度
            print(f"🔄 处理第 {i//batch_size + 1} 批（共 {len(main_texts)//batch_size + 1} 批）...")

            for text in batch_texts:
                if not text:
                    match_is_fake.append(np.nan)
                    continue

                # 步骤1：提取当前标题的关键词，找到相关标注的索引（缩小比对范围）
                words = text.split()
                related_indices = set()
                for word in words:
                    if len(word) < 2:
                        continue
                    if word in inverted_index:
                        related_indices.update(inverted_index[word])
                related_indices = list(related_indices)

                # 步骤2：如果无相关标注，标记为未匹配
                if not related_indices:
                    match_is_fake.append(np.nan)
                    continue

                # 步骤3：只在相关标注中匹配（核心提速点）
                related_targets = [target_texts[idx] for idx in related_indices]
                related_is_fake = [target_is_fake[idx] for idx in related_indices]

                # 步骤4：快速匹配（用rapidfuzz的process.extract，比fuzzywuzzy快10倍）
                matches = process.extract(
                    text,
                    related_targets,
                    scorer=fuzz.token_sort_ratio,  # 保持原有的匹配逻辑
                    limit=top_n,  # 只取前N个最相似的
                    score_cutoff=threshold  # 低于阈值的直接过滤
                )

                # 步骤5：取相似度最高的匹配结果
                if matches:
                    best_match = max(matches, key=lambda x: x[1])
                    best_idx = related_targets.index(best_match[0])
                    match_is_fake.append(related_is_fake[best_idx])
                else:
                    match_is_fake.append(np.nan)

        return match_is_fake

    # 执行批量匹配
    main_texts = df_sens['headline_text_clean'].tolist()
    df_sens['is_fake'] = batch_fuzzy_match(
        main_texts, target_texts, target_is_fake, inverted_index,
        threshold=threshold, top_n=top_n
    )

    # 过滤未匹配到的记录
    df_combined = df_sens[df_sens['is_fake'].notna()].reset_index(drop=True)
    print(f"✅ 匹配完成：共 {len(df_combined)} 条有效匹配记录")

    if len(df_combined) < 100:
        print("⚠️  匹配结果过少，建议降低阈值（如70）或减少关键词过滤（如保留单字符关键词）")
        return

    # 7. 后续主题级指标计算、相关性分析、可视化（逻辑不变）
    topic_metrics = df_combined.groupby('lda_topic').agg(
        主题新闻总数=('headline_text', 'count'),
        假新闻数量=('is_fake', 'sum'),
        平均耸人听闻分数=('sensationalism_score', 'mean')
    ).reset_index()

    topic_metrics['假新闻比例'] = topic_metrics['假新闻数量'] / topic_metrics['主题新闻总数'].replace(0, 1)
    topic_metrics = topic_metrics[topic_metrics['主题新闻总数'] >= 20].reset_index(drop=True)

    if len(topic_metrics) < 3:
        print("⚠️  有效主题数量过少（<3），无法进行相关性分析")
        return

    print(f"\n📊 主题级指标统计（过滤后）：")
    print(topic_metrics[['lda_topic', '主题新闻总数', '假新闻比例', '平均耸人听闻分数']].round(3))

    # 相关性分析
    x = topic_metrics['平均耸人听闻分数']
    y = topic_metrics['假新闻比例']
    corr_coef, p_value = pearsonr(x, y)

    print(f"\n📈 相关性分析结果：")
    print(f"皮尔逊相关系数（r）：{corr_coef:.3f}")
    print(f"显著性水平（p值）：{p_value:.3f}")
    if p_value < 0.05:
        significance = "显著（p<0.05）"
        interpretation = f"正相关：耸人听闻程度越高的主题，假新闻比例越高（r={corr_coef:.3f}）" if corr_coef > 0 else f"负相关：耸人听闻程度越高的主题，假新闻比例越低（r={corr_coef:.3f}）"
    else:
        significance = "不显著（p≥0.05）"
        interpretation = "未检测到显著的线性相关关系"
    print(f"结果解读：{interpretation}（{significance}）")

    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 散点图+拟合线
    ax1.scatter(x, y, color='darkred', alpha=0.7, s=60, label=f'r={corr_coef:.3f}, p={p_value:.3f}')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "b--", alpha=0.8, linewidth=2)
    ax1.set_xlabel('Average Sensationalism Score', fontsize=10)
    ax1.set_ylabel('Fake News Ratio', fontsize=10)
    ax1.set_title('Sensationalism Score vs Fake News Ratio\n(Scatter Plot with Trend Line)', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 双指标条形图
    topic_ids = topic_metrics['lda_topic'].astype(str)
    x_pos = np.arange(len(topic_ids))
    width = 0.35
    ax2.bar(x_pos - width/2, topic_metrics['平均耸人听闻分数'], width, label='Avg Sensationalism Score', color='purple', alpha=0.7)
    ax2.set_xlabel('Topic ID', fontsize=10)
    ax2.set_ylabel('Avg Sensationalism Score', fontsize=10, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(topic_ids)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x_pos + width/2, topic_metrics['假新闻比例'], width, label='Fake News Ratio', color='orange', alpha=0.7)
    ax2_twin.set_ylabel('Fake News Ratio', fontsize=10, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title('Sensationalism Score & Fake News Ratio by Topic', fontsize=11, fontweight='bold')

    # 热力图
    heatmap_data = topic_metrics[['平均耸人听闻分数', '假新闻比例', '主题新闻总数']].corr()
    sb.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, ax=ax3)
    ax3.set_title('Correlation Heatmap of Topic-Level Metrics', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('sensationalism_fake_news_correlation.png', dpi=150, bbox_inches='tight')
    print("\n📊 可视化已保存：sensationalism_fake_news_correlation.png")

    # 保存结果
    topic_metrics.to_csv('topic_sensationalism_fake_news_metrics.csv', index=False, encoding='utf-8')
    print("\n📄 详细指标已保存：topic_sensationalism_fake_news_metrics.csv")
# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 1. 生成标注模板（已完成标注，注释掉）
    # generate_annotation_template()

    # 2. 运行 VADER vs RoBERTa 模型对比（核心步骤）
    #compare_vader_roberta(annotated_csv='sentiment_annotation_template.csv')  # 标注文件路径确保正确

    # 3. 异常检测（直接运行）
    #topic_sentiment_anomaly_detection()

    # 4. 耸人听闻评分（直接运行）
    calculate_sensationalism_score()
    # 5. 新增：主题级耸人听闻程度 vs 假新闻比例 相关性分析（关键新增）
    #topic_sensationalism_fake_news_correlation(
    #    fake_news_csv='Fake.csv',  # 你的假新闻CSV文件名
    #    real_news_csv='True.csv'  # 你的真实新闻CSV文件名
    #)

    print("\n🎉 所有高级分析完成！")