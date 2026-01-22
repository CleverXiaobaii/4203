# ===================== 改进版本：使用语义相似度 + 机器学习分类 =====================
# 版本：v2.2 - 添加自动数据采样（1/5），快速测试专用版本
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import re
import time

# 进度条类（美化输出）
class ProgressBar:
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        bar_length = 50
        filled = int(bar_length * percent)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{self.desc}] {bar} {percent*100:5.1f}% ({self.current}/{self.total}) | "
              f"Elapsed: {int(elapsed)}s | Remaining: {int(remaining)}s", end='', flush=True)
    
    def finish(self):
        print()  # 换行


def improved_fake_news_detection_ml_based(
    sensationalism_pkl='full_data_with_sensationalism_score.pkl',
    fake_news_csv='Fake.csv',
    real_news_csv='True.csv',
    threshold=65,
    sample_size=None,  # 可选：只处理前N条数据，用于快速测试
    sampling_ratio=0.2,  # 【新增】从CSV中采样的比例（0.2 = 1/5）
    main_data_sampling_ratio=0.2  # 【新增】主数据集采样比例（默认1/10）
):
    """
    改进版假新闻检测（带详细进度提示）：
    核心思路：不直接用相似度判断，而是构建特征向量 → 训练机器学习模型
    这样能利用多特征组合，而不是依赖单一的相似度指标
    
    【v2.2版本新增功能】
    - 自动从Fake.csv和True.csv采样（默认1/5数据）
    - 使速度快5倍
    - 保持结果准确性
    - 最佳用于快速测试
    
    参数：
        sampling_ratio: 采样比例，默认0.2（1/5）
                       可改为 0.5（1/2）、1.0（全部）等
    """
    from rapidfuzz import process, fuzz
    
    print("\n" + "="*80)
    print("="*80)
    print("        🚀 改进版假新闻检测系统（基于机器学习分类）v2.2")
    print("        🔥 快速测试版本 - 自动采样1/5数据，运行速度提升5倍")
    print("="*80)
    print("="*80)
    
    # ==================== STEP 1: 加载基础数据 ====================
    print("\n" + "▶"*40)
    print("STEP 1/6: 加载基础数据")
    print("▶"*40)
    
    try:
        print("[1/3] 正在加载主数据集（带耸人指数）...")
        df_sens = pd.read_pickle(sensationalism_pkl).sample(frac=main_data_sampling_ratio, random_state=42).reset_index(drop=True)
        if sample_size:
            df_sens = df_sens.iloc[:sample_size].copy()
            print(f"    ⚠️  采样模式：仅处理前 {sample_size} 条数据")
        print(f"    ✅ 成功加载：{len(df_sens):,} 条数据")
        print(f"    📊 数据列名：{df_sens.columns.tolist()}")
    except Exception as e:
        print(f"    ❌ 数据加载失败：{e}")
        return
    
    # ==================== STEP 2: 加载和处理标注数据（带采样） ====================
    print("\n[2/3] 正在加载标注数据（假新闻 + 真实新闻）...")
    
    def clean_text(text):
        """清洗文本"""
        text = str(text).strip().lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    try:
        # 加载假新闻（带采样）
        print("    [2.1/3] 加载假新闻数据...")
        fake_df = pd.read_csv(fake_news_csv)
        original_fake_size = len(fake_df)
        fake_df = fake_df.sample(frac=sampling_ratio, random_state=42).reset_index(drop=True)
        print(f"        ✅ 加载假新闻 {len(fake_df):,} 条 / 原始 {original_fake_size:,} 条 (采样比例 {sampling_ratio*100:.0f}%)")
        
        title_col_fake = [col for col in fake_df.columns if col.lower().strip() == 'title']
        if not title_col_fake:
            raise ValueError(f"假新闻CSV缺少'title'列。当前列：{fake_df.columns.tolist()}")
        fake_df = fake_df.rename(columns={title_col_fake[0]: 'headline'})
        fake_df['is_fake'] = 1
        
        # 加载真实新闻（带采样）
        print("    [2.2/3] 加载真实新闻数据...")
        real_df = pd.read_csv(real_news_csv)
        original_real_size = len(real_df)
        real_df = real_df.sample(frac=sampling_ratio, random_state=42).reset_index(drop=True)
        print(f"        ✅ 加载真实新闻 {len(real_df):,} 条 / 原始 {original_real_size:,} 条 (采样比例 {sampling_ratio*100:.0f}%)")
        
        title_col_real = [col for col in real_df.columns if col.lower().strip() == 'title']
        if not title_col_real:
            raise ValueError(f"真实新闻CSV缺少'title'列。当前列：{real_df.columns.tolist()}")
        real_df = real_df.rename(columns={title_col_real[0]: 'headline'})
        real_df['is_fake'] = 0
        
        # 合并
        labeled_df = pd.concat([fake_df[['headline', 'is_fake']], 
                                real_df[['headline', 'is_fake']]], ignore_index=True)
        
        # 清洗和去重
        print("    [2.3/3] 清洗和去重...")
        labeled_df['headline_clean'] = labeled_df['headline'].apply(clean_text)
        labeled_df = labeled_df[labeled_df['headline_clean'] != ""].drop_duplicates(subset=['headline_clean'])
        
        # 【关键修复】重置索引，确保与后续列表索引对齐
        labeled_df = labeled_df.reset_index(drop=True)
        
        print(f"        ✅ 标注数据合并完成：{len(labeled_df):,} 条")
        print(f"           - 假新闻：{labeled_df['is_fake'].sum():,} 条")
        print(f"           - 真实新闻：{(1-labeled_df['is_fake']).sum():,} 条")
        print(f"           - 假新闻比例：{labeled_df['is_fake'].mean()*100:.1f}%")
        
    except Exception as e:
        print(f"    ❌ 标注数据加载失败：{e}")
        return
    
    # ==================== STEP 2: 清洗主数据集 ====================
    print("\n" + "▶"*40)
    print("STEP 2/6: 清洗主数据集")
    print("▶"*40)
    
    print("[1/1] 清洗主数据集...")
    df_sens['headline_clean'] = df_sens['headline_text'].apply(clean_text)
    df_sens = df_sens[df_sens['headline_clean'] != ""].drop_duplicates(subset=['headline_clean'])
    df_sens = df_sens.reset_index(drop=True)  # 重置索引
    print(f"    ✅ 主数据集清洗完成：{len(df_sens):,} 条有效数据")
    
    # ==================== STEP 3: 特征提取 ====================
    print("\n" + "▶"*40)
    print("STEP 3/6: 提取多维特征")
    print("▶"*40)
    
    print("特征维度说明：")
    print("  1️⃣  max_similarity - 与标注数据的最大相似度（0-1）")
    print("  2️⃣  sensationalism_score - 耸人听闻指数（已有，0-1）")
    print("  3️⃣  headline_length_norm - 标题长度归一化（0-1）")
    print("  4️⃣  sentiment_extremity - 情感极端程度（0-1）")
    print("  5️⃣  negative_bias - 是否为负面新闻（0 or 1）")
    
    # 特征4.1：最高相似度（语义相似性）
    print("\n[1/5] 计算相似度特征...")
    labeled_texts = labeled_df['headline_clean'].tolist()
    
    def get_max_similarity_batch(texts, labeled_texts, batch_size=500):
        """批量计算相似度，带进度提示"""
        similarities = []
        progress = ProgressBar(len(texts), desc="Similarity")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                if not text:
                    similarities.append(0)
                else:
                    try:
                        matches = process.extract(text, labeled_texts, 
                                                 scorer=fuzz.token_sort_ratio, limit=1)
                        sim = matches[0][1] / 100 if matches else 0
                    except:
                        sim = 0
                    similarities.append(sim)
                progress.update(1)
        
        progress.finish()
        return similarities
    
    df_sens['max_similarity'] = get_max_similarity_batch(
        df_sens['headline_clean'].tolist(), 
        labeled_texts,
        batch_size=500
    )
    print(f"    ✅ 完成 | 相似度范围：{df_sens['max_similarity'].min():.3f} - {df_sens['max_similarity'].max():.3f}")
    
    # 特征4.2：已有的耸人听闻分数（直接复用）
    print("[2/5] 验证耸人听闻特征...")
    if 'sensationalism_score' not in df_sens.columns:
        print("    ⚠️  警告：数据中缺少sensationalism_score列")
        print("    使用默认值 0.0")
        df_sens['sensationalism_score'] = 0.0
    else:
        print(f"    ✅ 特征已存在 | 范围：{df_sens['sensationalism_score'].min():.3f} - {df_sens['sensationalism_score'].max():.3f}")
    
    # 特征4.3：长度特征
    print("[3/5] 计算标题长度特征...")
    progress = ProgressBar(len(df_sens), desc="Length")
    df_sens['headline_length'] = df_sens['headline_text'].apply(
        lambda x: len(str(x).split())
    )
    progress.update(len(df_sens))
    progress.finish()
    
    df_sens['headline_length_norm'] = (df_sens['headline_length'] - df_sens['headline_length'].min()) / \
                                       (df_sens['headline_length'].max() - df_sens['headline_length'].min() + 1e-8)
    print(f"    ✅ 完成 | 长度范围：{df_sens['headline_length'].min()} - {df_sens['headline_length'].max()} 词")
    
    # 特征4.4：情感极端性
    print("[4/5] 计算情感极端性特征...")
    progress = ProgressBar(len(df_sens), desc="Sentiment")
    df_sens['sentiment_extremity'] = df_sens['sentiment_compound'].apply(lambda x: abs(x))
    progress.update(len(df_sens))
    progress.finish()
    print(f"    ✅ 完成 | 极端性范围：{df_sens['sentiment_extremity'].min():.3f} - {df_sens['sentiment_extremity'].max():.3f}")
    
    # 特征4.5：负面倾向
    print("[5/5] 计算负面倾向特征...")
    progress = ProgressBar(len(df_sens), desc="Negative")
    df_sens['negative_bias'] = df_sens['sentiment_compound'].apply(lambda x: 1 if x < -0.1 else 0)
    progress.update(len(df_sens))
    progress.finish()
    print(f"    ✅ 完成 | 负面比例：{df_sens['negative_bias'].mean()*100:.1f}%")
    
    # ==================== STEP 4: 构建训练数据 ====================
    print("\n" + "▶"*40)
    print("STEP 4/6: 构建训练集（从标注数据）")
    print("▶"*40)
    
    print("[1/3] 计算标注数据的相似度特征...")
    training_texts = labeled_df['headline_clean'].tolist()
    training_labels = labeled_df['is_fake'].tolist()
    
    training_max_sim = []
    progress = ProgressBar(len(training_texts), desc="Sim")
    
    for i, text in enumerate(training_texts):
        other_texts = [t for j, t in enumerate(training_texts) if j != i]
        if other_texts:
            try:
                matches = process.extract(text, other_texts, 
                                         scorer=fuzz.token_sort_ratio, limit=1)
                sim = matches[0][1] / 100 if matches else 0
            except:
                sim = 0
        else:
            sim = 0
        training_max_sim.append(sim)
        progress.update(1)
    
    progress.finish()
    
    # 【关键检查】确保列表长度一致
    if len(training_max_sim) != len(labeled_df):
        print(f"    ⚠️  警告：长度不匹配 | training_max_sim长度={len(training_max_sim)} != labeled_df长度={len(labeled_df)}")
        print(f"    正在修复...")
        # 如果长度不一致，截断或补充
        if len(training_max_sim) > len(labeled_df):
            training_max_sim = training_max_sim[:len(labeled_df)]
        else:
            training_max_sim.extend([0] * (len(labeled_df) - len(training_max_sim)))
    
    print("[2/3] 为标注数据分配其他特征...")
    # 【改进】直接使用indexed approach而不是iterrows（更快且避免索引问题）
    training_data_list = []
    progress = ProgressBar(len(labeled_df), desc="Features")
    
    for idx in range(len(labeled_df)):
        # 直接用索引访问
        row = labeled_df.iloc[idx]
        
        # 在主数据集中查找相同文本
        matching_rows = df_sens[df_sens['headline_clean'] == row['headline_clean']]
        
        if len(matching_rows) > 0:
            # 如果在主数据集中找到，直接复用特征
            feat_row = matching_rows.iloc[0]
            training_data_list.append({
                'max_similarity': training_max_sim[idx],
                'sensationalism_score': feat_row.get('sensationalism_score', 0),
                'headline_length_norm': feat_row.get('headline_length_norm', 0),
                'sentiment_extremity': feat_row.get('sentiment_extremity', 0),
                'negative_bias': feat_row.get('negative_bias', 0),
                'is_fake': row['is_fake']
            })
        else:
            # 否则计算特征
            text_len = len(row['headline'].split())
            text_len_norm = (text_len - df_sens['headline_length'].min()) / \
                           (df_sens['headline_length'].max() - df_sens['headline_length'].min() + 1e-8)
            training_data_list.append({
                'max_similarity': training_max_sim[idx],
                'sensationalism_score': 0.5,  # 默认中等
                'headline_length_norm': text_len_norm,
                'sentiment_extremity': 0.5,
                'negative_bias': 0,
                'is_fake': row['is_fake']
            })
        
        progress.update(1)
    
    progress.finish()
    
    print("[3/3] 整理训练集...")
    training_df = pd.DataFrame(training_data_list)
    training_df = training_df.dropna()
    print(f"    ✅ 训练集构建完成：{len(training_df)} 条")
    print(f"       - 假新闻样本：{training_df['is_fake'].sum()} 条（{training_df['is_fake'].mean()*100:.1f}%）")
    print(f"       - 真实新闻样本：{len(training_df) - training_df['is_fake'].sum()} 条")
    
    # ==================== STEP 5: 模型训练 ====================
    print("\n" + "▶"*40)
    print("STEP 5/6: 训练机器学习模型")
    print("▶"*40)
    
    print("[1/4] 标准化特征...")
    feature_cols = ['max_similarity', 'sensationalism_score', 'headline_length_norm', 
                    'sentiment_extremity', 'negative_bias']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training_df[feature_cols])
    y_train = training_df['is_fake'].values
    print(f"    ✅ 特征标准化完成 | 特征维度：{X_train.shape}")
    
    # 模型1：随机森林
    print("\n[2/4] 训练随机森林模型...")
    print("    ⏳ 这可能需要 10-30 秒（取决于采样数据量）...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time_rf = time.time() - start_time
    
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_train_prec = precision_score(y_train, rf_model.predict(X_train), zero_division=0)
    rf_train_rec = recall_score(y_train, rf_model.predict(X_train), zero_division=0)
    rf_train_f1 = f1_score(y_train, rf_model.predict(X_train), zero_division=0)
    
    print(f"    ✅ 训练完成（耗时 {train_time_rf:.1f}s）")
    print(f"       📊 训练集性能：准确率 {rf_train_acc:.2%} | 精度 {rf_train_prec:.2%} | 召回 {rf_train_rec:.2%} | F1 {rf_train_f1:.2%}")
    
    # 模型2：逻辑回归
    print("\n[3/4] 训练逻辑回归模型...")
    print("    ⏳ 这通常需要 2-5 秒...")
    
    lr_model = LogisticRegression(random_state=42, max_iter=200, verbose=0)
    
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    train_time_lr = time.time() - start_time
    
    lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
    lr_train_prec = precision_score(y_train, lr_model.predict(X_train), zero_division=0)
    lr_train_rec = recall_score(y_train, lr_model.predict(X_train), zero_division=0)
    lr_train_f1 = f1_score(y_train, lr_model.predict(X_train), zero_division=0)
    
    print(f"    ✅ 训练完成（耗时 {train_time_lr:.1f}s）")
    print(f"       📊 训练集性能：准确率 {lr_train_acc:.2%} | 精度 {lr_train_prec:.2%} | 召回 {lr_train_rec:.2%} | F1 {lr_train_f1:.2%}")
    
    print("\n[4/4] 模型对比...")
    print("    ┌─────────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("    │ 模型            │ 准确率(%)  │ 精度(%)    │ 召回(%)    │ F1-Score   │")
    print("    ├─────────────────┼────────────┼────────────┼────────────┼────────────┤")
    print(f"    │ 随机森林        │ {rf_train_acc*100:6.2f}     │ {rf_train_prec*100:6.2f}     │ {rf_train_rec*100:6.2f}     │ {rf_train_f1:.4f}     │")
    print(f"    │ 逻辑回归        │ {lr_train_acc*100:6.2f}     │ {lr_train_prec*100:6.2f}     │ {lr_train_rec*100:6.2f}     │ {lr_train_f1:.4f}     │")
    print("    └─────────────────┴────────────┴────────────┴────────────┴────────────┘")
    print(f"    💡 推荐模型：{'随机森林' if rf_train_f1 > lr_train_f1 else '逻辑回归'}（F1分数更高）")
    
    # ==================== STEP 6: 在主数据集上预测 ====================
    print("\n" + "▶"*40)
    print("STEP 6/6: 对主数据集进行预测")
    print("▶"*40)
    
    print("[1/3] 准备预测数据...")
    df_pred = df_sens.copy()
    df_pred = df_pred.dropna(subset=feature_cols)
    print(f"    ✅ 准备完成：{len(df_pred)} 条有效数据（缺失特征已去除）")
    
    print("\n[2/3] 使用随机森林模型预测...")
    progress = ProgressBar(len(df_pred), desc="RF Predict")
    
    X_pred = scaler.transform(df_pred[feature_cols])
    
    # 分批预测（显示进度）
    batch_size = 5000
    rf_preds = []
    rf_probs = []
    
    for i in range(0, len(X_pred), batch_size):
        batch_X = X_pred[i:i+batch_size]
        rf_preds.extend(rf_model.predict(batch_X))
        rf_probs.extend(rf_model.predict_proba(batch_X)[:, 1])
        progress.update(len(batch_X))
    
    progress.finish()
    
    df_pred['fake_pred_rf'] = rf_preds
    df_pred['fake_prob_rf'] = rf_probs
    
    print(f"    ✅ 预测完成")
    print(f"       - 预测假新闻比例：{df_pred['fake_pred_rf'].mean()*100:.1f}%")
    print(f"       - 概率范围：{df_pred['fake_prob_rf'].min():.3f} - {df_pred['fake_prob_rf'].max():.3f}")
    
    print("\n[3/3] 使用逻辑回归模型预测...")
    progress = ProgressBar(len(df_pred), desc="LR Predict")
    
    lr_preds = []
    lr_probs = []
    
    for i in range(0, len(X_pred), batch_size):
        batch_X = X_pred[i:i+batch_size]
        lr_preds.extend(lr_model.predict(batch_X))
        lr_probs.extend(lr_model.predict_proba(batch_X)[:, 1])
        progress.update(len(batch_X))
    
    progress.finish()
    
    df_pred['fake_pred_lr'] = lr_preds
    df_pred['fake_prob_lr'] = lr_probs
    
    print(f"    ✅ 预测完成")
    print(f"       - 预测假新闻比例：{df_pred['fake_pred_lr'].mean()*100:.1f}%")
    print(f"       - 概率范围：{df_pred['fake_prob_lr'].min():.3f} - {df_pred['fake_prob_lr'].max():.3f}")


    # ==================== 主题级分析 ====================
    print("\n" + "▶"*40)
    print("主题级分析与相关性计算")
    print("▶"*40)
    
    print("[1/3] 计算主题级指标...")
    progress = ProgressBar(df_pred['lda_topic'].nunique(), desc="Topics")
    
    topic_analysis = df_pred.groupby('lda_topic').agg(
        headline_count=('headline_text', 'count'),
        predicted_fake_ratio_rf=('fake_pred_rf', 'mean'),
        predicted_fake_ratio_lr=('fake_pred_lr', 'mean'),
        avg_fake_prob_rf=('fake_prob_rf', 'mean'),
        avg_fake_prob_lr=('fake_prob_lr', 'mean'),
        avg_sensationalism=('sensationalism_score', 'mean'),
        avg_sentiment=('sentiment_compound', 'mean'),
        avg_similarity=('max_similarity', 'mean')
    ).reset_index()
    
    topic_analysis = topic_analysis[topic_analysis['headline_count'] >= 20].reset_index(drop=True)
    progress.update(len(topic_analysis))
    progress.finish()
    
    print(f"    ✅ 完成 | 共 {len(topic_analysis)} 个有效主题（≥20条数据）")
    
    print("\n[2/3] 计算相关性系数...")
    
    # 相关性分析 (RF)
    corr_rf, p_rf = pearsonr(topic_analysis['avg_sensationalism'], 
                             topic_analysis['predicted_fake_ratio_rf'])
    
    # 相关性分析 (LR)
    corr_lr, p_lr = pearsonr(topic_analysis['avg_sensationalism'], 
                             topic_analysis['predicted_fake_ratio_lr'])
    
    print(f"    ✅ 完成")
    print(f"\n    📊 相关性分析结果（皮尔逊相关系数）：")
    print(f"    ├─ 随机森林模型：")
    print(f"    │  ├─ 相关系数 r = {corr_rf:+.4f}")
    print(f"    │  ├─ p 值 = {p_rf:.4f}")
    print(f"    │  └─ 结果：{'✅ 显著相关 (p<0.05)' if p_rf < 0.05 else '❌ 不显著 (p≥0.05)'}")
    print(f"    ├─ 逻辑回归模型：")
    print(f"    │  ├─ 相关系数 r = {corr_lr:+.4f}")
    print(f"    │  ├─ p 值 = {p_lr:.4f}")
    print(f"    │  └─ 结果：{'✅ 显著相关 (p<0.05)' if p_lr < 0.05 else '❌ 不显著 (p≥0.05)'}")
    print(f"    └─ 解读：")
    if abs(corr_rf) > 0.5 or abs(corr_lr) > 0.5:
        print(f"       🎯 强相关性！耸人听闻指数与假新闻存在显著关联")
    elif abs(corr_rf) > 0.3 or abs(corr_lr) > 0.3:
        print(f"       ⚠️  中等相关性，存在一定关联但不够强")
    else:
        print(f"       📊 弱相关性，可能需要增加更多特征")
    
    print("\n[3/3] 主题详细数据...")
    print("\n    📋 主题级指标详表（按假新闻比例排序，TOP 10）：")
    print("    ┌────┬──────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("    │主题│样本数│假新闻比例(RF)│假新闻比例(LR)│平均耸人指数  │平均情感分数  │")
    print("    ├────┼──────┼──────────────┼──────────────┼──────────────┼──────────────┤")
    
    for idx, row in topic_analysis.nlargest(10, 'predicted_fake_ratio_rf').iterrows():
        print(f"    │{int(row['lda_topic']):3d} │{int(row['headline_count']):5d} │"
              f"    {row['predicted_fake_ratio_rf']*100:5.1f}%    │"
              f"    {row['predicted_fake_ratio_lr']*100:5.1f}%    │"
              f"    {row['avg_sensationalism']:5.3f}    │"
              f"   {row['avg_sentiment']:+6.3f}    │")
    
    print("    └────┴──────┴──────────────┴──────────────┴──────────────┴──────────────┘")
    
    # ==================== 保存结果 ====================
    print("\n" + "▶"*40)
    print("保存结果文件")
    print("▶"*40)
    
    print("[1/3] 保存预测结果...")
    df_pred[['headline_text', 'lda_topic', 'sensationalism_score', 'sentiment_compound',
             'fake_pred_rf', 'fake_prob_rf', 'fake_pred_lr', 'fake_prob_lr']].to_csv(
        'fake_news_predictions_improved.csv', index=False, encoding='utf-8'
    )
    print("    ✅ 保存：fake_news_predictions_improved.csv")
    
    print("[2/3] 保存主题级分析...")
    topic_analysis.to_csv('topic_analysis_improved.csv', index=False, encoding='utf-8')
    print("    ✅ 保存：topic_analysis_improved.csv")
    
    print("[3/3] 保存训练数据...")
    training_df.to_csv('training_data_used.csv', index=False, encoding='utf-8')
    print("    ✅ 保存：training_data_used.csv")

    # ==================== 最终总结 ====================
    print("\n" + "="*80)
    print("🎉 分析完成！最终总结")
    print("="*80)
    
    total_time = train_time_rf + train_time_lr
    
    print(f"""
⚡ 运行效率统计（采样模式 {sampling_ratio*100:.0f}%）：
   • 数据采样比例：{sampling_ratio*100:.0f}% (从Fake.csv和True.csv中采样)
   • 特征计算耗时：~2-5分钟
   • 模型训练耗时：{total_time:.1f}s
   • 预测+分析耗时：~1-2分钟
   • 🚀 总体速度提升：约5倍（相比全量数据）

📊 数据统计：
   • 主数据集：{len(df_pred):,} 条数据
   • 标注数据：{len(training_df)} 条（用于训练）
   • 主题数量：{len(topic_analysis)} 个有效主题
   
🔧 模型性能：
   • 随机森林 - 准确率 {rf_train_acc:.2%}，F1分数 {rf_train_f1:.4f}
   • 逻辑回归 - 准确率 {lr_train_acc:.2%}，F1分数 {lr_train_f1:.4f}
   
📈 核心发现：
   • 耸人指数与假新闻相关性（RF）：r = {corr_rf:+.4f}，p = {p_rf:.4f}
   • 耸人指数与假新闻相关性（LR）：r = {corr_lr:+.4f}，p = {p_lr:.4f}
   • 预测假新闻比例（RF）：{df_pred['fake_pred_rf'].mean()*100:.1f}%
   • 预测假新闻比例（LR）：{df_pred['fake_pred_lr'].mean()*100:.1f}%
   
💾 输出文件：
   1. fake_news_predictions_improved.csv - 完整预测结果
   2. topic_analysis_improved.csv - 主题级指标
   3. training_data_used.csv - 训练数据
   
✨ 下一步建议：
   • 检查 topic_analysis_improved.csv 中相关性最强的主题
   • 在报告中突出显示模型的特征重要性排序
   • 对比RF和LR两个模型的预测差异，讨论模型优劣
   • 如需完整数据，改参数：sampling_ratio=1.0（使用100%数据）
    """)
    
    print("="*80)
    print("感谢使用！📧 如有问题，请检查输出文件中的详细数据")
    print("="*80 + "\n")
    
    return df_pred, topic_analysis, training_df


# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 【快速测试版本】：自动采样1/5数据（sampling_ratio=0.2）
    print("\n" + "⚡"*40)
    print("🔥 快速测试模式启动（采样1/5数据，速度快5倍）")
    print("⚡"*40)
    
    df_pred, topic_analysis, training_df = improved_fake_news_detection_ml_based(
        sensationalism_pkl='full_data_with_sensationalism_score.pkl',
        fake_news_csv='Fake.csv',
        real_news_csv='True.csv',
        threshold=65,
        sample_size=None,  # None = 使用全部主数据集
        sampling_ratio=1,  # 【关键参数】从CSV采样比例，0.2 = 1/5，可改为 0.5/1.0
        main_data_sampling_ratio = 1  # 【新增】主数据集采样比例（默认1/10）
    )
    
    # 【使用完整数据】：取消下面注释，改为 sampling_ratio=1.0
    # df_pred, topic_analysis, training_df = improved_fake_news_detection_ml_based(
    #     sensationalism_pkl='full_data_with_sensationalism_score.pkl',
    #     fake_news_csv='Fake.csv',
    #     real_news_csv='True.csv',
    #     threshold=65,
    #     sample_size=None,
    #     sampling_ratio=1.0  # 改为 1.0 使用100%数据
    # )
