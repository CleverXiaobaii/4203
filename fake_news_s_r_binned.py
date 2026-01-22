# ===================== 假新闻率 r 与耸人听闻指数 s 的数学关系拟合 =====================
# 按 s 的 0.1 区间分组，拟合三种数学模型（线性、对数、指数）
# 包含相关系数计算和完整可视化
# 版本：v1.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

np.random.seed(42)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.style.use('seaborn-v0_8-whitegrid')


# ===================== 1. 定义拟合函数 =====================

# 线性模型：r = a * s + b
def linear_model(s, a, b):
    return a * s + b

# 对数模型：r = a * log(b * s + c) + d
def log_model(s, a, b, c, d):
    return a * np.log(b * s + c) + d

# 指数模型：r = a * exp(b * s + c) + d
def exp_model(s, a, b, c, d):
    return a * np.exp(b * s + c) + d


def analyze_s_r_relation():
    """
    按 s 区间分组分析假新闻率 r 与耸人听闻指数 s 的关系
    """
    
    print("\n" + "=" * 90)
    print("📊 假新闻率 r 与耸人听闻指数 s 的数学关系分析")
    print("   按 0.1 区间分组 | 拟合三种数学模型 | 计算相关系数")
    print("=" * 90)

    # ===================== 2. 读取数据 =====================
    
    try:
        # 尝试读取实际数据文件
        df_pred = pd.read_csv('fake_news_predictions_improved.csv')
        s_all = df_pred['sensationalism_score'].values.astype(float)
        fake_pred_rf = df_pred['fake_pred_rf'].values.astype(float)
        
        print(f"\n✅ 加载实际数据：{len(df_pred):,} 条新闻")
        
    except FileNotFoundError:
        # 如果文件不存在，使用演示数据
        print("\n⚠️  未找到数据文件，使用模拟数据演示...")
        
        n_samples = 50000
        s_all = np.random.beta(2, 5, n_samples)
        r_true = 20 - 15 * s_all + 5 * s_all**2
        r_true = np.clip(r_true, 0, 100)
        noise = np.random.normal(0, 8, n_samples)
        fake_prob = np.clip(r_true + noise, 0, 100) / 100.0
        fake_pred_rf = (np.random.random(n_samples) < fake_prob).astype(int)
        
        print(f"\n✅ 生成模拟数据：{n_samples:,} 条新闻")
    
    # 清洗 NaN
    mask = ~np.isnan(s_all) & ~np.isnan(fake_pred_rf)
    s_all = s_all[mask]
    fake_pred_rf = fake_pred_rf[mask]
    
    print(f"   s 范围：[{s_all.min():.4f}, {s_all.max():.4f}]")
    print(f"   假新闻总比例：{np.mean(fake_pred_rf)*100:.2f}%")

    # ===================== 3. 按 s 区间分组 =====================
    
    print(f"\n【分组设置】")
    bin_width = 0.05
    bin_start = int(np.floor(s_all.min() / bin_width)) * bin_width
    bin_end = int(np.ceil(s_all.max() / bin_width)) * bin_width
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
    
    print(f"   范围：[{bin_start:.2f}, {bin_end:.2f}]，步长：{bin_width}")
    print(f"   总共 {len(bins)-1} 个区间")
    
    bin_indices = np.digitize(s_all, bins) - 1
    
    # 计算每个 bin 的统计
    bin_data = []
    for i in range(len(bins) - 1):
        mask_bin = (bin_indices == i)
        if np.sum(mask_bin) > 0:
            s_mid = (bins[i] + bins[i+1]) / 2.0
            r_bin = np.mean(fake_pred_rf[mask_bin]) * 100.0
            count_bin = np.sum(mask_bin)
            
            bin_data.append({
                's_bin': s_mid,
                'r_bin': r_bin,
                'count': count_bin,
                'bin_left': bins[i],
                'bin_right': bins[i+1]
            })
    
    bin_df = pd.DataFrame(bin_data)
    s = bin_df['s_bin'].values
    r = bin_df['r_bin'].values
    
    print(f"\n✅ 分组完成：{len(bin_df)} 个非空区间\n")
    
    # 打印分组统计表
    result_table = bin_df[['bin_left', 'bin_right', 's_bin', 'r_bin', 'count']].copy()
    result_table.columns = ['Bin左端点', 'Bin右端点', 's区间中点', 'r假新闻率%', '样本数']
    result_table['样本占比%'] = (result_table['样本数'] / result_table['样本数'].sum() * 100).round(2)
    
    print("=" * 100)
    print("【分组统计表】按 s 的 0.1 区间分组")
    print("=" * 100)
    print(result_table.to_string(index=False))
    print("=" * 100)

    # ===================== 4. 相关系数计算 =====================
    
    pearson_r, p_value = pearsonr(s, r)
    
    print(f"\n【Pearson 相关系数】")
    print(f"   r = {pearson_r:.4f}")
    print(f"   p-value = {p_value:.4f}")
    print(f"   结论 = {'✓ 显著相关 (p < 0.05)' if p_value < 0.05 else '✗ 不显著 (p ≥ 0.05)'}")

    # ===================== 5. 三种模型拟合 =====================
    
    print(f"\n【模型拟合】")
    
    results = []
    
    # 5.1 线性模型
    try:
        popt_lin, _ = curve_fit(linear_model, s, r)
        a_lin, b_lin = popt_lin
        r_pred_lin = linear_model(s, a_lin, b_lin)
        r2_lin = 1 - np.sum((r - r_pred_lin) ** 2) / np.sum((r - r.mean()) ** 2)
        rmse_lin = np.sqrt(np.mean((r - r_pred_lin) ** 2))
        lin_ok = True
        
        print(f"\n✓ 线性模型拟合成功")
        print(f"   公式：r = {a_lin:.6f}·s + {b_lin:.6f}")
        print(f"   R² = {r2_lin:.4f}，RMSE = {rmse_lin:.4f}")
        
        results.append({
            '模型': '线性',
            '公式': f'r = {a_lin:.6f}·s + {b_lin:.6f}',
            'R²': f'{r2_lin:.4f}',
            'RMSE': f'{rmse_lin:.4f}'
        })
    except Exception as e:
        print(f"\n✗ 线性模型拟合失败：{e}")
        lin_ok = False

    # 5.2 对数模型
    a0, b0, c0, d0 = 1.0, 1.0, 1e-3, r.mean()
    bounds_log = ([-np.inf, 1e-6, 1e-6, -np.inf], [ np.inf,  np.inf,  np.inf,  np.inf])
    try:
        popt_log, _ = curve_fit(log_model, s, r, p0=[a0, b0, c0, d0], bounds=bounds_log, maxfev=10000)
        a_log, b_log, c_log, d_log = popt_log
        r_pred_log = log_model(s, a_log, b_log, c_log, d_log)
        r2_log = 1 - np.sum((r - r_pred_log) ** 2) / np.sum((r - r.mean()) ** 2)
        rmse_log = np.sqrt(np.mean((r - r_pred_log) ** 2))
        log_ok = True
        
        print(f"\n✓ 对数模型拟合成功")
        print(f"   公式：r = {a_log:.6f}·ln({b_log:.6f}·s + {c_log:.6f}) + {d_log:.6f}")
        print(f"   R² = {r2_log:.4f}，RMSE = {rmse_log:.4f}")
        
        results.append({
            '模型': '对数',
            '公式': f'r = {a_log:.6f}·ln({b_log:.6f}·s + {c_log:.6f}) + {d_log:.6f}',
            'R²': f'{r2_log:.4f}',
            'RMSE': f'{rmse_log:.4f}'
        })
    except Exception as e:
        print(f"\n✗ 对数模型拟合失败：{e}")
        log_ok = False

    # 5.3 指数模型
    try:
        popt_exp, _ = curve_fit(exp_model, s, r, p0=[1.0, 1.0, 0.0, r.mean()], maxfev=10000)
        a_exp, b_exp, c_exp, d_exp = popt_exp
        r_pred_exp = exp_model(s, a_exp, b_exp, c_exp, d_exp)
        r2_exp = 1 - np.sum((r - r_pred_exp) ** 2) / np.sum((r - r.mean()) ** 2)
        rmse_exp = np.sqrt(np.mean((r - r_pred_exp) ** 2))
        exp_ok = True
        
        print(f"\n✓ 指数模型拟合成功")
        print(f"   公式：r = {a_exp:.6f}·exp({b_exp:.6f}·s + {c_exp:.6f}) + {d_exp:.6f}")
        print(f"   R² = {r2_exp:.4f}，RMSE = {rmse_exp:.4f}")
        
        results.append({
            '模型': '指数',
            '公式': f'r = {a_exp:.6f}·exp({b_exp:.6f}·s + {c_exp:.6f}) + {d_exp:.6f}',
            'R²': f'{r2_exp:.4f}',
            'RMSE': f'{rmse_exp:.4f}'
        })
    except Exception as e:
        print(f"\n✗ 指数模型拟合失败：{e}")
        exp_ok = False

    # 打印模型对比表
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 140)
    print("【三种模型拟合结果对比】")
    print("=" * 140)
    print(results_df.to_string(index=False))
    print("=" * 140)
    
    # 找最佳模型
    if lin_ok and log_ok and exp_ok:
        r2_values = [r2_lin, r2_log, r2_exp]
        best_idx = np.argmax(r2_values)
        models = ['线性', '对数', '指数']
        best_r2 = r2_values[best_idx]
        print(f"\n【最佳拟合模型】：{models[best_idx]} (R² = {best_r2:.4f})\n")

    # ===================== 6. 绘制图表 =====================
    
    fig, ax = plt.subplots(figsize=(13, 8))

    # 散点（点大小按样本量）
    scatter = ax.scatter(
        bin_df['s_bin'], bin_df['r_bin'],
        #s=bin_df['count']/10,
        color='#34495E',
        edgecolor='black',
        alpha=0.7,
        label=f'Binned Data',
        zorder=3
    )

    s_line = np.linspace(s.min() - 0.02, s.max() + 0.02, 300)

    # 绘制拟合曲线
    if lin_ok:
        ax.plot(
            s_line,
            linear_model(s_line, a_lin, b_lin),
            color='#E74C3C',
            linewidth=2.5,
            label=f'Linear: R²={r2_lin:.4f}',
            zorder=2
        )

    if log_ok:
        ax.plot(
            s_line,
            log_model(s_line, a_log, b_log, c_log, d_log),
            color='#3498DB',
            linewidth=2.5,
            linestyle='--',
            label=f'Log: R²={r2_log:.4f}',
            zorder=2
        )

    if exp_ok:
        ax.plot(
            s_line,
            exp_model(s_line, a_exp, b_exp, c_exp, d_exp),
            color='#27AE60',
            linewidth=2.5,
            linestyle='-.',
            label=f'Exp: R²={r2_exp:.4f}',
            zorder=2
        )

    ax.set_xlabel('Sensationalism Score (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fake News Ratio (r, %)', fontsize=13, fontweight='bold')
    
    title_text = (
        f'Relationship between Sensationalism and Fake News Ratio\n'
        f'Binned by 0.1 intervals | Pearson r = {pearson_r:.4f}, p = {p_value:.4f} '
        f'{"(Significant)" if p_value < 0.05 else "(Not Significant)"}'
    )
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_xlim([s.min() - 0.03, s.max() + 0.03])

    plt.tight_layout()
    plt.savefig('fake_news_s_r_relation_binned.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("✅ 图表已保存：fake_news_s_r_relation_binned.png\n")
    plt.show()
    
    # 保存统计表
    result_table.to_csv('fake_news_s_r_binned_analysis.csv', index=False)
    print("✅ 统计表已保存：fake_news_s_r_binned_analysis.csv\n")
    
    print("=" * 90 + "\n")


if __name__ == "__main__":
    analyze_s_r_relation()
