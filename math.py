import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.style.use('seaborn-v0_8-whitegrid')


# ===================== 1. 定义拟合函数 =====================

# 线性：r = a * s + b
def linear_model(s, a, b):
    return a * s + b

# 对数：r = a * log(b * s + c) + d
# 为避免 log 负数，约束 c > 0, b > 0
def log_model(s, a, b, c, d):
    return a * np.log(b * s + c) + d

# 指数：r = a * exp(b * s + c) + d
def exp_model(s, a, b, c, d):
    return a * np.exp(b * s + c) + d


def plot_s_r_relation():
    print("\n" + "=" * 70)
    print("📊 假新闻率 r 与耸人听闻指数 s 关系拟合与可视化")
    print("=" * 70)

    # ===================== 2. 读取 topic 数据 =====================
    try:
        topic_analysis = pd.read_csv('topic_analysis_improved.csv')
    except FileNotFoundError as e:
        print(f"❌ 文件未找到：{e}")
        print("请确认当前目录下存在 topic_analysis_improved.csv")
        return

    # 耸人听闻指数 s
    s = topic_analysis['avg_sensationalism'].values.astype(float)
    # 假新闻比例 r（%）
    r = (topic_analysis['predicted_fake_ratio_rf'].values.astype(float) * 100.0)

    # 如果有少量 NaN，做个简单清洗
    mask = ~np.isnan(s) & ~np.isnan(r)
    s = s[mask]
    r = r[mask]

    # ===================== 3. 拟合三种模型 =====================

    # 3.1 线性拟合
    popt_lin, _ = curve_fit(linear_model, s, r)
    a_lin, b_lin = popt_lin
    r_pred_lin = linear_model(s, a_lin, b_lin)
    r2_lin = 1 - np.sum((r - r_pred_lin) ** 2) / np.sum((r - r.mean()) ** 2)

    # 3.2 对数拟合：r = a log(b s + c) + d
    # 初始值 & 参数约束，让 b>0, c>0，避免 log 里面为负
    # 注意：如果 s 非常小，可以适当调大 c0
    a0, b0, c0, d0 = 1.0, 1.0, 1e-3, r.mean()
    bounds_log = ([-np.inf, 1e-6, 1e-6, -np.inf],
                  [ np.inf,  np.inf,  np.inf,  np.inf])
    try:
        popt_log, _ = curve_fit(
            log_model, s, r,
            p0=[a0, b0, c0, d0],
            bounds=bounds_log,
            maxfev=10000
        )
        a_log, b_log, c_log, d_log = popt_log
        r_pred_log = log_model(s, a_log, b_log, c_log, d_log)
        r2_log = 1 - np.sum((r - r_pred_log) ** 2) / np.sum((r - r.mean()) ** 2)
        log_ok = True
    except Exception as e:
        print(f"⚠️ 对数模型拟合失败：{e}")
        log_ok = False

    # 3.3 指数拟合：r = a exp(b s + c) + d
    a0, b0, c0, d0 = 1.0, 1.0, 0.0, r.mean()
    try:
        popt_exp, _ = curve_fit(
            exp_model, s, r,
            p0=[a0, b0, c0, d0],
            maxfev=10000
        )
        a_exp, b_exp, c_exp, d_exp = popt_exp
        r_pred_exp = exp_model(s, a_exp, b_exp, c_exp, d_exp)
        r2_exp = 1 - np.sum((r - r_pred_exp) ** 2) / np.sum((r - r.mean()) ** 2)
        exp_ok = True
    except Exception as e:
        print(f"⚠️ 指数模型拟合失败：{e}")
        exp_ok = False

    # ===================== 4. 画图：散点 + 拟合曲线 =====================

    fig, ax = plt.subplots(figsize=(9, 6))

    # 原始散点（每个点一个 topic）
    ax.scatter(s, r, color='#34495E', s=80, edgecolor='black', alpha=0.8, label='Topics')
    # 标记 topic ID
    for idx, row in topic_analysis[mask].iterrows():
        ax.annotate(f"T{int(row['lda_topic'])}",
                    (row['avg_sensationalism'], row['predicted_fake_ratio_rf'] * 100),
                    xytext=(6, 0),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    alpha=0.85)

    # 用更细的 s 取值范围画平滑曲线
    s_line = np.linspace(s.min(), s.max(), 200)

    # 4.1 线性曲线
    ax.plot(
        s_line,
        linear_model(s_line, a_lin, b_lin),
        color='#E74C3C',
        linewidth=2.0,
        label=f'Linear: r = {a_lin:.2f}·s + {b_lin:.2f}  (R²={r2_lin:.3f})'
    )

    # 4.2 对数曲线（若拟合成功）
    if log_ok:
        ax.plot(
            s_line,
            log_model(s_line, a_log, b_log, c_log, d_log),
            color='#3498DB',
            linewidth=2.0,
            linestyle='--',
            label=f'Log: r = {a_log:.2f}·ln({b_log:.2f}·s + {c_log:.3f}) + {d_log:.2f}  (R²={r2_log:.3f})'
        )

    # 4.3 指数曲线（若拟合成功）
    if exp_ok:
        ax.plot(
            s_line,
            exp_model(s_line, a_exp, b_exp, c_exp, d_exp),
            color='#27AE60',
            linewidth=2.0,
            linestyle='-.',
            label=f'Exp: r = {a_exp:.2f}·exp({b_exp:.2f}·s + {c_exp:.2f}) + {d_exp:.2f}  (R²={r2_exp:.3f})'
        )

    ax.set_xlabel('Average Sensationalism Score (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Fake News Ratio (r, %)', fontsize=12, fontweight='bold')
    ax.set_title('Relationship between Sensationalism (s) and Fake News Ratio (r)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig('fake_news_s_r_relation.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print("\n✅ 关联图已保存：fake_news_s_r_relation.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    plot_s_r_relation()
