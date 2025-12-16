import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# data = pd.read_csv('2024-05-5_aging_20s_aging.csv', parse_dates=['time'])
data = pd.read_csv('qq.csv', parse_dates=['time'])
# 设置老化数据范围（第4000到7125个数据点）
start_idx, end_idx = 5000, 7118
start_idx, end_idx = 9380, 10570

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
plt.subplots_adjust(hspace=0.3)

# 设置全局中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义绘图参数
metrics = ['cpu_usage', 'memory_usage', 'response_time']
titles = ['CPU使用率 (%)', '内存使用率 (%)', '系统响应时间 (s)']

# 使用统一的颜色方案
normal_color = '#4E79A7'  # 正常数据颜色（蓝色）
aging_color = '#E74C3C'  # 老化数据线条颜色（深红色）
normal_fill_color = '#A0CBE8'  # 正常数据填充色（浅蓝色）
aging_fill_color = '#FADBD8'  # 老化数据填充色（浅红色）
aging_span_color = '#D3D3D3'  # 老化时间段背景色（浅灰色）

# 遍历绘制每个子图
for idx, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
    # 绘制正常数据
    ax.plot(data.index, data[metric], color=normal_color, linewidth=1, label='正常数据')
    ax.fill_between(data.index, data[metric], color=normal_fill_color, alpha=0.3)

    # 绘制老化数据
    ax.plot(data.index[start_idx:end_idx + 1],
            data[metric].iloc[start_idx:end_idx + 1],
            color=aging_color, linewidth=1, label='老化数据')
    ax.fill_between(data.index[start_idx:end_idx + 1],
                    data[metric].iloc[start_idx:end_idx + 1],
                    color=aging_fill_color, alpha=0.3)

    # 标记老化时间段
    ax.axvspan(data.index[start_idx], data.index[end_idx], color=aging_span_color, alpha=0.2, label='老化阶段')

    # 设置子图标题和样式
    ax.set_ylabel(title, fontsize=9, labelpad=1)
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(data.index[0], data.index[-1])

    # 设置第一个子图（CPU使用率）的纵坐标范围为0到115
    if idx == 0:
        ax.set_ylim(0, 115)

    # 为每个子图添加图例
    lines = [
        plt.Line2D([0], [0], color=normal_color, lw=1, label='正常数据'),
        plt.Line2D([0], [0], color=aging_color, lw=1, label='老化数据'),
        plt.Line2D([0], [0], color=aging_span_color, alpha=0.9, lw=6, label='老化时间段')
    ]
    ax.legend(
        handles=lines,
        loc='upper left',
        framealpha=0.9,
        fontsize=8,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.5
    )

# 设置公共x轴标签
plt.xlabel('时间步', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存/显示
plt.savefig('aging_analysis_1.png', dpi=300, bbox_inches='tight')
plt.show()