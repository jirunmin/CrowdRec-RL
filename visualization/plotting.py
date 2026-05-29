"""
Visualization Tools for Experiment Results

Author: E角色
Description: 提供专业的数据可视化功能，用于生成报告和展示所需的图表
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11


def plot_comparison_bar_chart(
    results: List[Dict],
    metric: str = "avg_reward",
    title: str = "Method Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    绘制方法对比柱状图

    用途：
    ====
    这是最常用的图表，用于直观对比不同方法的性能。
    在报告中，每个评估指标都应该有这样一个图。

    图表特点：
    - X轴：不同方法名称（Random, Greedy-Worker, DQN等）
    - Y轴：选定的指标值（如avg_reward, hit_rate等）
    - 分组显示：如果有多个reward_mode（worker/requester），用不同颜色区分
    - 数值标签：在每个柱子上方显示具体数值

    Args:
        results: 评估结果列表（来自run_full_evaluation的返回值）
        metric: 要比较的指标名（"total_reward", "avg_reward", "hit_rate"等）
        title: 图表标题
        save_path: 保存路径（可选），支持.png, .pdf, .svg格式

    Returns:
        plt.Figure: matplotlib图表对象

    使用示例：
    ========
    >>> results = run_full_evaluation(split="val", reward_mode="worker")
    >>> fig = plot_comparison_bar_chart(results, metric="hit_rate",
    ...                                 title="Hit@1 Comparison (Worker Mode)")
    >>> plt.show()
    """
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = df['method_name'].unique()
    modes = df['reward_mode'].unique() if 'reward_mode' in df.columns else ['default']

    x = np.arange(len(methods))
    width = 0.35

    colors = ['#2ecc71', '#3498db']

    for i, mode in enumerate(modes):
        mode_data = df[df['reward_mode'] == mode] if 'reward_mode' in df.columns else df

        values = []
        for m in methods:
            match = mode_data[mode_data['method_name'] == m]
            if len(match) > 0 and 'results' in match.iloc[0]:
                values.append(match.iloc[0]['results'].get(metric, 0))
            else:
                values.append(0)

        bars = ax.bar(x + i * width, values, width,
                     label=mode.capitalize(),
                     color=colors[i % len(colors)],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001 * max(values),
                   f'{val:.4f}',
                   ha='center',
                   va='bottom',
                   fontsize=9,
                   fontweight='bold')

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.legend(title='Reward Mode', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved to: {save_path.absolute()}")

    return fig


def plot_training_curves(
    training_data: Dict[str, Dict[str, List[float]]],
    metrics: List[str] = ["reward", "loss"],
    smooth_window: int = 100,
    save_path: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    绘制DQN训练曲线（需要C/D提供训练日志）

    用途：
    ====
    展示模型训练过程中的收敛情况，是验证模型是否正常训练的重要依据。

    通常包含：
    - Episode Reward曲线：显示学习进度
    - Loss曲线：显示优化过程
    - Epsilon曲线：显示探索-利用策略的变化
    - Q-value曲线：显示价值估计稳定性

    Args:
        training_data: 训练数据字典
            格式：{
                "DQN": {
                    "steps": [100, 200, 300, ...],
                    "rewards": [1.2, 1.5, 1.8, ...],
                    "losses": [0.5, 0.3, 0.2, ...]
                },
                "Double-DQN": { ... }
            }
        metrics: 要绘制的指标列表
        smooth_window: 移动平均窗口大小（用于平滑噪声）
        save_path: 保存路径前缀（会自动添加"_reward.png"等后缀）

    Returns:
        Dict[str, plt.Figure]: 每个指标一个图表

    注意事项：
    ========
    这个函数需要C/D角色先完成训练并提供训练日志！
    如果没有训练数据，可以先跳过这个函数。
    """
    figures = {}

    for metric_name in metrics:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(training_data)))

        for idx, (method_name, data) in enumerate(training_data.items()):
            if metric_name not in data or 'steps' not in data:
                continue

            steps = data['steps']
            values = data[metric_name]

            if len(steps) == 0 or len(values) == 0:
                continue

            if smooth_window > 1 and len(values) > smooth_window:
                values_smooth = pd.Series(values).rolling(
                    window=smooth_window, min_periods=1).mean()
                axes.plot(steps, values_smooth,
                         label=method_name,
                         color=colors[idx],
                         linewidth=2.5,
                         alpha=0.9)

                axes.fill_between(steps,
                                 values_smooth - pd.Series(values).rolling(
                                     window=smooth_window, min_periods=1).std(),
                                 values_smooth + pd.Series(values).rolling(
                                     window=smooth_window, min_periods=1).std(),
                                 color=colors[idx], alpha=0.1)

                axes.plot(steps, values,
                         color=colors[idx],
                         linewidth=1,
                         alpha=0.3)
            else:
                axes.plot(steps, values,
                         label=method_name,
                         color=colors[idx],
                         linewidth=2)

        axes.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
        axes.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes.set_title(f'Training Curve: {metric_name.replace("_", " ").title()}',
                      fontsize=14, fontweight='bold', pad=20)
        axes.legend(fontsize=10, loc='best')
        axes.grid(True, alpha=0.3, linestyle='--')
        axes.set_axisbelow(True)

        for spine in ['top', 'right']:
            axes.spines[spine].set_visible(False)

        plt.tight_layout()

        figures[metric_name] = fig

        if save_path:
            path = Path(f"{save_path}_{metric_name}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Training curve saved: {path.absolute()}")

    return figures


def plot_improvement_chart(
    baseline_results: Dict[str, float],
    improved_results: Dict[str, float],
    baseline_name: str = "Random",
    improved_name: str = "DQN",
    title: str = "Performance Improvement",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    绘制相对提升率图表

    用途：
    ====
    直观展示DQN方法相比baseline提升了多少百分比。
    这种图在汇报时特别有说服力！

    计算公式：
    Improvement% = (improved - baseline) / |baseline| × 100%

    Args:
        baseline_results: 基线方法的结果字典 {"hit_rate": 0.3, "avg_reward": 1.2}
        improved_results: 改进方法的结果字典 {"hit_rate": 0.45, "avg_reward": 1.8}
        baseline_name: 基线方法名称
        improved_name: 改进方法名称
        title: 图表标题
        save_path: 保存路径

    Returns:
        plt.Figure: 图表对象
    """
    common_keys = set(baseline_results.keys()) & set(improved_results.keys())

    improvements = {}
    for key in common_keys:
        base_val = baseline_results[key]
        imp_val = improved_results[key]

        if base_val != 0:
            improvement = ((imp_val - base_val) / abs(base_val)) * 100
            improvements[key] = improvement
        else:
            improvements[key] = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))

    keys = list(improvements.keys())
    values = list(improvements.values())
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]

    bars = ax.barh(keys, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        width = bar.get_width()
        label_x_pos = width + (1 if width >= 0 else -1)
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2.,
               f'{val:+.1f}%',
               ha=ha, va='center',
               fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n({improved_name} vs {baseline_name})',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Improvement chart saved to: {save_path.absolute()}")

    return fig


def save_all_figures(
    results: List[Dict],
    output_dir: str = "report/images",
    prefix: str = ""
) -> List[Path]:
    """
    批量保存所有标准图表

    一键生成报告所需的所有图表。

    会生成的图表：
    1. avg_reward_comparison.png - 平均奖励对比
    2. hit_rate_comparison.png - Hit@1对比
    3. total_reward_comparison.png - 总奖励对比

    Args:
        results: 评估结果列表
        output_dir: 输出目录
        prefix: 文件名前缀（如"val_worker_"）

    Returns:
        List[Path]: 保存的文件路径列表
    """
    saved_files = []

    metrics_to_plot = [
        ("avg_reward", "Average Reward Comparison"),
        ("hit_rate", "Hit@1 Rate Comparison"),
        ("total_reward", "Total Reward Comparison"),
    ]

    for metric, title in metrics_to_plot:
        filename = f"{prefix}{metric}_comparison.png"
        save_path = Path(output_dir) / filename

        try:
            fig = plot_comparison_bar_chart(
                results,
                metric=metric,
                title=title,
                save_path=save_path
            )
            saved_files.append(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"❌ Error plotting {metric}: {e}")

    print(f"\n✓ Saved {len(saved_files)} figures to {output_dir}/")
    return saved_files


if __name__ == "__main__":
    print("Testing visualization tools...")

    mock_results = [
        {
            "method_name": "Random",
            "reward_mode": "worker",
            "results": {
                "total_reward": 15000.0,
                "avg_reward": 0.25,
                "hit_rate": 0.15,
                "std_reward": 0.08
            }
        },
        {
            "method_name": "Greedy-Worker",
            "reward_mode": "worker",
            "results": {
                "total_reward": 25000.0,
                "avg_reward": 0.42,
                "hit_rate": 0.28,
                "std_reward": 0.12
            }
        },
        {
            "method_name": "Greedy-Requester",
            "reward_mode": "worker",
            "results": {
                "total_reward": 18000.0,
                "avg_reward": 0.30,
                "hit_rate": 0.20,
                "std_reward": 0.09
            }
        }
    ]

    print("\n1. Testing plot_comparison_bar_chart...")
    fig = plot_comparison_bar_chart(mock_results, metric="hit_rate",
                                   title="Test: Hit@1 Comparison")
    plt.close(fig)
    print("   ✓ Bar chart test passed!")

    print("\n2. Testing plot_improvement_chart...")
    baseline = {"hit_rate": 0.15, "avg_reward": 0.25}
    improved = {"hit_rate": 0.35, "avg_reward": 0.45}
    fig = plot_improvement_chart(baseline, improved,
                                baseline_name="Random",
                                improved_name="DQN",
                                title="Test: Improvement")
    plt.close(fig)
    print("   ✓ Improvement chart test passed!")

    print("\n✓ All visualization tests passed!")
