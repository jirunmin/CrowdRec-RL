#!/usr/bin/env python3
"""
数据统计可视化：生成 8 张图表展示数据全貌。
输出到 figures/ 目录。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 设置中文字体
# ============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def load_data():
    print("Loading data...")
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_events.parquet"))
    val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_events.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_events.parquet"))
    workers = pd.read_parquet(os.path.join(PROCESSED_DIR, "worker_features.parquet"))
    projects = pd.read_parquet(os.path.join(PROCESSED_DIR, "project_features.parquet"))
    candidates = pd.read_parquet(os.path.join(PROCESSED_DIR, "candidates.parquet"))

    # 合并并标记数据集
    train["split"] = "Train"
    val["split"] = "Val"
    test["split"] = "Test"
    all_events = pd.concat([train, val, test], ignore_index=True)
    return all_events, workers, projects, candidates


def fig1_entry_over_time(all_events):
    """图1: entry 事件随时间的分布（按月聚合）"""
    print("  [1/8] Entry events over time...")
    pos = all_events[all_events["label"] == 1].copy()
    pos["month"] = pos["timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = pos.groupby("month").size()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(monthly.index, monthly.values, alpha=0.3, color="#2196F3")
    ax.plot(monthly.index, monthly.values, color="#1976D2", linewidth=1.5)
    ax.set_title("Monthly Entry Events Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Entries")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_entry_over_time.png"), dpi=150)
    plt.close(fig)


def fig2_worker_quality(workers):
    """图2: Worker quality 分布"""
    print("  [2/8] Worker quality distribution...")
    wf = workers[workers["worker_quality_raw"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(wf["worker_quality_raw"], bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax1.axvline(wf["worker_quality_raw"].median(), color="red", linestyle="--", linewidth=2,
                label=f"Median={wf['worker_quality_raw'].median():.0f}")
    ax1.set_title("Worker Quality Distribution (raw 0-100)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Quality Score")
    ax1.set_ylabel("Number of Workers")
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(wf["worker_total_entries"], bins=50, color="#FF9800", edgecolor="white", alpha=0.8)
    ax2.set_title("Worker Activity Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Total Entries per Worker")
    ax2.set_ylabel("Number of Workers")
    ax2.set_xlim(0, wf["worker_total_entries"].quantile(0.95))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_worker_quality.png"), dpi=150)
    plt.close(fig)


def fig3_project_stats(projects):
    """图3: 项目特征分布"""
    print("  [3/8] Project statistics...")
    pf = projects.copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 项目回答数分布
    ax = axes[0, 0]
    ax.hist(pf["project_entry_count"].clip(upper=pf["project_entry_count"].quantile(0.95)),
            bins=40, color="#9C27B0", edgecolor="white", alpha=0.8)
    ax.set_title("Project Entry Count Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Entry Count")

    # 奖金分布
    ax = axes[0, 1]
    awards = pf[pf["project_total_awards"] > 0]["project_total_awards"]
    ax.hist(awards.clip(upper=awards.quantile(0.95)), bins=40, color="#E91E63", edgecolor="white", alpha=0.8)
    ax.set_title("Project Award Distribution ($)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Awards ($)")

    # 持续天数
    ax = axes[1, 0]
    ax.hist(pf["project_duration_days"].clip(upper=30), bins=30, color="#00BCD4", edgecolor="white", alpha=0.8)
    ax.set_title("Project Duration Distribution (days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Duration (days)")

    # 精选比例
    ax = axes[1, 1]
    featured_counts = pf["project_is_featured"].value_counts()
    ax.pie(featured_counts.values, labels=["Not Featured", "Featured"],
           autopct="%1.1f%%", colors=["#CFD8DC", "#FFC107"], startangle=90)
    ax.set_title("Featured vs Non-featured Projects", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_project_stats.png"), dpi=150)
    plt.close(fig)


def fig4_reward_distribution(all_events):
    """图4: Reward 分布"""
    print("  [4/8] Reward distribution...")
    pos = all_events[all_events["label"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(pos["reward_worker"], bins=30, color="#FF5722", edgecolor="white", alpha=0.8)
    ax.axvline(pos["reward_worker"].mean(), color="blue", linestyle="--", linewidth=2,
               label=f"Mean={pos['reward_worker'].mean():.2f}")
    ax.set_title("Worker Reward Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reward")
    ax.legend()

    ax = axes[1]
    ax.hist(pos["reward_requester"], bins=30, color="#3F51B5", edgecolor="white", alpha=0.8)
    ax.axvline(pos["reward_requester"].mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean={pos['reward_requester'].mean():.2f}")
    ax.set_title("Requester Reward Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Reward")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_reward_distribution.png"), dpi=150)
    plt.close(fig)


def fig5_category_distribution(all_events, projects):
    """图5: 行业/类别分布"""
    print("  [5/8] Category/Industry distribution...")
    pos = all_events[all_events["label"] == 1]
    # 按项目聚合
    proj_popularity = pos.groupby("project_id").size().sort_values(ascending=False)

    pf = projects.set_index("project_id")
    top_projs = proj_popularity.head(15)
    top_industries = pf.loc[top_projs.index, "project_industry_code"].value_counts().head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = sns.color_palette("viridis", len(top_industries))
    ax.barh(range(len(top_industries)), top_industries.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_industries)))
    ax.set_yticklabels([f"Industry {i}" for i in top_industries.index])
    ax.set_title("Top 10 Industries by Entries", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Entries")
    ax.invert_yaxis()

    ax = axes[1]
    cats = pf["project_category"].value_counts().head(12)
    ax.barh(range(len(cats)), cats.values, color=sns.color_palette("plasma", len(cats)), edgecolor="white")
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"Cat {c}" for c in cats.index])
    ax.set_title("Top 12 Categories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Projects")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_category_distribution.png"), dpi=150)
    plt.close(fig)


def fig6_split_overview(all_events):
    """图6: 数据集划分概览"""
    print("  [6/8] Dataset split overview...")
    splits = all_events.groupby("split").agg(
        total=("label", "count"),
        positive=("label", "sum"),
    ).reset_index()
    splits["negative"] = splits["total"] - splits["positive"]
    splits["pos_ratio"] = splits["positive"] / splits["total"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = range(len(splits))
    w = 0.35
    ax.bar([i - w/2 for i in x], splits["positive"], w, label="Positive (label=1)",
           color="#4CAF50", edgecolor="white")
    ax.bar([i + w/2 for i in x], splits["negative"], w, label="Negative (label=0)",
           color="#FF9800", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(splits["split"])
    ax.set_title("Dataset Split: Positive vs Negative", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Events")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend()

    ax = axes[1]
    time_ranges = all_events.groupby("split").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
    ).reset_index()
    for _, row in time_ranges.iterrows():
        ax.barh(row["split"], (row["end"] - row["start"]).days,
                left=0, height=0.5, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.set_title("Time Span per Split", fontsize=13, fontweight="bold")
    ax.set_xlabel("Duration (days)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_split_overview.png"), dpi=150)
    plt.close(fig)


def fig7_positive_vs_negative(all_events):
    """图7: 正负样本特征对比"""
    print("  [7/8] Positive vs Negative comparison...")
    df = all_events.copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Worker quality 对比
    ax = axes[0]
    for label, color, name in [(1, "#4CAF50", "Positive"), (0, "#F44336", "Negative")]:
        subset = df[df["label"] == label]["worker_quality"].dropna()
        ax.hist(subset, bins=30, alpha=0.5, color=color, label=name)
    ax.set_title("Worker Quality: Pos vs Neg", fontsize=12, fontweight="bold")
    ax.set_xlabel("Worker Quality")
    ax.legend()

    # Match category 对比
    ax = axes[1]
    match_pos = df[df["label"] == 1]["match_category"].mean()
    match_neg = df[df["label"] == 0]["match_category"].mean()
    ax.bar(["Positive", "Negative"], [match_pos, match_neg],
           color=["#4CAF50", "#F44336"], edgecolor="white")
    ax.set_title("Category Match Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Match Rate")
    for i, v in enumerate([match_pos, match_neg]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontweight="bold")

    # Project entry count 对比
    ax = axes[2]
    for label, color, name in [(1, "#4CAF50", "Positive"), (0, "#F44336", "Negative")]:
        subset = df[df["label"] == label]["project_entry_count"].dropna().clip(upper=200)
        ax.hist(subset, bins=30, alpha=0.5, color=color, label=name)
    ax.set_title("Project Entry Count: Pos vs Neg", fontsize=12, fontweight="bold")
    ax.set_xlabel("Project Entry Count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_pos_vs_neg.png"), dpi=150)
    plt.close(fig)


def fig8_candidate_stats(candidates):
    """图8: 候选集统计"""
    print("  [8/8] Candidate statistics...")
    cand_cnt = candidates["candidate_projects"].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(cand_cnt, bins=25, color="#009688", edgecolor="white", alpha=0.8)
    ax.axvline(cand_cnt.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean={cand_cnt.mean():.1f}")
    ax.set_title("Candidate Pool Size Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Available Candidates")
    ax.set_ylabel("Frequency")
    ax.legend()

    ax = axes[1]
    ax.plot(cand_cnt.values[:500], linewidth=0.5, alpha=0.7, color="#009688")
    ax.set_title("Candidate Count Over First 500 Events", fontsize=13, fontweight="bold")
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Candidates Available")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "08_candidate_stats.png"), dpi=150)
    plt.close(fig)


def print_summary(all_events, workers, projects, candidates):
    """打印文本统计摘要"""
    pos = all_events[all_events["label"] == 1]
    print("\n" + "=" * 60)
    print("DATA STATISTICS SUMMARY")
    print("=" * 60)
    print(f"  Total events:        {len(all_events):,}")
    print(f"  Positive events:     {len(pos):,} ({100*len(pos)/len(all_events):.1f}%)")
    print(f"  Negative events:     {len(all_events)-len(pos):,}")
    print(f"  Unique workers:      {workers['worker_id'].nunique():,}")
    print(f"  Unique projects:     {projects['project_id'].nunique():,}")
    print(f"  Timespan:            {all_events['timestamp'].min().date()} ~ {all_events['timestamp'].max().date()}")
    print(f"  Worker quality:      mean={workers['worker_quality_raw'].mean():.1f}, median={workers['worker_quality_raw'].median():.0f}")
    print(f"  Avg entries/worker:  {workers['worker_total_entries'].mean():.1f}")
    print(f"  Avg entries/project: {projects['project_entry_count'].mean():.1f}")
    print(f"  Winners:             {pos['winner'].sum():,} ({100*pos['winner'].sum()/len(pos):.1f}%)")
    print(f"  Finalists:           {pos['finalist'].sum():,}")
    print(f"  Avg candidates:      {candidates['candidate_projects'].apply(len).mean():.1f}")
    print("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_events, workers, projects, candidates = load_data()

    print_summary(all_events, workers, projects, candidates)

    print("\nGenerating figures...")
    fig1_entry_over_time(all_events)
    fig2_worker_quality(workers)
    fig3_project_stats(projects)
    fig4_reward_distribution(all_events)
    fig5_category_distribution(all_events, projects)
    fig6_split_overview(all_events)
    fig7_positive_vs_negative(all_events)
    fig8_candidate_stats(candidates)

    print(f"\nDone! {len(os.listdir(OUTPUT_DIR))} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
