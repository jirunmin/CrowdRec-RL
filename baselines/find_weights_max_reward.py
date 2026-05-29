"""
最大化 Reward 的权重学习。

方法：用线性回归预测 reward 值 → 选预测 reward 最大的
目标：累积 reward 最大化

原理：
  reward = w^T · features + bias
  线性回归的系数 w 就是使预测 reward 最准的权重。
  与 max_hit 的区别：这里用 reward 值做回归，不是用 label 做分类。
  高 reward 的样本（quality 高、中标）对权重影响更大。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("最大化 Reward — 线性回归权重学习")
print("=" * 60)

events = pd.read_parquet("processed/train_events.parquet")
print(f"\n训练事件: {len(events)} (正:{(events['label']==1).sum()}, 负:{(events['label']==0).sum()})")

# Worker reward 和 Requester reward
y_worker = events["reward_worker"].values
y_requester = events["reward_requester"].values


def train(X, y, feature_names, target_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_scaled, y)
    r2 = lr.score(X_scaled, y)

    raw_coef = lr.coef_
    stds = scaler.scale_
    coef = raw_coef / stds

    print(f"\n  目标: {target_name}")
    print(f"  R²: {r2:.4f}")
    print(f"  预测值范围: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    print(f"\n  {'特征':>20} | {'权重':>10} | {'影响力':>10}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10}")
    for i, name in enumerate(feature_names):
        print(f"  {name:>20} | {coef[i]:>10.4f} | {coef[i]*stds[i]:>10.4f}")

    return coef


# --- Greedy-Worker 特征 ---
X_w = np.stack([
    events["project_total_awards"].fillna(0).values,
    (events["match_category"].fillna(0) + events["match_sub_category"].fillna(0) + events["match_industry"].fillna(0)).values,
    1.0 - events["match_quality_gap"].fillna(0.5).values,
    events["project_is_featured"].fillna(0).values,
    events["project_average_score"].fillna(0).values,
], axis=1)
w_names = ["awards", "total_match", "quality_match", "featured", "avg_score"]

# --- Greedy-Requester 特征 ---
X_r = np.stack([
    events["project_average_score"].fillna(0).values,
    events["project_has_winner"].fillna(0).values,
    events["project_worker_count"].fillna(0).values,
    events["match_industry"].fillna(0).values,
    events["project_entry_count"].fillna(0).values,
], axis=1)
r_names = ["avg_score", "has_winner", "worker_count", "match_industry", "entry_count"]


# ================================================================
print("\n" + "=" * 60)
print("Greedy-Worker 特征 → 预测 Worker Reward")
print("=" * 60)
w_coef_w = train(X_w, y_worker, w_names, "reward_worker")

print("\n" + "=" * 60)
print("Greedy-Worker 特征 → 预测 Requester Reward")
print("=" * 60)
w_coef_r = train(X_w, y_requester, w_names, "reward_requester")

print("\n" + "=" * 60)
print("Greedy-Requester 特征 → 预测 Worker Reward")
print("=" * 60)
r_coef_w = train(X_r, y_worker, r_names, "reward_worker")

print("\n" + "=" * 60)
print("Greedy-Requester 特征 → 预测 Requester Reward")
print("=" * 60)
r_coef_r = train(X_r, y_requester, r_names, "reward_requester")


# --- 输出 ---
print("\n" + "=" * 60)
print("可直接复制到代码的参数")
print("=" * 60)

# Greedy-Worker 参数名映射（脚本特征名 → 代码参数名）
# 注意：quality_match = 1 - quality_gap，所以符号取反
w_name_map = {
    "awards":          ("w_awards",        lambda c: c),
    "total_match":     ("w_match",         lambda c: c),
    "quality_match":   ("w_quality_gap",   lambda c: -c),  # 取反
    "featured":        ("w_featured",      lambda c: c),
    "avg_score":       ("w_score",         lambda c: c),
}

# Greedy-Requester 参数名映射
r_name_map = {
    "avg_score":       ("w_quality",       lambda c: c),
    "has_winner":      ("w_winner",        lambda c: c),
    "worker_count":    ("w_worker_count",  lambda c: c),
    "match_industry":  ("w_match",         lambda c: c),
    "entry_count":     ("w_entry_count",   lambda c: c),
}

def print_params(names, coefs, name_map, title):
    print(f"\n--- {title} ---")
    for name, c in zip(names, coefs):
        param_name, transform = name_map[name]
        val = transform(c)
        print(f"  {param_name}: float = {val:.8f},")

print_params(w_names, w_coef_w, w_name_map, "Greedy-Worker（worker reward）→ greedy_worker.py")
print_params(w_names, w_coef_r, w_name_map, "Greedy-Worker（requester reward）→ greedy_worker.py")
print_params(r_names, r_coef_w, r_name_map, "Greedy-Requester（worker reward）→ greedy_requester.py")
print_params(r_names, r_coef_r, r_name_map, "Greedy-Requester（requester reward）→ greedy_requester.py")


# ================================================================
# 全特征模型（合并所有特征）
# ================================================================
print("\n" + "=" * 60)
print("全特征模型（合并 Worker + Requester 特征）")
print("=" * 60)

X_all = np.stack([
    events["project_total_awards"].fillna(0).values,
    events["match_category"].fillna(0).values,
    events["match_sub_category"].fillna(0).values,
    events["match_industry"].fillna(0).values,
    1.0 - events["match_quality_gap"].fillna(0.5).values,
    events["project_is_featured"].fillna(0).values,
    events["project_average_score"].fillna(0).values,
    events["project_has_winner"].fillna(0).values,
    events["project_worker_count"].fillna(0).values,
    events["project_entry_count"].fillna(0).values,
], axis=1)
all_names = ["awards", "match_cat", "match_sub", "match_ind", "quality_match",
             "featured", "avg_score", "has_winner", "worker_count", "entry_count"]

print("\n--- 预测 Worker Reward ---")
all_coef_w = train(X_all, y_worker, all_names, "reward_worker")

print("\n--- 预测 Requester Reward ---")
all_coef_r = train(X_all, y_requester, all_names, "reward_requester")

# 输出合并后的权重（用于统一策略）
print("\n--- 全特征（worker reward）→ 统一策略 ---")
for name, c in zip(all_names, all_coef_w):
    print(f"  w_{name}: float = {c:.8f},")

print("\n--- 全特征（requester reward）→ 统一策略 ---")
for name, c in zip(all_names, all_coef_r):
    print(f"  w_{name}: float = {c:.8f},")
