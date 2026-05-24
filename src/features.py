"""
特征工程模块：
- Worker 特征: quality, 历史参与统计, category/industry 偏好分布
- Project 特征: category, industry, 时间特征, 报酬, 参与度, 质量
- Worker-Project 匹配特征: category 匹配度, industry 匹配度, 质量匹配度
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. Worker 特征
# ============================================================

def build_worker_features(worker_df: pd.DataFrame,
                           entries_df: pd.DataFrame,
                           projects_dict: Dict[int, dict],
                           industry_map: Dict[str, int]) -> pd.DataFrame:
    """
    构造 worker 特征表。
    特征列:
        worker_id
        worker_quality           (归一化后, 0-1)
        worker_quality_raw       (原始值)
        worker_total_entries     (历史总回答数)
        worker_total_projects    (参与的不同项目数)
        worker_win_count         (中标次数)
        worker_win_rate          (中标率)
        worker_finalist_count    (入围次数)
        worker_avg_score         (平均得分)
        worker_pref_category     (最常参与的大类)
        worker_pref_sub_category (最常参与的子类)
        worker_pref_industry     (最常参与的行业编码)
        worker_category_entropy  (category 分布的熵)
        worker_active_days       (活跃天数)
        worker_first_entry       (首次参与时间)
        worker_last_entry        (最后参与时间)
    """
    print("\n[3.1] 构造 Worker 特征 ...")

    # --- 基础特征来自 worker_df ---
    wf = worker_df[["worker_id", "worker_quality", "worker_quality_raw"]].copy()
    wf = wf.set_index("worker_id")

    # --- 从 entries 聚合统计 ---
    if len(entries_df) == 0:
        print("  Warning: entries_df is empty!")
        return wf.reset_index()

    # 每个 worker 的 entry 统计
    entry_grp = entries_df.groupby("worker")
    wf["worker_total_entries"] = entry_grp.size().astype(int)
    wf["worker_total_projects"] = entry_grp["project_id"].nunique().astype(int)
    wf["worker_win_count"] = entry_grp["winner"].sum().astype(int)
    wf["worker_finalist_count"] = entry_grp["finalist"].sum().astype(int)
    wf["worker_avg_score"] = entry_grp["score"].mean()
    wf["worker_first_entry"] = entry_grp["entry_created_at"].min()
    wf["worker_last_entry"] = entry_grp["entry_created_at"].max()

    # 派生特征
    wf["worker_win_rate"] = np.where(
        wf["worker_total_entries"] > 0,
        wf["worker_win_count"] / wf["worker_total_entries"],
        0.0
    )
    wf["worker_active_days"] = (
        (wf["worker_last_entry"] - wf["worker_first_entry"]).dt.days + 1
    ).fillna(0).astype(int)

    # 填充 NaN (对于在 entries 中没有记录的 worker)
    for col in ["worker_total_entries", "worker_total_projects",
                 "worker_win_count", "worker_finalist_count",
                 "worker_avg_score", "worker_win_rate", "worker_active_days"]:
        wf[col] = wf[col].fillna(0)

    # --- Category / Sub_category / Industry 偏好 ---
    # 构建 project_id -> (category, sub_category, industry) 映射
    pid_to_cat = {}
    pid_to_subcat = {}
    pid_to_ind = {}
    for pid, proj in projects_dict.items():
        pid_to_cat[pid] = proj.get("category", -1)
        pid_to_subcat[pid] = proj.get("sub_category", -1)
        pid_to_ind[pid] = industry_map.get(proj.get("industry", ""), -1)

    entries_with_cat = entries_df.copy()
    entries_with_cat["category"] = entries_with_cat["project_id"].map(pid_to_cat)
    entries_with_cat["sub_category"] = entries_with_cat["project_id"].map(pid_to_subcat)
    entries_with_cat["industry_code"] = entries_with_cat["project_id"].map(pid_to_ind)

    # 计算每个 worker 偏好的 category/sub_category/industry（最频繁出现的）
    def _mode(x):
        if len(x) == 0:
            return -1
        return x.mode().iloc[0] if not x.mode().empty else -1

    def _entropy(x):
        """计算分布的熵"""
        if len(x) == 0:
            return 0.0
        counts = x.value_counts(normalize=True)
        return -np.sum(counts * np.log(counts + 1e-10))

    cat_grp = entries_with_cat.groupby("worker")
    wf["worker_pref_category"] = cat_grp["category"].apply(_mode)
    wf["worker_pref_sub_category"] = cat_grp["sub_category"].apply(_mode)
    wf["worker_pref_industry"] = cat_grp["industry_code"].apply(_mode)
    wf["worker_category_entropy"] = cat_grp["category"].apply(_entropy)

    # 填充无历史记录的 worker
    fill_cols = ["worker_pref_category", "worker_pref_sub_category",
                  "worker_pref_industry", "worker_category_entropy"]
    wf[fill_cols] = wf[fill_cols].fillna(-1)

    wf = wf.reset_index()
    print(f"  Worker 特征: {len(wf)} workers, {len(wf.columns)} features")
    return wf


# ============================================================
# 2. Project 特征
# ============================================================

def build_project_features(projects_dict: Dict[int, dict],
                            entries_df: pd.DataFrame,
                            industry_map: Dict[str, int],
                            global_start_time: pd.Timestamp) -> pd.DataFrame:
    """
    构造 project 特征表。
    特征列:
        project_id
        project_category
        project_sub_category
        project_industry_code
        project_entry_count       (总回答数)
        project_total_awards      (奖金金额)
        project_duration_days     (项目持续天数: deadline - start_date)
        project_is_featured       (是否精选)
        project_average_score     (项目平均得分)
        project_worker_count      (参与worker数)
        project_winner_quality    (中标者平均quality)
        project_has_winner        (是否有中标者)
    """
    print("\n[3.2] 构造 Project 特征 ...")

    rows = []
    for pid, proj in projects_dict.items():
        start = proj.get("start_date_parsed")
        deadline = proj.get("deadline_parsed")
        duration_days = (deadline - start).days if start and deadline else -1

        rows.append({
            "project_id": pid,
            "project_category": proj.get("category", -1),
            "project_sub_category": proj.get("sub_category", -1),
            "project_industry_code": industry_map.get(proj.get("industry", ""), -1),
            "project_entry_count": proj.get("entry_count", 0),
            "project_total_awards": float(proj.get("total_awards", 0) or 0),
            "project_duration_days": duration_days,
            "project_is_featured": int(proj.get("featured", False)),
            "project_average_score": float(proj.get("average_score", 0) or 0),
            "project_start_date": start,
            "project_deadline": deadline,
        })

    pf = pd.DataFrame(rows)

    # --- 从 entries 聚合统计 ---
    if len(entries_df) > 0:
        e_grp = entries_df.groupby("project_id")
        pf["project_worker_count"] = pf["project_id"].map(
            e_grp["worker"].nunique()
        ).fillna(0).astype(int)
        pf["project_has_winner"] = pf["project_id"].map(
            e_grp["winner"].max()
        ).fillna(False).astype(int)
    else:
        pf["project_worker_count"] = 0
        pf["project_has_winner"] = 0

    # 填充 NaN
    pf = pf.fillna({
        "project_category": -1,
        "project_sub_category": -1,
        "project_industry_code": -1,
        "project_entry_count": 0,
        "project_total_awards": 0.0,
        "project_duration_days": 0,
        "project_average_score": 0.0,
    })

    print(f"  Project 特征: {len(pf)} projects, {len(pf.columns)} features")
    return pf


# ============================================================
# 3. Worker-Project 匹配特征（动态计算，用于每个事件）
# ============================================================

def compute_match_features(worker_row: pd.Series,
                            project_row: pd.Series) -> Dict[str, float]:
    """
    给定一个 worker 和 project 的特征行，计算匹配特征。
    返回字典包含:
        match_category, match_sub_category, match_industry, match_quality_gap
    """
    features = {}

    def _safe_val(val, default=-1):
        """处理 NaN / NaT"""
        try:
            if pd.isna(val):
                return default
            return float(val) if not isinstance(val, (str, bool)) else val
        except (TypeError, ValueError):
            return default

    # Category 匹配
    w_cat = _safe_val(worker_row.get("worker_pref_category", -1))
    p_cat = _safe_val(project_row.get("project_category", -1))
    features["match_category"] = 1.0 if (w_cat == p_cat and w_cat != -1) else 0.0

    # Sub-category 匹配
    w_sub = _safe_val(worker_row.get("worker_pref_sub_category", -1))
    p_sub = _safe_val(project_row.get("project_sub_category", -1))
    features["match_sub_category"] = 1.0 if (w_sub == p_sub and w_sub != -1) else 0.0

    # Industry 匹配
    w_ind = _safe_val(worker_row.get("worker_pref_industry", -1))
    p_ind = _safe_val(project_row.get("project_industry_code", -1))
    features["match_industry"] = 1.0 if (w_ind == p_ind and w_ind != -1) else 0.0

    # Quality gap
    w_q = _safe_val(worker_row.get("worker_quality", 0.5), 0.5)
    p_avg = _safe_val(project_row.get("project_average_score", 0.0), 0.0)
    p_avg_norm = min(max(p_avg / 5.0, 0.0), 1.0)  # clip
    features["match_quality_gap"] = abs(w_q - p_avg_norm)

    return features
