"""
数据清洗模块：
- 处理 worker_quality 缺失值（-1）
- 统一时间格式
- 过滤无效项目（status 不为 awarded/active 等）
- 对齐 worker ID（entry 中的 worker 是否在 worker_quality 中）
- 处理重复、异常值
"""

from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse


# ============================================================
# 1. Worker Quality 清洗
# ============================================================

def clean_worker_quality(worker_df: pd.DataFrame,
                          fill_na: Optional[str] = "median") -> pd.DataFrame:
    """
    清洗 worker_quality：
    - quality == -1 视为缺失
    - 按 fill_na 策略填充: "median"(默认) / "mean" / "drop"(丢弃)
    - 将 quality 归一化到 [0, 1] 区间
    返回清洗后的 DataFrame，列: [worker_id, worker_quality, worker_quality_raw]
    """
    df = worker_df.copy()
    df["worker_quality_raw"] = df["worker_quality"].copy()

    # 标记缺失：-1 或 <=0 视为无效
    invalid_mask = df["worker_quality"] <= 0
    n_invalid = invalid_mask.sum()
    print(f"  [worker_quality] 缺失/无效值（<=0）: {n_invalid} / {len(df)} "
          f"({100*n_invalid/len(df):.1f}%)")

    if fill_na == "drop":
        df = df[~invalid_mask].copy()
        print(f"  [worker_quality] 丢弃后剩余: {len(df)}")
    elif fill_na in ("median", "mean"):
        valid_vals = df.loc[~invalid_mask, "worker_quality"]
        fill_val = valid_vals.median() if fill_na == "median" else valid_vals.mean()
        df.loc[invalid_mask, "worker_quality"] = fill_val
        print(f"  [worker_quality] 用 {fill_na} ({fill_val:.1f}) 填充缺失值")
    else:
        raise ValueError(f"Unknown fill_na strategy: {fill_na}")

    # 归一化到 [0, 1]
    df["worker_quality"] = df["worker_quality"] / 100.0
    df["worker_quality"] = df["worker_quality"].clip(0.0, 1.0)

    return df


# ============================================================
# 2. 时间字段解析
# ============================================================

def parse_time_series(time_series: pd.Series) -> pd.Series:
    """批量解析时间字符串为 pd.Timestamp，解析失败则返回 NaT。"""
    def _safe_parse(t):
        if pd.isna(t) or t == "":
            return pd.NaT
        try:
            return parse(str(t))
        except Exception:
            return pd.NaT
    return time_series.apply(_safe_parse)


def clean_entry_times(entries_df: pd.DataFrame) -> pd.DataFrame:
    """
    解析 entries_df 中的 entry_created_at 为 datetime 类型。
    丢弃无法解析时间的行。
    """
    df = entries_df.copy()
    df["entry_created_at"] = parse_time_series(df["entry_created_at"])
    before = len(df)
    df = df.dropna(subset=["entry_created_at"]).copy()
    after = len(df)
    if before > after:
        print(f"  [entry time] 丢弃无法解析时间的行: {before - after}")
    return df


# ============================================================
# 3. 项目过滤
# ============================================================

def filter_projects(projects_dict: Dict[int, dict]) -> Dict[int, dict]:
    """
    过滤项目：
    - 去掉 status 不是有效状态的项目（如 draft, cancelled 等）
    - 保留: "awarded", "completed", "active", "pending" 等
    - 确保有 start_date 和 deadline
    返回过滤后的 {project_id: project_dict}
    """
    valid_statuses = {"awarded", "completed", "active", "pending", "open",
                       "approval", "fulfilled"}
    filtered = {}
    removed_status = 0
    removed_no_date = 0

    for pid, proj in projects_dict.items():
        status = proj.get("status", "")
        if status not in valid_statuses:
            removed_status += 1
            continue
        if not proj.get("start_date") or not proj.get("deadline"):
            removed_no_date += 1
            continue
        # 解析时间
        try:
            proj["start_date_parsed"] = parse(proj["start_date"])
            proj["deadline_parsed"] = parse(proj["deadline"])
        except Exception:
            removed_no_date += 1
            continue
        filtered[pid] = proj

    print(f"  [projects] 状态无效: {removed_status}, 缺少日期: {removed_no_date}")
    print(f"  [projects] 过滤后保留: {len(filtered)} / {len(projects_dict)}")
    return filtered


# ============================================================
# 4. Worker ID 对齐
# ============================================================

def align_worker_ids(entries_df: pd.DataFrame,
                      worker_df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[int]]:
    """
    检查 entry 中的 worker_id 是否在 worker_df 中存在。
    对于不存在的 worker，将其 quality 设为中位数。
    返回:
        entries_df:   过滤后的 entries
        new_workers:  在 entries 中出现但不在 worker_df 中的 worker_id 集合
    """
    known_workers = set(worker_df["worker_id"].values)
    entry_workers = set(entries_df["worker"].unique())

    new_workers = entry_workers - known_workers
    missing_in_entry = known_workers - entry_workers

    print(f"  [worker alignment] entry 中出现的 worker: {len(entry_workers)}")
    print(f"  [worker alignment] worker_df 中的 worker: {len(known_workers)}")
    print(f"  [worker alignment] entry 中有但 worker_df 中无: {len(new_workers)}")
    print(f"  [worker alignment] worker_df 中有但 entry 中无: {len(missing_in_entry)}")

    # 移除 entry 中 worker == -1 的行（无效 worker）
    before = len(entries_df)
    entries_df = entries_df[entries_df["worker"] != -1].copy()
    print(f"  [worker alignment] 移除 worker=-1: {before - len(entries_df)}")

    return entries_df, new_workers


# ============================================================
# 5. 主清洗流程
# ============================================================

def run_cleaning(worker_df: pd.DataFrame,
                  entries_df: pd.DataFrame,
                  projects_dict: Dict[int, dict],
                  fill_na: str = "median") -> Tuple[pd.DataFrame,
                                                      pd.DataFrame,
                                                      Dict[int, dict],
                                                      Set[int]]:
    """
    执行完整的数据清洗流程。
    参数:
        worker_df:      原始 worker DataFrame
        entries_df:     原始 entries DataFrame
        projects_dict:  原始 projects 字典
        fill_na:        worker_quality 缺失值填充策略
    返回:
        worker_df_clean:    清洗后的 worker DataFrame
        entries_df_clean:   清洗后的 entries DataFrame
        projects_clean:     过滤后的 projects 字典
        new_workers:        新发现的 worker ID 集合
    """
    print("=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)

    # 2.1 清洗 worker quality
    print("\n[2.1] 清洗 worker_quality ...")
    worker_df_clean = clean_worker_quality(worker_df, fill_na=fill_na)

    # 2.2 解析 entry 时间
    print("\n[2.2] 解析 entry 时间 ...")
    entries_df_clean = clean_entry_times(entries_df)

    # 2.3 过滤项目
    print("\n[2.3] 过滤项目 ...")
    projects_clean = filter_projects(projects_dict)

    # 2.4 对齐 worker ID
    print("\n[2.4] 对齐 worker ID ...")
    entries_df_clean, new_workers = align_worker_ids(entries_df_clean,
                                                       worker_df_clean)

    # 2.5 只保留在 projects_clean 中的 entry
    clean_pids = set(projects_clean.keys())
    before = len(entries_df_clean)
    entries_df_clean = entries_df_clean[
        entries_df_clean["project_id"].isin(clean_pids)
    ].copy()
    print(f"\n  [project filter] 移除无效 project 的 entry: {before - len(entries_df_clean)}")

    # 2.6 去掉 entry_created_at 晚于 deadline 的异常记录
    deadline_map = {pid: p["deadline_parsed"] for pid, p in projects_clean.items()}
    before = len(entries_df_clean)

    def is_before_deadline(row):
        dl = deadline_map.get(row["project_id"])
        if dl is None or pd.isna(row["entry_created_at"]):
            return True  # 无法判断则保留
        return row["entry_created_at"] <= dl

    entries_df_clean = entries_df_clean[
        entries_df_clean.apply(is_before_deadline, axis=1)
    ].copy()
    print(f"  [deadline check] 移除超期 entry: {before - len(entries_df_clean)}")

    print(f"\n  === 清洗完成 ===")
    print(f"  Workers:  {len(worker_df_clean)}")
    print(f"  Projects: {len(projects_clean)}")
    print(f"  Entries:  {len(entries_df_clean)}")

    return worker_df_clean, entries_df_clean, projects_clean, new_workers
