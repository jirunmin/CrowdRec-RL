"""
数据集划分与候选生成模块。
- 按时间顺序划分 train/val/test（严格时序，防止数据泄露）
- 为每个 worker 到达时刻生成 Top-K 候选项目集合
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. 时序划分
# ============================================================

def split_by_time(events_df: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame]:
    """
    按时间顺序将事件流划分为 train / val / test。
    不随机 shuffle，严格按 timestamp 顺序切分。

    参数:
        events_df:   事件流 DataFrame，必须有 timestamp 列
        train_ratio: 训练集比例
        val_ratio:   验证集比例
        test_ratio:  测试集比例
    返回:
        train_df, val_df, test_df
    """
    print("\n[5.1] 时序划分数据集 ...")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "ratios must sum to 1.0"

    df = events_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"  总事件数: {n}")
    print(f"  Train: {len(train_df)} ({100*len(train_df)/n:.1f}%)")
    print(f"  Val:   {len(val_df)} ({100*len(val_df)/n:.1f}%)")
    print(f"  Test:  {len(test_df)} ({100*len(test_df)/n:.1f}%)")

    # 打印时间范围
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        t_min = subset["timestamp"].min()
        t_max = subset["timestamp"].max()
        print(f"  {name} time range: {t_min} ~ {t_max}")

    return train_df, val_df, test_df


# ============================================================
# 2. Top-K 候选项目生成（numpy 向量化版）
# ============================================================

# 全局缓存：避免重复构建 numpy 数组
_cached_project_arrays = None

def _build_project_arrays(projects_dict: Dict[int, dict]):
    """构建 numpy 数组用于向量化候选计算"""
    pids = sorted(projects_dict.keys())
    n = len(pids)
    
    sd_arr = np.zeros(n)
    dl_arr = np.zeros(n)
    featured_arr = np.zeros(n)
    awards_arr = np.zeros(n)
    pid_arr = np.array(pids, dtype=np.int64)
    
    for i, pid in enumerate(pids):
        proj = projects_dict[pid]
        sd = proj.get("start_date_parsed")
        dl = proj.get("deadline_parsed")
        sd_arr[i] = sd.timestamp() if sd else 0
        dl_arr[i] = dl.timestamp() if dl else 0
        featured_arr[i] = 1.0 if proj.get("featured", False) else 0.0
        awards_arr[i] = float(proj.get("total_awards", 0) or 0)
    
    return {
        "pids": pid_arr,
        "sd": sd_arr,
        "dl": dl_arr,
        "featured": featured_arr,
        "awards": awards_arr,
    }


def generate_candidates_fast(projects_dict: Dict[int, dict],
                               worker_id: int,
                               timestamp: pd.Timestamp,
                               worker_done_projects: set,
                               top_k: int = 20) -> List[int]:
    """
    向量化版：为给定 worker 在给定时间点生成 Top-K 候选项目。
    使用 numpy 数组加速，比纯 Python 循环快 50-100x。
    """
    global _cached_project_arrays
    if _cached_project_arrays is None:
        _cached_project_arrays = _build_project_arrays(projects_dict)
    
    arr = _cached_project_arrays
    t_ts = timestamp.timestamp()
    
    # 向量化过滤：活跃 + 未过期
    active_mask = (arr["sd"] <= t_ts) & (arr["dl"] >= t_ts)
    active_idx = np.where(active_mask)[0]
    
    if len(active_idx) <= 1:
        return []
    
    # 排除 worker 已参与的项目
    pids_arr = arr["pids"][active_idx]
    done_mask = np.array([pid in worker_done_projects for pid in pids_arr])
    candidate_idx = active_idx[~done_mask]
    
    if len(candidate_idx) == 0:
        return []
    
    # 向量化计算得分
    c_sd = arr["sd"][candidate_idx]
    c_dl = arr["dl"][candidate_idx]
    c_pids = arr["pids"][candidate_idx]
    
    duration = np.maximum(c_dl - c_sd, 1.0)
    remaining = np.maximum(c_dl - t_ts, 0.0)
    urgency = 1.0 - remaining / duration  # 越临近截止越高
    featured = arr["featured"][candidate_idx]
    awards_score = np.log1p(arr["awards"][candidate_idx]) / 10.0
    
    scores = urgency + featured + awards_score
    
    # 取 top-k
    if len(scores) <= top_k:
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    return c_pids[top_indices].tolist()


def generate_candidates_batch(events_df: pd.DataFrame,
                                projects_dict: Dict[int, dict],
                                top_k: int = 20) -> pd.DataFrame:
    """
    为事件流中的每个正样本事件生成候选项目列表。
    """
    print(f"\n[5.2] 生成 Top-{top_k} 候选项目集合 ...")

    pos_events = events_df[events_df["label"] == 1].sort_values("timestamp").copy()
    
    # 预热缓存
    global _cached_project_arrays
    _cached_project_arrays = _build_project_arrays(projects_dict)
    
    worker_done = {}
    candidate_lists = []
    
    for _, row in tqdm(pos_events.iterrows(), total=len(pos_events),
                        desc="  Generating candidates"):
        w = row["worker"]
        t = row["timestamp"]
        pid = row["project_id"]
        
        done_set = worker_done.get(w, set())
        candidates = generate_candidates_fast(
            projects_dict, w, t, done_set, top_k=top_k
        )
        candidate_lists.append(candidates)
        
        if w not in worker_done:
            worker_done[w] = set()
        worker_done[w].add(pid)
    
    pos_events["candidate_projects"] = candidate_lists
    cand_counts = pos_events["candidate_projects"].apply(len)
    print(f"  候选数统计: min={cand_counts.min()}, mean={cand_counts.mean():.1f}, "
          f"median={cand_counts.median():.0f}, max={cand_counts.max()}")
    
    return pos_events


# ============================================================
# 3. 保存处理后的数据
# ============================================================

def save_processed_data(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         worker_features_df: pd.DataFrame,
                         project_features_df: pd.DataFrame,
                         output_dir: str = "./processed"):
    """保存所有处理后的数据为 parquet 文件。"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[5.3] 保存处理后的数据到 {output_dir}/ ...")

    train_df.to_parquet(f"{output_dir}/train_events.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val_events.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test_events.parquet", index=False)
    worker_features_df.to_parquet(f"{output_dir}/worker_features.parquet", index=False)
    project_features_df.to_parquet(f"{output_dir}/project_features.parquet", index=False)

    # 保存统计信息
    stats = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_workers": len(worker_features_df),
        "n_projects": len(project_features_df),
    }
    pd.Series(stats).to_json(f"{output_dir}/stats.json")
    print(f"  保存完成. stats: {stats}")


    os.makedirs(output_dir, exist_ok=True)
