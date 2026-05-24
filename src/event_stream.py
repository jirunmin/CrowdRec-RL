"""
强化学习事件流构造模块。
核心思路：
- 将所有 entry_created_at 视为 "worker 到达并参与某项目" 的事件
- 按时间排序，构造 (worker, project, time, reward_signal) 四元组
- 为每个事件生成负样本：同一时刻候选集中的其他项目（worker 看到但未选）

输出：
- events_df: 每个事件一行，包含 worker_id, project_id, timestamp, reward 等
- 包含正样本 (label=1) 和负样本 (label=0)
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_event_stream(entries_df: pd.DataFrame,
                        projects_dict: Dict[int, dict],
                        worker_df: pd.DataFrame,
                        project_df: pd.DataFrame,
                        global_start_time: Optional[pd.Timestamp] = None,
                        neg_sample_ratio: float = 4.0,
                        random_seed: int = 42) -> pd.DataFrame:
    """
    构造 RL 事件流。

    参数:
        entries_df:     清洗后的 entries，列包含 [project_id, worker, entry_created_at,
                         winner, finalist, score]
        projects_dict:  项目详情字典 {project_id: {...}}
        worker_df:      清洗后的 worker 特征 DataFrame
        project_df:     project 特征 DataFrame
        global_start_time: 全局起始时间（默认用最早 entry 时间）
        neg_sample_ratio:  负正样本比例（每个正样本配几个负样本）
        random_seed:       随机种子

    返回:
        events_df: 列:
            event_id         事件序号（按时间）
            worker            worker ID
            project_id       项目 ID
            timestamp         事件时间
            label             1=正样本(worker实际参与), 0=负样本(未参与)
            reward_worker     参与者视角 reward
            reward_requester  请求者视角 reward
            winner            是否中标（仅正样本有效）
            finalist          是否入围
            score             评分
            day_index         从首事件起的第几天
    """
    print("\n[4.1] 构造 RL 事件流 ...")
    np.random.seed(random_seed)

    # --- 1. 按时间排序正样本 ---
    pos = entries_df[["project_id", "worker", "entry_created_at",
                       "winner", "finalist", "score"]].copy()
    pos = pos.sort_values("entry_created_at").reset_index(drop=True)
    pos["label"] = 1
    if global_start_time is None:
        global_start_time = pos["entry_created_at"].min()

    # 计算 day_index
    pos["day_index"] = (pos["entry_created_at"] - global_start_time).dt.days

    print(f"  正样本数量: {len(pos)}")

    # --- 2. 构建每个时间点可用的 project 候选集 ---
    # 对于每个正样本，找出当时正在进行的 project（已开始且未过期）
    print("  构建候选集 ...")

    # 预处理：project_id -> (start_date, deadline)
    project_time_map = {}
    for pid, proj in projects_dict.items():
        sd = proj.get("start_date_parsed")
        dl = proj.get("deadline_parsed")
        if sd is not None and dl is not None:
            project_time_map[pid] = (sd, dl)

    valid_pids = set(project_time_map.keys())

    # --- 3. 为每个正样本采样负样本（高效版）---
    neg_samples = []
    worker_pos_projects = pos.groupby("worker")["project_id"].apply(set).to_dict()

    # 用 numpy 数组加速
    all_pids_arr = np.array(sorted(valid_pids))
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids_arr)}
    
    # 预计算每个 project 的时间范围
    sd_arr = np.array([
        project_time_map[p][0].timestamp() if p in project_time_map else 0
        for p in all_pids_arr
    ])
    dl_arr = np.array([
        project_time_map[p][1].timestamp() if p in project_time_map else 0
        for p in all_pids_arr
    ])

    # 批次处理: 每次处理 batch_size 个事件
    batch_size = 5000
    n_pos = len(pos)
    # 每个正样本多采样一些候选，再筛选
    sample_multiplier = max(int(neg_sample_ratio * 5), 10)

    for batch_start in tqdm(range(0, n_pos, batch_size), desc="  Generating negative samples"):
        batch_end = min(batch_start + batch_size, n_pos)
        batch = pos.iloc[batch_start:batch_end]
        
        # 向量化：获取批次中每个事件的时间戳
        t_arr = np.array([t.timestamp() for t in batch["entry_created_at"]])
        
        for i, (_, row) in enumerate(batch.iterrows()):
            t = row["entry_created_at"]
            t_ts = t_arr[i]
            w = row["worker"]
            chosen_pid = row["project_id"]
            done_set = worker_pos_projects.get(w, set())
            
            # 找到此时活跃的 project（用 numpy 向量化）
            active_mask = (sd_arr <= t_ts) & (dl_arr >= t_ts)
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) <= 1:
                continue
            
            # 随机采样候选（而不是遍历全部）
            n_sample = min(sample_multiplier, len(active_indices))
            sampled_indices = np.random.choice(active_indices, size=n_sample, replace=False)
            
            # 筛选：排除 chosen 和 worker 已做过的
            n_neg_added = 0
            for idx in sampled_indices:
                neg_pid = int(all_pids_arr[idx])
                if neg_pid == chosen_pid or neg_pid in done_set:
                    continue
                neg_samples.append({
                    "project_id": neg_pid,
                    "worker": w,
                    "entry_created_at": t,
                    "winner": False,
                    "finalist": False,
                    "score": 0,
                    "label": 0,
                    "day_index": (t - global_start_time).days,
                })
                n_neg_added += 1
                if n_neg_added >= neg_sample_ratio:
                    break

    neg_df = pd.DataFrame(neg_samples)
    print(f"  负样本数量: {len(neg_df)}")

    # --- 4. 合并正负样本 ---
    events_df = pd.concat([pos, neg_df], ignore_index=True)
    events_df = events_df.sort_values(["entry_created_at", "label"],
                                        ascending=[True, False]).reset_index(drop=True)
    events_df["event_id"] = range(len(events_df))
    events_df = events_df.rename(columns={"entry_created_at": "timestamp"})

    # --- 5. 计算 Reward ---
    print("  计算 reward ...")

    # Worker 视角 reward:
    # - 正样本: 基础 reward + quality bonus + winner bonus
    # - 负样本: 0
    w_quality_map = dict(zip(worker_df["worker_id"], worker_df["worker_quality"]))
    events_df["worker_quality"] = events_df["worker"].map(w_quality_map).fillna(0.5)

    events_df["reward_worker"] = 0.0
    mask_pos = events_df["label"] == 1
    events_df.loc[mask_pos, "reward_worker"] = (
        1.0  # 基础参与奖励
        + 0.5 * events_df.loc[mask_pos, "worker_quality"]  # quality bonus
        + 2.0 * events_df.loc[mask_pos, "winner"].astype(float)  # winner bonus
        + 1.0 * events_df.loc[mask_pos, "finalist"].astype(float)  # finalist bonus
    )

    # Requester 视角 reward:
    # - 正样本: 高质量 worker 参与 = 高 reward
    # - 负样本: 0
    events_df["reward_requester"] = 0.0
    events_df.loc[mask_pos, "reward_requester"] = (
        events_df.loc[mask_pos, "worker_quality"] * 2.0  # 质量贡献
        + 1.0 * events_df.loc[mask_pos, "winner"].astype(float)  # 选出优胜者
    )

    print(f"  事件总数: {len(events_df)}")
    print(f"  正样本: {mask_pos.sum()}, 负样本: {(~mask_pos).sum()}")
    print(f"  Worker reward 范围: [{events_df['reward_worker'].min():.2f}, "
          f"{events_df['reward_worker'].max():.2f}]")
    print(f"  Requester reward 范围: [{events_df['reward_requester'].min():.2f}, "
          f"{events_df['reward_requester'].max():.2f}]")

    return events_df


def build_training_samples(events_df: pd.DataFrame,
                            worker_features_df: pd.DataFrame,
                            project_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    将事件流与 worker/project 特征关联，生成可供模型训练的数据。
    使用向量化操作加速。
    """
    print("\n[4.2] 生成训练样本（关联特征）...")

    samples = events_df.copy()

    # --- 关联 worker 特征 ---
    wf = worker_features_df.set_index("worker_id")
    worker_feat_cols = [c for c in wf.columns]
    # Merge on worker
    samples = samples.merge(wf, left_on="worker", right_index=True,
                             how="left", suffixes=("", "_wf"))
    
    # --- 关联 project 特征 ---
    pf = project_features_df.set_index("project_id")
    project_feat_cols = [c for c in pf.columns]
    samples = samples.merge(pf, left_on="project_id", right_index=True,
                             how="left", suffixes=("", "_pf"))

    # --- 向量化计算匹配特征 ---
    # Category 匹配
    samples["match_category"] = (
        (samples["worker_pref_category"].fillna(-1) == 
         samples["project_category"].fillna(-2)) &
        (samples["worker_pref_category"].fillna(-1) != -1)
    ).astype(float)

    # Sub-category 匹配
    samples["match_sub_category"] = (
        (samples["worker_pref_sub_category"].fillna(-1) == 
         samples["project_sub_category"].fillna(-2)) &
        (samples["worker_pref_sub_category"].fillna(-1) != -1)
    ).astype(float)

    # Industry 匹配
    samples["match_industry"] = (
        (samples["worker_pref_industry"].fillna(-1) == 
         samples["project_industry_code"].fillna(-2)) &
        (samples["worker_pref_industry"].fillna(-1) != -1)
    ).astype(float)

    # Quality gap
    w_q = samples["worker_quality"].fillna(0.5).clip(0, 1)
    p_avg = samples["project_average_score"].fillna(0.0).clip(0, 5)
    samples["match_quality_gap"] = (w_q - p_avg / 5.0).abs()

    print(f"  训练样本: {len(samples)} rows, {len(samples.columns)} columns")
    return samples
