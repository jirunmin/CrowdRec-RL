"""
数据加载模块：读取所有原始数据文件。
数据关系说明：
  - entry/ 目录包含 1499 个项目的 worker 参与记录
  - project_list.csv 包含 2501 个项目（几乎与 entry 不重叠）
  - project/ 目录包含 5335 个项目详情文件
  策略：以 entry 文件驱动，发现有交互数据的项目，再从 project/ 读取其详情。
  project_list.csv 中的项目作为额外的候选池（无交互历史）。
"""

import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm


# ============================================================
# 路径配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_QUALITY_PATH = os.path.join(BASE_DIR, "worker_quality.csv")
PROJECT_LIST_PATH = os.path.join(BASE_DIR, "project_list.csv")
PROJECT_DIR = os.path.join(BASE_DIR, "project")
ENTRY_DIR = os.path.join(BASE_DIR, "entry")


# ============================================================
# 1. Worker Quality
# ============================================================

def load_worker_quality(path: str = WORKER_QUALITY_PATH) -> pd.DataFrame:
    """读取 worker_quality.csv"""
    df = pd.read_csv(path)
    df.columns = ["worker_id", "worker_quality"]
    return df


# ============================================================
# 2. 扫描 entry 目录，发现所有项目
# ============================================================

def discover_projects_from_entries(entry_dir: str = ENTRY_DIR) -> Dict[int, int]:
    """
    扫描 entry/ 目录，发现所有有 entry 数据的项目。
    解析文件名 entry_{project_id}_{offset}.txt
    返回 {project_id: entry_count}（从文件推算的总回答数）
    """
    pattern = re.compile(r"entry_(\d+)_(\d+)\.txt$")
    project_offsets = defaultdict(set)

    for fname in os.listdir(entry_dir):
        m = pattern.match(fname)
        if m:
            pid = int(m.group(1))
            offset = int(m.group(2))
            project_offsets[pid].add(offset)

    # 推算 entry_count：假设分页每隔 24 条，最大 offset + 24（估算）
    project_entry_counts = {}
    for pid, offsets in project_offsets.items():
        max_offset = max(offsets)
        # 估算：如果 max_offset=0 且文件存在，至少有 1 条；否则按 pagesize=24 估算
        page_size = 24
        estimated_count = max_offset + page_size  # 上界估计
        project_entry_counts[pid] = estimated_count

    print(f"  从 entry 目录发现 {len(project_entry_counts)} 个项目")
    return project_entry_counts


# ============================================================
# 3. Project Details (JSON)
# ============================================================

def load_single_project(project_id: int) -> Optional[dict]:
    """读取单个 project_{id}.txt"""
    filepath = os.path.join(PROJECT_DIR, f"project_{project_id}.txt")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_projects_by_ids(project_ids: List[int]) -> Dict[int, dict]:
    """批量读取项目详情，跳过不存在的文件"""
    projects = {}
    for pid in tqdm(project_ids, desc="Loading projects"):
        proj = load_single_project(pid)
        if proj is not None:
            projects[pid] = proj
    return projects


# ============================================================
# 4. Entry Records (分页 JSON)
# ============================================================

def load_entries_for_project(project_id: int,
                              max_pages: int = 200) -> List[dict]:
    """
    读取某个项目的所有 entry 分页文件（自动发现分页）。
    参数:
        project_id:  项目 ID
        max_pages:   最大分页数（安全上限）
    返回:
        list[dict]: 所有回答记录
    """
    all_entries = []
    page_size = 24
    for page in range(max_pages):
        offset = page * page_size
        filepath = os.path.join(ENTRY_DIR, f"entry_{project_id}_{offset}.txt")
        if not os.path.exists(filepath):
            break
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        for item in results:
            all_entries.append({
                "project_id": project_id,
                "worker": int(item.get("author", -1)),
                "entry_number": int(item.get("entry_number", -1)),
                "entry_created_at": item.get("entry_created_at", None),
                "winner": item.get("winner", False),
                "finalist": item.get("finalist", False),
                "score": (item.get("revisions", [{}]) or [{}])[0].get("score", 0)
                          if item.get("revisions") else 0,
            })
    return all_entries


def load_entries_for_projects(project_ids: List[int]) -> pd.DataFrame:
    """批量读取多个项目的 entry 记录"""
    all_rows = []
    for pid in tqdm(project_ids, desc="Loading entries"):
        entries = load_entries_for_project(pid)
        all_rows.extend(entries)
    return pd.DataFrame(all_rows)


# ============================================================
# 5. 辅助：industry 编码
# ============================================================

def build_industry_mapping(projects: Dict[int, dict]) -> Dict[str, int]:
    """为每个 industry 字符串分配整数编码"""
    industry_map = {}
    for proj in projects.values():
        ind = proj.get("industry", "")
        if ind and ind not in industry_map:
            industry_map[ind] = len(industry_map)
    return industry_map


# ============================================================
# 6. 一次性全部加载（推荐入口）
# ============================================================

def load_all_data(verbose: bool = True,
                   use_project_list: bool = True) -> Tuple[pd.DataFrame,
                                                              pd.DataFrame,
                                                              Dict[int, dict],
                                                              Dict[str, int],
                                                              Set[int]]:
    """
    主加载函数。
    策略:
      1. 扫描 entry/ 发现所有有交互的项目 → entry_project_ids
      2. 加载这些项目的详情 + 加载 project_list.csv 中项目的详情
      3. 加载所有 entry 记录
      4. 合并为统一数据集

    返回:
        worker_df:       [worker_id, worker_quality]
        entries_df:      [project_id, worker, entry_number, entry_created_at,
                          winner, finalist, score]
        projects_dict:   {project_id: project_json}  所有可用项目
        industry_map:    {industry_name: industry_code}
        entry_pids:      有 entry 数据的项目 ID 集合
    """
    if verbose:
        print("=" * 60)
        print("Step 1: 加载数据")
        print("=" * 60)

    # 1. Worker quality
    if verbose:
        print("[1/5] Loading worker_quality.csv ...")
    worker_df = load_worker_quality()
    if verbose:
        print(f"       -> {len(worker_df)} workers")

    # 2. 从 entry 目录发现项目
    if verbose:
        print("[2/5] Scanning entry directory for projects ...")
    entry_project_counts = discover_projects_from_entries()
    entry_pids = set(entry_project_counts.keys())

    # 3. 从 project_list.csv 读取项目
    if verbose:
        print("[3/5] Loading project_list.csv ...")
    project_list_df = pd.read_csv(PROJECT_LIST_PATH, header=None,
                                    names=["project_id", "entry_count"])
    csv_pids = set(project_list_df["project_id"].tolist())
    if verbose:
        print(f"       -> {len(csv_pids)} projects in project_list.csv")

    # 4. 合并需要加载的项目 ID
    all_project_ids = list(entry_pids | csv_pids)
    if verbose:
        print(f"[4/5] Loading {len(all_project_ids)} project detail files ...")
        print(f"       (entry projects: {len(entry_pids)}, "
              f"csv projects: {len(csv_pids)}, "
              f"overlap: {len(entry_pids & csv_pids)})")

    projects_dict = load_projects_by_ids(all_project_ids)
    industry_map = build_industry_mapping(projects_dict)
    if verbose:
        print(f"       -> {len(projects_dict)} project files loaded")
        print(f"       -> {len(industry_map)} unique industries")

    # 5. 加载 entry 记录（仅对有 entry 的项目）
    if verbose:
        print(f"[5/5] Loading entries for {len(entry_pids)} projects ...")
    entries_df = load_entries_for_projects(list(entry_pids))
    if verbose:
        print(f"       -> {len(entries_df)} entry records loaded")

    return worker_df, entries_df, projects_dict, industry_map, entry_pids

