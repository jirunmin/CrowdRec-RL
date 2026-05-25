#!/usr/bin/env python3
"""
众包推荐 RL 项目 —— 数据预处理主流程。
运行顺序:
  1. 加载数据 (data_loader)
  2. 清洗数据 (preprocessing)
  3. 特征工程 (features)
  4. 构造 RL 事件流 (event_stream)
  5. 划分数据集 & 候选生成 (dataset)
  6. 保存处理结果

用法:
  python main_preprocess.py [--output_dir processed] [--neg_ratio 4.0] [--top_k 20]
"""

import argparse
import os
import sys

import pandas as pd

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_data
from src.preprocessing import run_cleaning
from src.features import (build_project_features,
                           build_worker_features)
from src.quality_predictor import (train_quality_predictor,
                                    predict_quality)
from src.event_stream import build_event_stream, build_training_samples
from src.dataset import (generate_candidates_batch,
                          save_processed_data,
                          split_by_time)


def main():
    parser = argparse.ArgumentParser(description="CrowdRec-RL 数据预处理")
    parser.add_argument("--output_dir", type=str, default="processed",
                        help="输出目录")
    parser.add_argument("--neg_ratio", type=float, default=4.0,
                        help="负正样本比例")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-K 候选数")
    parser.add_argument("--fill_na", type=str, default="median",
                        choices=["median", "mean", "drop"],
                        help="Worker quality 缺失值处理策略")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--skip_candidates", action="store_true",
                        help="跳过候选集生成（加速预处理）")
    parser.add_argument("--quality_mode", type=str, default="predict",
                        choices=["median", "predict"],
                        help="Worker quality 模式: median=中位数填充, predict=ML预测")
    args = parser.parse_args()

    # ============================================================
    # Step 1: 加载数据
    # ============================================================
    worker_df, entries_df, projects_dict, industry_map, entry_pids = \
        load_all_data(verbose=True)

    # ============================================================
    # Step 2: 清洗数据
    # ============================================================
    worker_df, entries_df, projects_dict, new_workers = run_cleaning(
        worker_df, entries_df, projects_dict, fill_na=args.fill_na
    )

    # 2.5: 扩展 worker_df，纳入 entry 中出现但 worker_quality.csv 中没有的 worker
    if len(new_workers) > 0:
        print(f"\n[2.5] 扩展 worker_df：添加 {len(new_workers)} 个 entry 中出现的新 worker ...")
        new_rows = pd.DataFrame({
            "worker_id": list(new_workers),
            "worker_quality": [-1.0] * len(new_workers),   # -1 标记为待预测
            "worker_quality_raw": [-1.0] * len(new_workers),
        })
        worker_df = pd.concat([worker_df, new_rows], ignore_index=True)
        print(f"  worker_df 总行数: {len(worker_df)} (原始 1807 + 新增 {len(new_workers)})")

    # ============================================================
    # Step 3: 特征工程
    # ============================================================
    global_start_time = entries_df["entry_created_at"].min()

    worker_features_df = build_worker_features(
        worker_df, entries_df, projects_dict, industry_map
    )
    project_features_df = build_project_features(
        projects_dict, entries_df, industry_map, global_start_time
    )

    # ============================================================
    # Step 3.5: Worker Quality 预测（可选）
    # ============================================================
    predictor = None
    if args.quality_mode == "predict":
        predictor = train_quality_predictor(worker_features_df, verbose=True)
        worker_features_df = predict_quality(predictor, worker_features_df)
        # 用预测值覆盖 worker_df 中的 quality（仅对原本无标签的 worker）
        pred_quality_map = dict(zip(worker_features_df["worker_id"],
                                     worker_features_df["worker_quality_pred"]))
        # 记录哪些 worker 原本有标签
        orig_has_label = worker_features_df["worker_quality_raw"] > 0
        has_label_map = dict(zip(worker_features_df["worker_id"], orig_has_label))
        # 对有标签 worker 保留原值，对无标签 worker 用预测值
        def _get_quality(wid):
            if has_label_map.get(wid, False):
                return worker_df.loc[worker_df["worker_id"] == wid, "worker_quality"].values[0]
            return pred_quality_map.get(wid, 0.5)
        worker_df["worker_quality"] = worker_df["worker_id"].apply(_get_quality)
        print(f"  Using ML-predicted quality for rewards")
    else:
        print(f"  Using median-filled quality for rewards")

    # ============================================================
    # Step 4: 构造 RL 事件流
    # ============================================================
    events_df = build_event_stream(
        entries_df=entries_df,
        projects_dict=projects_dict,
        worker_df=worker_df,
        project_df=project_features_df,
        global_start_time=global_start_time,
        neg_sample_ratio=args.neg_ratio,
    )

    # 关联特征生成训练样本
    training_samples_df = build_training_samples(
        events_df, worker_features_df, project_features_df
    )

    # ============================================================
    # Step 5: 划分数据集
    # ============================================================
    train_df, val_df, test_df = split_by_time(
        training_samples_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # ============================================================
    # Step 6: 生成候选集（高速 numpy 版）& 保存
    # ============================================================
    if not args.skip_candidates:
        pos_events_with_candidates = generate_candidates_batch(
            train_df, projects_dict, top_k=args.top_k
        )

    save_processed_data(
        train_df, val_df, test_df,
        worker_features_df, project_features_df,
        output_dir=args.output_dir
    )

    if not args.skip_candidates:
        cand_path = os.path.join(args.output_dir, "candidates.parquet")
        pos_events_with_candidates.to_parquet(cand_path, index=False)

    print("\n" + "=" * 60)
    print("预处理完成！输出文件:")
    print(f"  {args.output_dir}/train_events.parquet")
    print(f"  {args.output_dir}/val_events.parquet")
    print(f"  {args.output_dir}/test_events.parquet")
    print(f"  {args.output_dir}/worker_features.parquet")
    print(f"  {args.output_dir}/project_features.parquet")
    if not args.skip_candidates:
        print(f"  {args.output_dir}/candidates.parquet")
    print(f"  {args.output_dir}/stats.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
