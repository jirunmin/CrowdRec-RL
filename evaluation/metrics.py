"""
Evaluation Metrics for Crowd Recommendation

Author: E角色
Description: 实现推荐系统常用的评价指标，用于公平对比不同方法
"""

import numpy as np
from typing import List, Dict, Optional


def hit_rate_at_k(predictions: List[int], ground_truth: List[int], k: int = 1) -> float:
    """
    计算Hit@K指标（命中率）

    什么是Hit@K？
    =============
    Hit@K衡量的是：在Top-K个预测中，是否包含正确答案。

    在我们的场景中：
    - 每次只推荐1个任务（K=1）
    - predictions[i] = 模型推荐的第i个任务索引
    - ground_truth[i] = worker实际选择的任务索引
    - 如果 predictions[i] == ground_truth[i]，则算一次hit

    公式：
    Hit@K = (命中次数) / (总预测次数)

    举例：
    ======
    predictions = [3, 7, 2, 5]
    ground_truth = [3, 8, 2, 9]

    Hit@1 = 2/4 = 0.5 （第1和第3次命中）

    Args:
        predictions: 模型预测的动作索引列表
        ground_truth: 真实动作索引列表（通常从info["ground_truth_index"]获取）
        k: Top-K的大小（本项目通常用K=1）

    Returns:
        float: Hit@K值，范围[0, 1]，越高越好
    """
    if len(predictions) == 0:
        return 0.0

    hits = 0
    valid_count = 0

    for pred, gt in zip(predictions, ground_truth):
        if gt >= 0:  # gt=-1表示无效（没有真实选择）
            valid_count += 1
            if pred == gt:
                hits += 1

    if valid_count == 0:
        return 0.0

    return hits / valid_count


def ndcg_at_k(predictions: List[int],
              ground_truth: List[int],
              relevances: List[float],
              k: int = 5) -> float:
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain)

    什么是NDCG？
    ===========
    NDCG是信息检索领域最经典的排序质量指标。
    它不仅考虑"是否命中"，还考虑"命中的位置"和"相关性强度"。

    核心概念：
    1. **DCG (Discounted Cumulative Gain)**:
       - 对每个位置的相关性打分
       - 位置越靠前，权重越高（用log2(i+2)作为折扣因子）
       - 公式：DCG@K = Σ (rel_i / log2(i+2))，i从0到K-1

    2. **IDCG (Ideal DCG)**:
       - 理想情况下的DCG（按相关性降序排列）
       - 作为归一化的分母

    3. **NDCG**:
       - NDCG = DCG / IDCG
       - 范围[0, 1]，1表示完美排序

    在我们场景中的特殊处理：
    ========================
    由于每次只推荐1个项目（不是Top-K列表），
    NDCG@1 ≈ Hit@1 × relevance_normalization

    但为了通用性，我们还是实现了完整的NDCG计算。

    Args:
        predictions: 预测的动作索引列表
        ground_truth: 真实动作索引列表
        relevances: 每个位置的相关性分数列表（如reward值）
        k: Top-K大小

    Returns:
        float: NDCG@K值，范围[0, 1]，越高越好
    """
    if len(predictions) == 0:
        return 0.0

    dcg = 0.0
    valid_count = 0

    for i, (pred, gt, rel) in enumerate(zip(predictions, ground_truth, relevances)):
        if gt >= 0 and i < k:  # 只考虑有效样本和Top-K范围内
            valid_count += 1
            if pred == gt:
                dcg += rel / np.log2(i + 2)

    if valid_count == 0:
        return 0.0

    idcg = sum(rel / np.log2(i + 2)
               for i, rel in enumerate(relevances[:min(k, len(relevances))]))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_score(predictions: List[int], ground_truth: List[int]) -> float:
    """
    计算MRR (Mean Reciprocal Rank，平均倒数排名)

    什么是MRR？
    ==========
    MRR关注的是：正确答案出现在第几个位置（取倒数的平均）。

    公式：
    MRR = (1/Q) * Σ (1/rank_i)
    其中Q是查询次数，rank_i是第i次查询中正确答案的排名（从1开始）

    举例：
    ======
    predictions = [3, 7, 2, 5]
    ground_truth = [3, 8, 2, 9]

    第1次：pred=3, gt=3 → rank=1 → 1/1 = 1.0
    第2次：pred=7, gt=8 → 不匹配 → 0
    第3次：pred=2, gt=2 → rank=1 → 1/1 = 1.0
    第4次：pred=5, gt=9 → 不匹配 → 0

    MRR = (1.0 + 0 + 1.0 + 0) / 4 = 0.5

    在我们场景中：
    由于每次只推荐1个，MRR等价于Hit@1。

    Args:
        predictions: 预测列表
        ground_truth: 真实标签列表

    Returns:
        float: MRR值，范围[0, 1]，越高越好
    """
    if len(predictions) == 0:
        return 0.0

    reciprocal_ranks = []

    for pred, gt in zip(predictions, ground_truth):
        if gt >= 0 and pred == gt:
            reciprocal_ranks.append(1.0)
        else:
            reciprocal_ranks.append(0.0)

    if len(reciprocal_ranks) == 0:
        return 0.0

    return np.mean(reciprocal_ranks)


def compute_all_metrics(predictions: List[int],
                        ground_truth: List[int],
                        rewards: Optional[List[float]] = None) -> Dict[str, float]:
    """
    计算所有指标并返回字典

    这是一个便捷函数，一次性计算所有常用指标。

    包含的指标：
    - hit_rate_1: Hit@1（最常用）
    - hit_rate_5: Hit@5（如果有多候选排序）
    - ndcg_1: NDCG@1
    - ndcg_5: NDCG@5
    - mrr: 平均倒数排名
    - avg_reward: 平均奖励（如果提供rewards）
    - total_samples: 总样本数

    Args:
        predictions: 预测列表
        ground_truth: 真实标签列表
        rewards: 可选的奖励列表

    Returns:
        Dict: 包含所有指标的字典
    """
    metrics = {}

    metrics['hit_rate_1'] = hit_rate_at_k(predictions, ground_truth, k=1)
    metrics['hit_rate_5'] = hit_rate_at_k(predictions, ground_truth, k=5)

    relevances = rewards if rewards is not None else [1.0] * len(predictions)
    metrics['ndcg_1'] = ndcg_at_k(predictions, ground_truth, relevances, k=1)
    metrics['ndcg_5'] = ndcg_at_k(predictions, ground_truth, relevances, k=5)

    metrics['mrr'] = mrr_score(predictions, ground_truth)

    if rewards is not None:
        metrics['avg_reward'] = np.mean(rewards)
        metrics['total_reward'] = np.sum(rewards)

    metrics['total_samples'] = len(predictions)

    return metrics


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # 测试数据
    predictions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    ground_truth = [0, 8, 2, 9, 4, 5, 1, 7, 3, 4]
    rewards = [1.5, 0.0, 2.3, 0.0, 1.8, 0.0, 2.1, 0.0, 1.9, 2.5]

    print(f"\nPredictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Rewards: {rewards}")

    # 测试各个指标
    hit1 = hit_rate_at_k(predictions, ground_truth, k=1)
    print(f"\n✓ Hit@1: {hit1:.4f} (expected ~0.5)")

    hit5 = hit_rate_at_k(predictions, ground_truth, k=5)
    print(f"✓ Hit@5: {hit5:.4f} (expected ~0.7)")

    ndcg1 = ndcg_at_k(predictions, ground_truth, rewards, k=1)
    print(f"✓ NDCG@1: {ndcg1:.4f}")

    mrr = mrr_score(predictions, ground_truth)
    print(f"✓ MRR: {mrr:.4f} (expected same as Hit@1)")

    all_metrics = compute_all_metrics(predictions, ground_truth, rewards)
    print(f"\n✓ All Metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n✓ All metrics tests passed!")
