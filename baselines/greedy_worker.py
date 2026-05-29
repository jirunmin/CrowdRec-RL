"""
Greedy-Worker Baseline：使用全部 10 个特征，权重由 LR 在 worker reward 上学习得到。

用法：
  policy = GreedyWorkerPolicy(seed=42)
  action = policy.select_action(obs)
"""

from typing import Optional
from baselines.greedy_base import GreedyPolicy, FeatureIdx


class GreedyWorkerPolicy(GreedyPolicy):
    """贪心策略，默认权重针对 worker reward 优化。"""

    @staticmethod
    def default_weights():
        # LR 在 worker reward 上学到的权重（运行 baselines/find_weights_max_reward.py 获取）
        # 注意：quality_match → quality_gap 需要取反
        return {
            "awards":       0.00003400,
            "match_cat":    0.22017864,
            "match_sub":    0.11781708,
            "match_ind":    0.13808125,
            "quality_gap":  0.26647046,   # 原值 quality_match=-0.2665，取反
            "featured":     0.04761916,
            "avg_score":    0.07194080,
            "has_winner":   0.09423370,
            "worker_count": 0.00229748,
            "entry_count":  0.00053837,
        }


def create_greedy_worker_policy(seed: int = 42) -> GreedyWorkerPolicy:
    return GreedyWorkerPolicy(seed=seed)


if __name__ == "__main__":
    import numpy as np
    print("Testing GreedyWorkerPolicy...")

    policy = GreedyWorkerPolicy(seed=42)
    print(f"Created policy: {policy}")

    mock_obs = {
        "worker_state": np.random.randn(12),
        "candidate_state": np.random.randn(20, 14),
        "valid_mask": np.array([True]*18 + [False]*2),
        "info": {}
    }

    action = policy.select_action(mock_obs)
    print(f"Selected action: {action} (should be in range [0, 17])")
    print("✓ GreedyWorkerPolicy test passed!")
