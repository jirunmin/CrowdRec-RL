"""
Greedy-Requester Baseline：使用全部 10 个特征，权重由 LR 在 requester reward 上学习得到。

用法：
  policy = GreedyRequesterPolicy(seed=42)
  action = policy.select_action(obs)
"""

from typing import Optional
from baselines.greedy_base import GreedyPolicy, FeatureIdx


class GreedyRequesterPolicy(GreedyPolicy):
    """贪心策略，默认权重针对 requester reward 优化。"""

    @staticmethod
    def default_weights():
        # LR 在 requester reward 上学到的权重（运行 baselines/find_weights_max_reward.py 获取）
        # 注意：quality_match → quality_gap 需要取反
        return {
            "awards":       0.00003164,
            "match_cat":    0.26436990,
            "match_sub":    0.12628057,
            "match_ind":    0.14821165,
            "quality_gap":  0.71283859,   # 原值 quality_match=-0.7128，取反
            "featured":     0.05883334,
            "avg_score":    0.16107203,
            "has_winner":   0.10417320,
            "worker_count": 0.00258450,
            "entry_count":  0.00065133,
        }


def create_greedy_requester_policy(seed: int = 42) -> GreedyRequesterPolicy:
    return GreedyRequesterPolicy(seed=seed)


if __name__ == "__main__":
    import numpy as np
    print("Testing GreedyRequesterPolicy...")

    policy = GreedyRequesterPolicy(seed=42)
    print(f"Created policy: {policy}")

    mock_obs = {
        "worker_state": np.random.randn(12),
        "candidate_state": np.random.randn(20, 14),
        "valid_mask": np.array([True]*18 + [False]*2),
        "info": {}
    }

    action = policy.select_action(mock_obs)
    print(f"Selected action: {action} (should be in range [0, 17])")
    print("✓ GreedyRequesterPolicy test passed!")
