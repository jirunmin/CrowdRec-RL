"""
Greedy 策略基类：使用全部 10 个可用特征，线性加权打分。

所有特征在决策时都可观测，不做人为子集划分。
GreedyWorkerPolicy 和 GreedyRequesterPolicy 继承此类，只改变默认权重。
"""

import numpy as np
from typing import Dict, Any, Optional


class FeatureIdx:
    """候选任务特征向量的索引定义（14维）"""
    # Project特征 (10维)
    CATEGORY = 0
    SUB_CATEGORY = 1
    INDUSTRY = 2
    ENTRY_COUNT = 3
    TOTAL_AWARDS = 4
    DURATION = 5
    IS_FEATURED = 6
    AVERAGE_SCORE = 7
    WORKER_COUNT = 8
    HAS_WINNER = 9
    # Match特征 (4维)
    MATCH_CATEGORY = 10
    MATCH_SUB_CATEGORY = 11
    MATCH_INDUSTRY = 12
    QUALITY_GAP = 13


# 全部 10 个可用特征及其索引
ALL_FEATURES = [
    ("awards",       FeatureIdx.TOTAL_AWARDS),
    ("match_cat",    FeatureIdx.MATCH_CATEGORY),
    ("match_sub",    FeatureIdx.MATCH_SUB_CATEGORY),
    ("match_ind",    FeatureIdx.MATCH_INDUSTRY),
    ("quality_gap",  FeatureIdx.QUALITY_GAP),
    ("featured",     FeatureIdx.IS_FEATURED),
    ("avg_score",    FeatureIdx.AVERAGE_SCORE),
    ("has_winner",   FeatureIdx.HAS_WINNER),
    ("worker_count", FeatureIdx.WORKER_COUNT),
    ("entry_count",  FeatureIdx.ENTRY_COUNT),
]


class GreedyPolicy:
    """
    统一的 Greedy 策略：score = w^T · features

    使用全部 10 个可观测特征，线性加权。
    子类只需覆盖 default_weights() 提供不同的默认权重。
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, seed: Optional[int] = None):
        if weights is None:
            weights = self.default_weights()
        self.weights = weights
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def default_weights() -> Dict[str, float]:
        """子类应覆盖此方法。"""
        return {name: 0.0 for name, _ in ALL_FEATURES}

    def _compute_score(self, candidate: np.ndarray) -> float:
        score = 0.0
        for name, idx in ALL_FEATURES:
            score += self.weights.get(name, 0.0) * candidate[idx]
        return score

    def select_action(self, obs: Dict[str, Any]) -> int:
        candidates = obs["candidate_state"]
        valid_mask = obs["valid_mask"]

        best_score = -float('inf')
        best_actions = []

        for i in range(len(candidates)):
            if valid_mask[i]:
                score = self._compute_score(candidates[i])
                if score > best_score:
                    best_score = score
                    best_actions = [i]
                elif abs(score - best_score) < 1e-6:
                    best_actions.append(i)

        if len(best_actions) == 0:
            return 0
        return int(self.rng.choice(best_actions))

    def __repr__(self):
        w = ", ".join(f"{k}={v:.4f}" for k, v in self.weights.items())
        return f"{self.__class__.__name__}({w})"
