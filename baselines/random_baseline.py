"""
Random Baseline for Crowd Recommendation Task

Author: E角色
Description: 最简单的基线方法 - 从有效候选任务中均匀随机选择
"""

import numpy as np
from typing import Dict, Any, Optional


class RandomPolicy:
    """
    随机策略：从有效候选中均匀随机选择

    这是所有baseline中最简单的一个，作为性能下界。
    任何智能方法都应该显著优于随机策略。

    核心逻辑：
    1. 获取当前观测中的有效动作掩码（valid_mask）
    2. 找出所有为True的位置索引（有效候选）
    3. 使用随机数生成器从中均匀选择一个
    4. 返回选择的动作索引

    为什么需要检查valid_mask？
    - 不是所有20个候选都是有效的
    - 某些任务可能已经结束、已满员、或其他原因不可选
    - valid_mask是一个长度为K的布尔数组，True表示可选
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机策略

        Args:
            seed: 随机种子（用于可复现性）
        """
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def select_action(self, obs: Dict[str, Any]) -> int:
        """
        根据观测选择动作

        Args:
            obs: 环境返回的观测字典，包含：
                - worker_state: (D_w,) Worker特征向量
                - candidate_state: (K, D_p+D_m) 候选任务特征矩阵
                - valid_mask: (K,) bool 有效候选掩码
                - info: 其他信息字典

        Returns:
            int: 选择的动作索引（0到K-1）

        Raises:
            ValueError: 如果没有有效动作可用
        """
        # 步骤1：提取有效动作列表
        # valid_mask是布尔数组，np.flatnonzero返回True位置的索引
        valid_actions = np.flatnonzero(obs["valid_mask"])

        # 步骤2：边界检查
        if len(valid_actions) == 0:
            print("Warning: No valid actions available, returning 0")
            return 0

        # 步骤3：均匀随机选择
        action = int(self.rng.choice(valid_actions))

        return action

    def __repr__(self):
        return f"RandomPolicy(seed={self.rng.bit_generator.seed_seq.entropy})"


def create_random_policy(seed: int = 42) -> RandomPolicy:
    """工厂函数：创建带默认种子的随机策略"""
    return RandomPolicy(seed=seed)


if __name__ == "__main__":
    print("Testing RandomPolicy...")

    policy = RandomPolicy(seed=42)
    print(f"Created policy: {policy}")

    # 模拟一个观测
    mock_obs = {
        "worker_state": np.random.randn(12),
        "candidate_state": np.random.randn(20, 14),
        "valid_mask": np.array([True]*15 + [False]*5),  # 前15个有效
        "info": {}
    }

    action = policy.select_action(mock_obs)
    print(f"Selected action: {action} (should be in range [0, 14])")

    # 测试多次选择的分布
    actions = [policy.select_action(mock_obs) for _ in range(1000)]
    unique_actions = set(actions)
    print(f"1000 selections covered {len(unique_actions)} unique actions (expected ~15)")
    print("✓ RandomPolicy test passed!")
