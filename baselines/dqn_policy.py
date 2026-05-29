"""
DQN 策略封装：将 C 角色的 DQN 模型包装为标准策略接口，接入评估框架。

用法：
  policy = DQNPolicy(model_path="c_basic_dqn/basic_dqn_best_worker_model.pth")
  action = policy.select_action(obs)
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
import torch
# 添加 c_basic_dqn 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "c_basic_dqn"))


class DQNPolicy:
    """封装 DQNAgent 为标准策略接口。"""

    def __init__(self, model_path: str, device: str = "auto"):
        from agent_dqn import DQNAgent
        from src.env import make_env

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device.lower()
        print(f"Loading DQN model from {model_path} to device {device}")

        # 获取环境参数
        env = make_env(split="val", reward_mode="worker", seed=42)
        state_dim = env.state_dim
        action_dim = env.max_candidates

        # 创建 agent 并加载模型
        self.agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
        full_path = os.path.join(os.path.dirname(__file__), "..", model_path)
        self.agent.load_model(full_path)
        self.name = "DQN"

    def select_action(self, obs: Dict[str, Any]) -> int:
        state = self._obs_to_state(obs)
        return self.agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)

    @staticmethod
    def _obs_to_state(obs: Dict[str, Any]) -> np.ndarray:
        worker = np.asarray(obs["worker_state"], dtype=np.float32).reshape(-1)
        candidate = np.asarray(obs["candidate_state"], dtype=np.float32).reshape(-1)
        return np.concatenate([worker, candidate], axis=0)

    def __repr__(self):
        return f"DQNPolicy(device={self.agent.device})"
