"""
DQN 策略封装：将 C/D 角色的 DQN/Double-DQN 模型包装为标准策略接口，接入评估框架。

用法：
  policy = DQNPolicy(model_path="c_basic_dqn/basic_dqn_best_worker_model.pth")
  action = policy.select_action(obs)
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
import torch


class DQNPolicy:
    """封装 DQNAgent/DoubleDQNAgent 为标准策略接口。"""

    def __init__(self, model_path: str, device: str = "auto", agent_type: str = "dqn"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device.lower()
        print(f"Loading {agent_type} model from {model_path} to device {device}")

        # 根据 agent_type 添加对应的路径并导入
        project_root = os.path.join(os.path.dirname(__file__), "..")
        if agent_type == "double-dqn":
            agent_dir = os.path.join(project_root, "d_double_dqn")
            sys.path.insert(0, agent_dir)
            from agent_double_dqn import DoubleDQNAgent as AgentClass
        else:
            agent_dir = os.path.join(project_root, "c_basic_dqn")
            sys.path.insert(0, agent_dir)
            from agent_dqn import DQNAgent as AgentClass

        # 获取环境参数
        from src.env import make_env
        env = make_env(split="val", reward_mode="worker", seed=42)
        state_dim = env.state_dim
        action_dim = env.max_candidates

        # 创建 agent 并加载模型
        self.agent = AgentClass(state_dim=state_dim, action_dim=action_dim, device=device)
        full_path = os.path.join(project_root, model_path)
        self.agent.load_model(full_path)
        self.name = agent_type.upper()

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
