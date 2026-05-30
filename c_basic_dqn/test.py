import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.fastenv import make_fast_env
from agent_dqn import DQNAgent

SEED = 42
CANDIDATE_MODE = "event_group"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_state(obs):
    worker = np.asarray(obs["worker_state"], dtype=np.float32).reshape(-1)
    candidate = np.asarray(obs["candidate_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([worker, candidate], axis=0)


def evaluate(reward_mode, model_path, device="auto"):
    env = make_fast_env(
        split="test",
        reward_mode=reward_mode,
        candidate_mode=CANDIDATE_MODE,
        seed=SEED,
        normalize_features=True,
    )
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.max_candidates, device=device)
    agent.load_model(model_path)

    obs = env.reset()
    total_reward = 0.0
    done = False

    with tqdm(total=len(env), desc=f"{reward_mode} test", unit="step") as pbar:
        while not done:
            state = obs_to_state(obs)
            action = agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            pbar.update(1)

    return float(total_reward), str(agent.device)


def parse_args():
    parser = argparse.ArgumentParser(description="Test DQN for CrowdRec")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="评估设备：auto(默认自动选择), cpu, cuda",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    w_score, used_device = evaluate("worker", "basic_dqn_best_worker_model.pth", device=args.device)
    r_score, _ = evaluate("requester", "basic_dqn_best_requester_model.pth", device=args.device)

    print(f"[Info] 评估设备: {used_device}")
    print(f"Worker score: {w_score:.4f}")
    print(f"Requester score: {r_score:.4f}")

    plt.bar(["Worker Agent", "Requester Agent"], [w_score, r_score], color=["blue", "orange"])
    plt.ylabel("Cumulative Reward")
    plt.title("Performance Comparison")
    plt.savefig("final_comparison.png")
    plt.show()
