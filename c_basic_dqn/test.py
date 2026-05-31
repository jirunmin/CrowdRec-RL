import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    hits = 0
    resolvable = 0

    with tqdm(total=len(env), desc=f"{reward_mode} test", unit="step") as pbar:
        while not done:
            state = obs_to_state(obs)
            action = agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if info["ground_truth_index"] >= 0:
                resolvable += 1
                hits += int(info["hit"])
            pbar.update(1)

    hit_rate = hits / max(resolvable, 1)
    return float(total_reward), float(hit_rate), str(agent.device)


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

    w_score, w_hit, used_device = evaluate("worker", OUT_DIR / "basic_dqn_best_worker_model.pth", device=args.device)
    r_score, r_hit, _ = evaluate("requester", OUT_DIR / "basic_dqn_best_requester_model.pth", device=args.device)

    print(f"[Info] 评估设备: {used_device}")
    print(f"Worker   score: {w_score:.4f} | hit_rate: {w_hit:.4f}")
    print(f"Requester score: {r_score:.4f} | hit_rate: {r_hit:.4f}")

    plt.bar(["Worker Agent", "Requester Agent"], [w_score, r_score], color=["blue", "orange"])
    plt.ylabel("Cumulative Reward")
    plt.title("Performance Comparison")
    plt.savefig("final_comparison.png")
    plt.show()
