import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import make_env  # noqa: E402
from agent_double_dqn import DoubleDQNAgent  # noqa: E402


SEED = 42
DEFAULT_CANDIDATE_MODE = "event_group"
DEFAULT_PROCESSED_DIR = ROOT / "processed"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def obs_to_state(obs):
    worker = np.asarray(obs["worker_state"], dtype=np.float32).reshape(-1)
    candidate = np.asarray(obs["candidate_state"], dtype=np.float32).reshape(-1)
    return np.concatenate([worker, candidate], axis=0)


def evaluate(
    reward_mode,
    model_path,
    device="auto",
    processed_dir=DEFAULT_PROCESSED_DIR,
    candidate_mode=DEFAULT_CANDIDATE_MODE,
):
    resolved_device = resolve_device(device)
    env = make_env(
        split="test",
        processed_dir=str(processed_dir),
        reward_mode=reward_mode,
        candidate_mode=candidate_mode,
        seed=SEED,
    )
    agent = DoubleDQNAgent(state_dim=env.state_dim, action_dim=env.max_candidates, device=resolved_device)
    agent.load_model(model_path)

    obs = env.reset()
    total_reward = 0.0
    done = False
    hits = 0
    resolvable = 0

    with tqdm(total=len(env), desc=f"Double DQN {reward_mode} test", unit="step") as pbar:
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
    parser = argparse.ArgumentParser(description="Test Double DQN for CrowdRec")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="both",
        choices=["worker", "requester", "both"],
    )
    parser.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default=DEFAULT_CANDIDATE_MODE,
        choices=["event_group", "top_k"],
    )
    parser.add_argument("--worker-model", type=str, default=None)
    parser.add_argument("--requester-model", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    out_dir = Path(args.out_dir)
    model_paths = {
        "worker": Path(args.worker_model) if args.worker_model else out_dir / "double_dqn_best_worker_model.pth",
        "requester": (
            Path(args.requester_model)
            if args.requester_model
            else out_dir / "double_dqn_best_requester_model.pth"
        ),
    }

    modes = ["worker", "requester"] if args.reward_mode == "both" else [args.reward_mode]
    scores = {}
    hit_rates = {}
    used_device = None

    for mode in modes:
        score, hit_rate, used_device = evaluate(
            mode,
            model_paths[mode],
            device=args.device,
            processed_dir=args.processed_dir,
            candidate_mode=args.candidate_mode,
        )
        scores[mode] = score
        hit_rates[mode] = hit_rate

    print(f"[Info] device: {used_device}")
    for mode in modes:
        print(f"{mode.capitalize()} score: {scores[mode]:.4f} | hit_rate: {hit_rates[mode]:.4f}")

    if len(modes) == 2:
        labels = ["Worker Agent", "Requester Agent"]
        values = [scores["worker"], scores["requester"]]
        plt.bar(labels, values, color=["blue", "orange"])
        plt.ylabel("Cumulative Reward")
        plt.title("Double DQN Performance Comparison")
        plt.tight_layout()
        plt.savefig(out_dir / "double_dqn_final_comparison.png")
        plt.close()
