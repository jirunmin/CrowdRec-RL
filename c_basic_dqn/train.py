import argparse
import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from src.env import make_env
from agent_dqn import DQNAgent

WARMUP_STEPS = 3_000
STEP_LOG_INTERVAL = 5_000
REWARD_SCALE = 20.0
EPSILON_DECAY_STEPS = 1_675_000
TARGET_UPDATE_FREQ = 2_000
EVAL_EPISODES = 3
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


def evaluate_model(agent, split="val", reward_mode="worker", eval_episodes=EVAL_EPISODES):
    episode_rewards = []
    for _ in range(eval_episodes):
        env = make_env(
            split=split,
            reward_mode=reward_mode,
            candidate_mode=CANDIDATE_MODE,
            seed=SEED,
        )
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state = obs_to_state(obs)
            action = agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        episode_rewards.append(float(total_reward))

    return float(np.mean(episode_rewards))


def _resolve_batch_size(device: str, base_batch_size: int = 64) -> int:
    if device != "cpu":
        return base_batch_size

    cpu_cores = os.cpu_count() or 4
    if cpu_cores >= 16:
        return max(base_batch_size, 256)
    if cpu_cores >= 8:
        return max(base_batch_size, 128)
    return max(base_batch_size, 64)


def run_training_pipeline(reward_mode="worker", num_episodes=6, device="auto"):
    print(
        f"\n>>> 开始训练: reward_mode={reward_mode}, candidate_mode={CANDIDATE_MODE}, "
        f"warmup={WARMUP_STEPS}, seed={SEED}, device={device}, "
        f"reward_scale=1/{REWARD_SCALE}, epsilon_linear_steps={EPSILON_DECAY_STEPS}, "
        f"target_update_freq={TARGET_UPDATE_FREQ}, eval_episodes={EVAL_EPISODES} <<<"
    )

    train_env = make_env(
        split="train",
        reward_mode=reward_mode,
        candidate_mode=CANDIDATE_MODE,
        seed=SEED,
    )

    resolved_batch_size = _resolve_batch_size(device)
    agent = DQNAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.max_candidates,
        device=device,
        lr=5e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        batch_size=resolved_batch_size,
        target_update_freq=TARGET_UPDATE_FREQ,
        grad_clip_norm=20.0,
    )
    print(f"[Info] 实际训练设备: {agent.device}, batch_size={agent.batch_size}")

    best_val_reward = -float("inf")
    train_rewards, losses, eps_history = [], [], []
    step_logs = []

    step_log_path = f"train_log_{reward_mode}_per_{STEP_LOG_INTERVAL}_steps.csv"
    if os.path.exists(step_log_path):
        os.remove(step_log_path)

    warmup_last_print = -1
    warmup_done_announced = False
    global_step = 0

    episode_pbar = tqdm(range(num_episodes), desc=f"Training [{reward_mode}]", unit="ep")
    for ep in episode_pbar:
        obs = train_env.reset()
        done = False
        ep_reward, ep_loss, steps = 0.0, 0.0, 0

        while not done:
            state = obs_to_state(obs)
            agent.set_epsilon_by_step(global_step)
            action = agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=False)
            next_obs, reward, done, _ = train_env.step(action)
            next_state = obs_to_state(next_obs)

            scaled_reward = reward / REWARD_SCALE
            agent.store_transition(
                state,
                action,
                scaled_reward,
                next_state,
                next_obs["valid_mask"],
                done,
            )

            buffer_len = len(agent.replay_buffer)
            if buffer_len < WARMUP_STEPS:
                warmup_pct = int(buffer_len * 100 / WARMUP_STEPS)
                if warmup_pct >= warmup_last_print + 5:
                    warmup_last_print = warmup_pct
                    tqdm.write(f"[Warmup] 经验池预热进度: {buffer_len}/{WARMUP_STEPS} ({warmup_pct}%)")
            elif not warmup_done_announced:
                warmup_done_announced = True
                tqdm.write(f"[Warmup] 经验池预热完成: {buffer_len}/{WARMUP_STEPS}，开始更新网络")

            if buffer_len >= WARMUP_STEPS:
                loss = agent.update()
                if loss is not None:
                    ep_loss += loss

            obs = next_obs
            ep_reward += reward
            steps += 1
            global_step += 1

            if global_step % STEP_LOG_INTERVAL == 0:
                avg_loss = ep_loss / steps if steps > 0 else 0.0
                tqdm.write(f"[Train] 全局步数: {global_step}")
                step_record = {
                    "global_step": global_step,
                    "episode": ep + 1,
                    "episode_inner_step": steps,
                    "buffer_size": len(agent.replay_buffer),
                    "epsilon": agent.epsilon,
                    "episode_reward_so_far": ep_reward,
                    "episode_avg_loss_so_far": avg_loss,
                }
                step_logs.append(step_record)
                pd.DataFrame([step_record]).to_csv(
                    step_log_path,
                    mode="a",
                    header=not os.path.exists(step_log_path),
                    index=False,
                )

        train_rewards.append(ep_reward)
        losses.append(ep_loss / steps if steps > 0 else 0.0)
        eps_history.append(agent.epsilon)

        val_reward = evaluate_model(agent, split="val", reward_mode=reward_mode, eval_episodes=EVAL_EPISODES)
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            model_path = f"basic_dqn_best_{reward_mode}_model.pth"
            agent.save_model(model_path)
            tqdm.write(
                f"[Checkpoint] 保存新的最优模型: {model_path} "
                f"(ep={ep + 1}, val_reward={val_reward:.2f})"
            )

        tqdm.write(
            f"Ep {ep + 1:03d} | Buffer: {len(agent.replay_buffer)} | "
            f"Train Reward: {ep_reward:.2f} | Val Reward(mean@{EVAL_EPISODES}): {val_reward:.2f} | "
            f"Loss: {losses[-1]:.4f} | Eps: {agent.epsilon:.4f}"
        )

        episode_pbar.set_postfix(
            {
                "reward": f"{ep_reward:.2f}",
                "loss": f"{losses[-1]:.4f}",
                "eps": f"{agent.epsilon:.4f}",
                "buf": len(agent.replay_buffer),
                "val": f"{val_reward:.2f}",
            }
        )

    if len(step_logs) == 0 or step_logs[-1]["global_step"] != global_step:
        final_step_record = {
            "global_step": global_step,
            "episode": num_episodes,
            "episode_inner_step": steps if num_episodes > 0 else 0,
            "buffer_size": len(agent.replay_buffer),
            "epsilon": agent.epsilon,
            "episode_reward_so_far": train_rewards[-1] if train_rewards else 0.0,
            "episode_avg_loss_so_far": losses[-1] if losses else 0.0,
        }
        step_logs.append(final_step_record)
        pd.DataFrame([final_step_record]).to_csv(
            step_log_path,
            mode="a",
            header=not os.path.exists(step_log_path),
            index=False,
        )

    pd.DataFrame(
        {
            "reward": train_rewards,
            "loss": losses,
            "eps": eps_history,
        }
    ).to_csv(f"train_log_{reward_mode}_per_episode.csv", index=False)

    return train_rewards, losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN for CrowdRec")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备：auto(默认自动选择), cpu, cuda",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=6,
        help="训练轮数（每个 reward_mode 各训练这么多轮）",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="requester",
        choices=["worker", "requester", "both"],
        help="训练哪种 reward。both 会顺序训练 worker 和 requester。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    if args.reward_mode == "both":
        run_training_pipeline("worker", num_episodes=args.episodes, device=args.device)
        run_training_pipeline("requester", num_episodes=args.episodes, device=args.device)
    else:
        run_training_pipeline(args.reward_mode, num_episodes=args.episodes, device=args.device)
