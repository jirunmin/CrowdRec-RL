import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import make_env  # noqa: E402
from agent_double_dqn import DoubleDQNAgent  # noqa: E402


WARMUP_STEPS = 3_000
STEP_LOG_INTERVAL = 5_000
REWARD_SCALE = 20.0
EPSILON_DECAY_STEPS = 1_675_000
TARGET_UPDATE_FREQ = 2_000
EVAL_EPISODES = 3
SEED = 42
EARLY_STOP_PATIENCE = 5
VAL_METRIC_MIN_DELTA = 0.002
EARLY_STOP_METRIC = "val_reward"
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


def evaluate_model(
    agent,
    split="val",
    reward_mode="worker",
    processed_dir=DEFAULT_PROCESSED_DIR,
    candidate_mode=DEFAULT_CANDIDATE_MODE,
    eval_episodes=EVAL_EPISODES,
):
    episode_rewards = []
    episode_success_rates = []
    for _ in range(eval_episodes):
        env = make_env(
            split=split,
            processed_dir=str(processed_dir),
            reward_mode=reward_mode,
            candidate_mode=candidate_mode,
            seed=SEED,
            normalize_features=True,
        )
        obs = env.reset()
        total_reward = 0.0
        done = False
        hits = 0
        valid_steps = 0

        while not done:
            state = obs_to_state(obs)
            action = agent.select_action(state, valid_mask=obs["valid_mask"], eval_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if info.get("ground_truth_index", -1) >= 0:
                valid_steps += 1
                hits += int(info.get("hit", 0))

        episode_rewards.append(float(total_reward))
        episode_success_rates.append(float(hits / valid_steps) if valid_steps > 0 else 0.0)

    return float(np.mean(episode_rewards)), float(np.mean(episode_success_rates))


def resolve_batch_size(device: str, base_batch_size: int = 64) -> int:
    if device != "cpu":
        return base_batch_size

    cpu_cores = os.cpu_count() or 4
    if cpu_cores >= 16:
        return max(base_batch_size, 256)
    if cpu_cores >= 8:
        return max(base_batch_size, 128)
    return max(base_batch_size, 64)


def run_training_pipeline(
    reward_mode="worker",
    num_episodes=6,
    device="auto",
    processed_dir=DEFAULT_PROCESSED_DIR,
    out_dir=DEFAULT_OUT_DIR,
    candidate_mode=DEFAULT_CANDIDATE_MODE,
    gamma=0.99,
):
    resolved_device = resolve_device(device)
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n>>> Train Double DQN: reward_mode={reward_mode}, candidate_mode={candidate_mode}, "
        f"warmup={WARMUP_STEPS}, seed={SEED}, device={resolved_device}, gamma={gamma}, "
        f"reward_scale=1/{REWARD_SCALE}, epsilon_linear_steps={EPSILON_DECAY_STEPS}, "
        f"target_update_freq={TARGET_UPDATE_FREQ}, eval_episodes={EVAL_EPISODES} <<<"
    )

    train_env = make_env(
        split="train",
        processed_dir=str(processed_dir),
        reward_mode=reward_mode,
        candidate_mode=candidate_mode,
        seed=SEED,
        normalize_features=True,
    )

    batch_size = resolve_batch_size(resolved_device)
    agent = DoubleDQNAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.max_candidates,
        device=resolved_device,
        lr=5e-4,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        batch_size=batch_size,
        target_update_freq=TARGET_UPDATE_FREQ,
        grad_clip_norm=20.0,
    )
    print(f"[Info] device={agent.device}, batch_size={agent.batch_size}")

    best_val_reward = -float("inf")
    best_val_success_rate = -float("inf")
    no_improve_count = 0
    train_rewards, losses, eps_history = [], [], []
    step_logs = []

    step_log_path = out_dir / f"train_log_double_dqn_{reward_mode}_per_{STEP_LOG_INTERVAL}_steps.csv"
    if step_log_path.exists():
        step_log_path.unlink()

    warmup_last_print = -1
    warmup_done_announced = False
    global_step = 0

    episode_pbar = tqdm(range(num_episodes), desc=f"Double DQN [{reward_mode}]", unit="ep")
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

            agent.store_transition(
                state,
                action,
                reward / REWARD_SCALE,
                next_state,
                next_obs["valid_mask"],
                done,
            )

            buffer_len = len(agent.replay_buffer)
            if buffer_len < WARMUP_STEPS:
                warmup_pct = int(buffer_len * 100 / WARMUP_STEPS)
                if warmup_pct >= warmup_last_print + 5:
                    warmup_last_print = warmup_pct
                    tqdm.write(f"[Warmup] replay buffer: {buffer_len}/{WARMUP_STEPS} ({warmup_pct}%)")
            elif not warmup_done_announced:
                warmup_done_announced = True
                tqdm.write(f"[Warmup] replay buffer ready: {buffer_len}/{WARMUP_STEPS}")

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
                    header=not step_log_path.exists(),
                    index=False,
                )
                tqdm.write(f"[Train] global_step={global_step}, avg_loss={avg_loss:.6f}")

        train_rewards.append(ep_reward)
        losses.append(ep_loss / steps if steps > 0 else 0.0)
        eps_history.append(agent.epsilon)

        val_reward, val_success_rate = evaluate_model(
            agent,
            split="val",
            reward_mode=reward_mode,
            processed_dir=processed_dir,
            candidate_mode=candidate_mode,
            eval_episodes=EVAL_EPISODES,
        )

        # Early stopping (match C baseline)
        monitored_metric = val_reward if EARLY_STOP_METRIC == "val_reward" else val_success_rate
        best_monitored = best_val_reward if EARLY_STOP_METRIC == "val_reward" else best_val_success_rate

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            model_path = out_dir / f"double_dqn_best_{reward_mode}_model.pth"
            agent.save_model(model_path)
            tqdm.write(
                f"[Checkpoint] saved {model_path} "
                f"(ep={ep + 1}, val_reward={val_reward:.2f})"
            )

        if val_success_rate > best_val_success_rate:
            best_val_success_rate = val_success_rate

        if monitored_metric - best_monitored > VAL_METRIC_MIN_DELTA:
            best_monitored = monitored_metric
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOP_PATIENCE:
            tqdm.write(f"[EarlyStop] no improvement for {EARLY_STOP_PATIENCE} episodes, stopping")
            break

        tqdm.write(
            f"Ep {ep + 1:03d} | Buffer: {len(agent.replay_buffer)} | "
            f"Train Reward: {ep_reward:.2f} | Val Reward: {val_reward:.2f} | "
            f"Val Succ: {val_success_rate:.4f} | Loss: {losses[-1]:.6f} | Eps: {agent.epsilon:.4f}"
        )

        episode_pbar.set_postfix(
            {
                "reward": f"{ep_reward:.2f}",
                "loss": f"{losses[-1]:.6f}",
                "eps": f"{agent.epsilon:.4f}",
                "buf": len(agent.replay_buffer),
                "val": f"{val_reward:.2f}",
                "succ": f"{val_success_rate:.3f}",
            }
        )

    if num_episodes > 0 and (len(step_logs) == 0 or step_logs[-1]["global_step"] != global_step):
        final_step_record = {
            "global_step": global_step,
            "episode": num_episodes,
            "episode_inner_step": steps,
            "buffer_size": len(agent.replay_buffer),
            "epsilon": agent.epsilon,
            "episode_reward_so_far": train_rewards[-1] if train_rewards else 0.0,
            "episode_avg_loss_so_far": losses[-1] if losses else 0.0,
        }
        pd.DataFrame([final_step_record]).to_csv(
            step_log_path,
            mode="a",
            header=not step_log_path.exists(),
            index=False,
        )

    pd.DataFrame(
        {
            "reward": train_rewards,
            "loss": losses,
            "eps": eps_history,
        }
    ).to_csv(out_dir / f"train_log_double_dqn_{reward_mode}_per_episode.csv", index=False)

    return train_rewards, losses


def parse_args():
    parser = argparse.ArgumentParser(description="Train Double DQN for CrowdRec")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="requester",
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Use 0.99 to match the C baseline, or 0.0 for one-step offline ranking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    if args.reward_mode == "both":
        run_training_pipeline(
            "worker",
            num_episodes=args.episodes,
            device=args.device,
            processed_dir=args.processed_dir,
            out_dir=args.out_dir,
            candidate_mode=args.candidate_mode,
            gamma=args.gamma,
        )
        run_training_pipeline(
            "requester",
            num_episodes=args.episodes,
            device=args.device,
            processed_dir=args.processed_dir,
            out_dir=args.out_dir,
            candidate_mode=args.candidate_mode,
            gamma=args.gamma,
        )
    else:
        run_training_pipeline(
            args.reward_mode,
            num_episodes=args.episodes,
            device=args.device,
            processed_dir=args.processed_dir,
            out_dir=args.out_dir,
            candidate_mode=args.candidate_mode,
            gamma=args.gamma,
        )
