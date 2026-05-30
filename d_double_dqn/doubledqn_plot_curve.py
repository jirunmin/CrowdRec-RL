import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def episode_boundary_steps(df_step):
    if "episode" not in df_step.columns or "global_step" not in df_step.columns:
        return []
    starts = df_step.groupby("episode", sort=True)["global_step"].min().tolist()
    return [int(x) for x in starts]


def plot_step_loss_epsilon(df_step, out_path):
    required = {"global_step", "episode_avg_loss_so_far", "epsilon", "episode"}
    missing = required - set(df_step.columns)
    if missing:
        raise ValueError(f"step log missing columns: {missing}")

    x = df_step["global_step"]
    y_loss = df_step["episode_avg_loss_so_far"]
    y_eps = df_step["epsilon"]
    boundaries = episode_boundary_steps(df_step)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x, y_loss, color="tab:blue", linewidth=1.8, label="episode_avg_loss_so_far")
    ax1.set_xlabel("global_step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x, y_eps, color="tab:orange", linewidth=1.6, label="epsilon")
    ax2.set_ylabel("epsilon", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    for i, step in enumerate(boundaries):
        ax1.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
        ax1.text(
            step,
            ax1.get_ylim()[1],
            f"ep{i + 1}",
            fontsize=8,
            color="gray",
            alpha=0.8,
            rotation=90,
            va="bottom",
            ha="right",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Double DQN Step Log: Loss & Epsilon vs Global Step")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_step_reward(df_step, out_path):
    required = {"global_step", "episode_reward_so_far", "episode"}
    missing = required - set(df_step.columns)
    if missing:
        raise ValueError(f"step log missing columns: {missing}")

    x = df_step["global_step"]
    y = df_step["episode_reward_so_far"]
    boundaries = episode_boundary_steps(df_step)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="tab:green", linewidth=1.8, label="episode_reward_so_far")
    for i, step in enumerate(boundaries):
        ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(
            step,
            ax.get_ylim()[1],
            f"ep{i + 1}",
            fontsize=8,
            color="gray",
            alpha=0.8,
            rotation=90,
            va="bottom",
            ha="right",
        )

    ax.set_xlabel("global_step")
    ax.set_ylabel("reward")
    ax.set_title("Double DQN Step Log: Reward vs Global Step")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_episode_reward(df_ep, out_path):
    if "reward" not in df_ep.columns:
        raise ValueError("episode log missing column: reward")

    episodes = list(range(1, len(df_ep) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, df_ep["reward"], marker="o", linewidth=1.8, color="tab:purple", label="reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_xticks(episodes)
    ax.set_title("Double DQN Episode Log: Reward vs Episode")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_episode_loss(df_ep, out_path):
    if "loss" not in df_ep.columns:
        raise ValueError("episode log missing column: loss")

    episodes = list(range(1, len(df_ep) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, df_ep["loss"], marker="o", linewidth=1.8, color="tab:red", label="loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.set_xticks(episodes)
    ax.set_title("Double DQN Episode Log: Loss vs Episode")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Double DQN training curves.")
    parser.add_argument("--reward-mode", type=str, default="requester", choices=["worker", "requester"])
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    step_log_path = out_dir / f"train_log_double_dqn_{args.reward_mode}_per_5000_steps.csv"
    episode_log_path = out_dir / f"train_log_double_dqn_{args.reward_mode}_per_episode.csv"

    if not step_log_path.exists():
        raise FileNotFoundError(f"step log not found: {step_log_path}")
    if not episode_log_path.exists():
        raise FileNotFoundError(f"episode log not found: {episode_log_path}")

    df_step = pd.read_csv(str(step_log_path))
    df_ep = pd.read_csv(str(episode_log_path))

    plot_step_loss_epsilon(df_step, out_dir / f"double_dqn_{args.reward_mode}_step_loss_epsilon.png")
    plot_step_reward(df_step, out_dir / f"double_dqn_{args.reward_mode}_step_reward.png")
    plot_episode_reward(df_ep, out_dir / f"double_dqn_{args.reward_mode}_episode_reward.png")
    plot_episode_loss(df_ep, out_dir / f"double_dqn_{args.reward_mode}_episode_loss.png")

    print("Saved Double DQN plots to:", out_dir)


if __name__ == "__main__":
    main()
