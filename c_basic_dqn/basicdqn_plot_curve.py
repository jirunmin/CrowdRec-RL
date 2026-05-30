import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _episode_boundary_steps(df_step):
    if "episode" not in df_step.columns or "global_step" not in df_step.columns:
        return []

    starts = df_step.groupby("episode", sort=True)["global_step"].min().tolist()
    return [int(x) for x in starts]


def plot_step_loss_epsilon(df_step, out_path):
    required = {"global_step", "episode_avg_loss_so_far", "epsilon", "episode"}
    missing = required - set(df_step.columns)
    if missing:
        raise ValueError("step log missing columns: {}".format(missing))

    x = df_step["global_step"]
    y_loss = df_step["episode_avg_loss_so_far"]
    y_eps = df_step["epsilon"]
    boundaries = _episode_boundary_steps(df_step)

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
            "ep{}".format(i + 1),
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
    ax1.set_title("Step Log: Loss & Epsilon vs Global Step")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_step_reward(df_step, out_path):
    required = {"global_step", "episode_reward_so_far", "episode"}
    missing = required - set(df_step.columns)
    if missing:
        raise ValueError("step log missing columns: {}".format(missing))

    x = df_step["global_step"]
    y = df_step["episode_reward_so_far"]
    boundaries = _episode_boundary_steps(df_step)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="tab:green", linewidth=1.8, label="episode_reward_so_far")

    for i, step in enumerate(boundaries):
        ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(
            step,
            ax.get_ylim()[1],
            "ep{}".format(i + 1),
            fontsize=8,
            color="gray",
            alpha=0.8,
            rotation=90,
            va="bottom",
            ha="right",
        )

    ax.set_xlabel("global_step")
    ax.set_ylabel("reward")
    ax.set_title("Step Log: Reward vs Global Step")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_step_success_rate(df_step, out_path):
    required = {"global_step", "episode_success_rate_so_far", "episode"}
    missing = required - set(df_step.columns)
    if missing:
        raise ValueError("step log missing columns: {}".format(missing))

    x = df_step["global_step"]
    y = df_step["episode_success_rate_so_far"]
    boundaries = _episode_boundary_steps(df_step)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="tab:cyan", linewidth=1.8, label="episode_success_rate_so_far")

    for i, step in enumerate(boundaries):
        ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(
            step,
            ax.get_ylim()[1],
            "ep{}".format(i + 1),
            fontsize=8,
            color="gray",
            alpha=0.8,
            rotation=90,
            va="bottom",
            ha="right",
        )

    ax.set_xlabel("global_step")
    ax.set_ylabel("success_rate")
    ax.set_title("Step Log: Success Rate vs Global Step")
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
    ax.set_title("Episode Log: Reward vs Episode")
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
    ax.set_title("Episode Log: Loss vs Episode")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def plot_episode_success_rate(df_ep, out_path):
    if "success_rate" not in df_ep.columns:
        raise ValueError("episode log missing column: success_rate")

    episodes = list(range(1, len(df_ep) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        episodes,
        df_ep["success_rate"],
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="success_rate",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("success_rate")
    ax.set_xticks(episodes)
    ax.set_title("Episode Log: Success Rate vs Episode")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DQN training curves from step/episode CSV logs.")
    parser.add_argument(
        "--step-log",
        type=str,
        default="train_log_worker_per_5000_steps.csv",
        help="Path to per-5000-step training log CSV.",
    )
    parser.add_argument(
        "--episode-log",
        type=str,
        default="train_log_worker_per_episode.csv",
        help="Path to per-episode training log CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="c_basic_dqn",
        help="Output directory for saved figures.",
    )
    args = parser.parse_args()

    step_log_path = Path(args.step_log)
    episode_log_path = Path(args.episode_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not step_log_path.exists():
        raise FileNotFoundError("step log not found: {}".format(step_log_path))
    if not episode_log_path.exists():
        raise FileNotFoundError("episode log not found: {}".format(episode_log_path))

    df_step = pd.read_csv(str(step_log_path))
    df_ep = pd.read_csv(str(episode_log_path))

    plot_step_loss_epsilon(df_step, out_dir / "worker_step_loss_epsilon.png")
    plot_step_reward(df_step, out_dir / "worker_step_reward.png")
    plot_step_success_rate(df_step, out_dir / "worker_step_success_rate.png")
    plot_episode_reward(df_ep, out_dir / "worker_episode_reward.png")
    plot_episode_loss(df_ep, out_dir / "worker_episode_loss.png")
    plot_episode_success_rate(df_ep, out_dir / "worker_episode_success_rate.png")

    print("Saved plots:")
    print(out_dir / "worker_step_loss_epsilon.png")
    print(out_dir / "worker_step_reward.png")
    print(out_dir / "worker_step_success_rate.png")
    print(out_dir / "worker_episode_reward.png")
    print(out_dir / "worker_episode_loss.png")
    print(out_dir / "worker_episode_success_rate.png")


if __name__ == "__main__":
    main()
