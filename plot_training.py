# plot_training.py

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    """Load a training CSV log and return episodes, rewards, and avg rewards."""
    episodes = []
    rewards = []
    avg_rewards = []

    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return None, None, None

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avg_rewards.append(float(row["avg_reward_50"]))

    return episodes, rewards, avg_rewards


def plot_single(ax, episodes, rewards, avg_rewards, title, color):
    """Plot reward curve on a given axis."""
    ax.plot(episodes, rewards, alpha=0.3, color=color, label="Episode Reward")
    ax.plot(episodes, avg_rewards, color=color, linewidth=2, label="Moving Avg (50)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    dqn_eps, dqn_rew, dqn_avg = load_log("logs/dqn_training.csv")
    ppo_eps, ppo_rew, ppo_avg = load_log("logs/ppo_training.csv")

    has_dqn = dqn_eps is not None
    has_ppo = ppo_eps is not None

    if not has_dqn and not has_ppo:
        print("No training logs found in logs/. Run training first.")
        return

    n_plots = int(has_dqn) + int(has_ppo)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    idx = 0
    if has_dqn:
        plot_single(axes[idx], dqn_eps, dqn_rew, dqn_avg, "DQN Training Rewards", "#e74c3c")
        idx += 1
    if has_ppo:
        plot_single(axes[idx], ppo_eps, ppo_rew, ppo_avg, "PPO Training Rewards", "#2ecc71")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved training_curves.png")


if __name__ == "__main__":
    main()
