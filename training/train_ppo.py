# training/train_ppo.py

import os
import csv
import shutil
import torch
import numpy as np

from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from agents.ppo.agent import PPOAgent
from utils.config import COMMON, DATA_CONFIG, TRAIN_CONFIG, REWARD_CONFIG
from utils.checkpoints import PIPELINE_VERSION, save_checkpoint
from utils.reproducibility import set_seed
from utils.metrics import buy_and_hold_values, validation_score
from utils.validation import split_validation_windows


set_seed(COMMON["seed"])

# -----------------------------------------------------------------------
# Load multi-ticker data; carve out a chronological validation slice
# -----------------------------------------------------------------------
tickers      = TRAIN_CONFIG["tickers"]
random_start = TRAIN_CONFIG["random_start"]
episode_length = TRAIN_CONFIG["episode_length"]
val_windows = TRAIN_CONFIG["val_windows"]
val_window_size = TRAIN_CONFIG["val_window_size"]

reward_kwargs = REWARD_CONFIG.copy()

train_envs = []
val_envs   = []

for ticker in tickers:
    data = load_data(ticker, **DATA_CONFIG)
    train_data, _ = train_test_split(data, train_ratio=COMMON["train_split"])

    train_only, ticker_val_windows = split_validation_windows(
        train_data,
        observation_window=COMMON["window_size"],
        num_windows=val_windows,
        window_rows=val_window_size,
    )

    t_env = TradingEnv(
        train_only,
        window_size=COMMON["window_size"],
        random_start=random_start,
        episode_length=episode_length,
        **reward_kwargs,
    )
    train_envs.append((ticker, t_env))
    val_envs.extend(
        (f"{ticker}-window-{index + 1}", TradingEnv(
            val_data,
            window_size=COMMON["window_size"],
            random_start=False,
            **reward_kwargs,
        ))
        for index, val_data in enumerate(ticker_val_windows)
    )
    print(f"  {ticker}: {len(train_only)} train / {len(ticker_val_windows)} x "
          f"{val_window_size}-day validation windows, "
          f"state_size={t_env.reset().shape[0]}")

state_size  = train_envs[0][1].reset().shape[0]
action_size = COMMON["action_size"]
window_size = COMMON["window_size"]

agent = PPOAgent(state_size, action_size=action_size, window_size=window_size)

episodes        = TRAIN_CONFIG["ppo_episodes"]
update_timestep = TRAIN_CONFIG["ppo_update_timestep"]
save_every      = TRAIN_CONFIG["save_every"]
val_every       = TRAIN_CONFIG["val_every"]
log_dir         = TRAIN_CONFIG["log_dir"]
timestep        = 0

os.makedirs("models", exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

log_path = os.path.join(log_dir, "ppo_training.csv")
log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["pipeline_version", "episode", "ticker", "reward", "timesteps", "avg_reward_50", "val_return_pct", "val_score"])

reward_history  = []
best_val_score = -np.inf
validations_without_improvement = 0


# -----------------------------------------------------------------------
# Greedy validation helper
# -----------------------------------------------------------------------
def validate_ppo(agent, val_env):
    state = val_env.reset()
    done  = False
    agent.policy.eval()

    initial_value = val_env.balance
    portfolio_values = [initial_value]

    while not done:
        valid_actions = val_env.get_valid_actions()
        action        = agent.select_action(state, valid_actions=valid_actions, deterministic=True)
        state, _, done = val_env.step(action)
        portfolio_values.append(val_env.portfolio_value())

    agent.policy.train()

    price       = float(val_env.data.iloc[val_env.current_step]['Close'])
    final_value = val_env.balance + val_env.shares_held * price
    total_return = (final_value - initial_value) / (initial_value + 1e-8)
    benchmark_prices = val_env.data.iloc[val_env.window_size - 1 :]["Close"].to_numpy(dtype=float)
    benchmark_values = buy_and_hold_values(
        benchmark_prices,
        initial_value=initial_value,
        transaction_fee=val_env.transaction_fee,
    )
    return total_return, validation_score(
        portfolio_values, val_env.total_turnover, benchmark_values
    )


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
for episode in range(episodes):
    ticker, env = train_envs[episode % len(train_envs)]

    state        = env.reset()
    total_reward = 0.0
    done         = False

    while not done:
        valid_actions            = env.get_valid_actions()
        action, logprob, value   = agent.act(state, valid_actions=valid_actions)
        next_state, reward, done = env.step(action)

        agent.memory.add(state, action, logprob, reward, value, done, valid_actions)
        state        = next_state
        total_reward += reward
        timestep     += 1

        if timestep % update_timestep == 0:
            agent.train(next_state=next_state, next_done=done)

    if len(agent.memory):
        agent.train(next_state=state, next_done=True)

    reward_history.append(total_reward)
    avg_50 = np.mean(reward_history[-50:])

    # ---- Periodic validation ----
    val_pct = None
    val_score = None
    if (episode + 1) % val_every == 0:
        metrics = [validate_ppo(agent, ve) for _, ve in val_envs]
        returns = [result[0] for result in metrics]
        scores = [result[1] for result in metrics]
        mean_return = float(np.mean(returns))
        val_pct     = round(mean_return * 100, 2)
        val_score = round(float(np.mean(scores)), 6)
        if val_score > best_val_score + TRAIN_CONFIG["early_stopping_min_delta"]:
            best_val_score = val_score
            validations_without_improvement = 0
            save_checkpoint("models/ppo_best.pth", agent.policy.state_dict(), "ppo", state_size, window_size)
            print(f"  *** New best validation score: {val_score:.4f} ({val_pct:.2f}% return)")
        else:
            validations_without_improvement += 1

    log_writer.writerow([
        PIPELINE_VERSION,
        episode + 1, ticker,
        round(total_reward, 4),
        timestep,
        round(avg_50, 4),
        val_pct,
        val_score,
    ])
    log_file.flush()

    if (episode + 1) % save_every == 0:
        save_checkpoint(
            f"models/ppo_ep{episode+1}.pth",
            agent.policy.state_dict(),
            "ppo",
            state_size,
            window_size,
        )

    print(
        f"Episode {episode+1}/{episodes} [{ticker}] | Reward: {total_reward:.2f} | "
        f"Steps: {timestep} | Avg50: {avg_50:.2f}"
        + (f" | Val: {val_pct:.2f}%" if val_pct is not None else "")
    )

    if validations_without_improvement >= TRAIN_CONFIG["early_stopping_patience"]:
        print(
            f"Early stopping after {validations_without_improvement} validation checks "
            "without improvement"
        )
        break

log_file.close()

# ---- Promote best validation checkpoint as the final model ----
if os.path.exists("models/ppo_best.pth"):
    shutil.copy("models/ppo_best.pth", "models/ppo.pth")
    print("Copied best validation checkpoint -> models/ppo.pth")
else:
    save_checkpoint("models/ppo.pth", agent.policy.state_dict(), "ppo", state_size, window_size)
    print("No best checkpoint found; saving final episode -> models/ppo.pth")

print(f"Training log saved to {log_path}")
