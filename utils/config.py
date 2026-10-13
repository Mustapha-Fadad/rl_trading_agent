# utils/config.py

COMMON = {
    "gamma": 0.99,
    "window_size": 10,
    "initial_balance": 10000,
    "train_split": 0.8,
    "seed": 42,
    "action_size": 5,   # target allocations: 0%, 25%, 50%, 75%, 100%
}

DATA_CONFIG = {
    "start": "2010-01-01",
    "end": "2025-01-01",
    "source": "yfinance",
}

DQN_CONFIG = {
    "lr": 0.0003,
    "lr_decay_step": 50,        # decay lr every N episodes (was 100)
    "lr_decay_gamma": 0.95,     # multiply lr by this factor (was 0.5, too aggressive)
    "epsilon": 1.0,
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.05,
    "batch_size": 64,
    "min_replay_size": 1000,
    "replay_buffer_size": 50000,
    "tau": 0.005,               # soft target update rate
    "double_dqn": True,         # use Double DQN
    "dropout": 0.2,
}

PPO_CONFIG = {
    "lr": 0.0001,
    "eps_clip": 0.2,
    "gae_lambda": 0.95,
    "k_epochs": 4,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "dropout": 0.0,        # PPO ratios require deterministic policy logits
}

REWARD_CONFIG = {
    "reward_scale": 10.0,
    "turnover_penalty": 0.002,
    "drawdown_penalty": 0.002,
    "rebalance_penalty": 0.0005,
    "transaction_fee": 0.001,
    "rebalance_threshold": 0.02,
    "min_rebalance_steps": 20,
}

TRAIN_CONFIG = {
    "dqn_episodes": 400,
    "ppo_episodes": 500,
    "ppo_update_timestep": 128,  # Tuned via Optuna (was 256)
    "save_every": 50,
    "val_every": 50,             # run greedy validation every N episodes
    "val_split": 0.1,            # fraction of train data reserved for validation
    "val_windows": 3,            # score checkpoints across distinct market regimes
    "val_window_size": 252,
    "early_stopping_patience": 4,
    "early_stopping_min_delta": 0.0001,
    "log_dir": "logs",
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "random_start": True,
    "episode_length": 252,
}
