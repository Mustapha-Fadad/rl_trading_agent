import os
import torch
import numpy as np
import optuna
from optuna.pruners import MedianPruner

from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from agents.ppo.agent import PPOAgent
from utils.config import COMMON, DATA_CONFIG, TRAIN_CONFIG, REWARD_CONFIG
from utils.reproducibility import set_seed
from utils.metrics import buy_and_hold_values, validation_score
from utils.validation import split_validation_windows

def objective(trial):
    # Set seed for reproducibility across trials
    set_seed(COMMON["seed"])

    # Optuna suggestions for Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.1, log=True)
    k_epochs = trial.suggest_int("k_epochs", 2, 10)
    ppo_update_timestep = trial.suggest_categorical("ppo_update_timestep", [128, 256, 512, 1024])

    print(f"Trial {trial.number} starting with lr={lr:.5f}, entropy={entropy_coef:.4f}, k_epochs={k_epochs}, batch={ppo_update_timestep}")

    # Load multi-ticker data & setup envs
    tickers = TRAIN_CONFIG["tickers"]
    reward_kwargs = REWARD_CONFIG.copy()

    train_envs = []
    val_envs = []

    for ticker in tickers:
        data = load_data(ticker, save=False, **DATA_CONFIG)
        train_data, _ = train_test_split(data, train_ratio=COMMON["train_split"])

        train_only, ticker_val_windows = split_validation_windows(
            train_data,
            observation_window=COMMON["window_size"],
            num_windows=TRAIN_CONFIG["val_windows"],
            window_rows=TRAIN_CONFIG["val_window_size"],
        )

        t_env = TradingEnv(
            train_only,
            window_size=COMMON["window_size"],
            random_start=True,
            episode_length=TRAIN_CONFIG["episode_length"],
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

    state_size = train_envs[0][1].reset().shape[0]
    action_size = COMMON["action_size"]
    window_size = COMMON["window_size"]

    # Instantiate Agent with Trial specs
    agent = PPOAgent(state_size, action_size=action_size, window_size=window_size)
    agent.lr = lr
    agent.entropy_coef = entropy_coef
    agent.k_epochs = k_epochs
    # Reset optimizer with new LR
    agent.optimizer = torch.optim.Adam([
        {'params': agent.policy.lstm.parameters(), 'lr': lr * 0.1},
        {'params': agent.policy.scalar_net.parameters(), 'lr': lr * 0.1},
        {'params': agent.policy.actor_head.parameters(), 'lr': lr},
        {'params': agent.policy.critic_head.parameters(), 'lr': lr}
    ])

    episodes = 60  # Shortened episode count for rapid iteration
    val_every = 15
    timestep = 0


    for episode in range(episodes):
        ticker, env = train_envs[episode % len(train_envs)]
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action, logprob, value = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action)

            agent.memory.add(state, action, logprob, reward, value, done, valid_actions)
            state = next_state
            total_reward += reward
            timestep += 1

            if timestep % ppo_update_timestep == 0:
                agent.train(next_state=next_state, next_done=done)

        if len(agent.memory):
            agent.train(next_state=state, next_done=True)

        # Validation Step
        if (episode + 1) % val_every == 0:
            agent.policy.eval()
            scores = []
            for _, v_env in val_envs:
                v_state = v_env.reset()
                v_done = False
                v_initial_val = v_env.balance
                portfolio_values = [v_initial_val]
                while not v_done:
                    v_actions = v_env.get_valid_actions()
                    v_act = agent.select_action(v_state, valid_actions=v_actions, deterministic=True)
                    v_state, _, v_done = v_env.step(v_act)
                    portfolio_values.append(v_env.portfolio_value())

                benchmark_prices = v_env.data.iloc[
                    v_env.window_size - 1 :
                ]["Close"].to_numpy(dtype=float)
                benchmark_values = buy_and_hold_values(
                    benchmark_prices,
                    initial_value=v_initial_val,
                    transaction_fee=v_env.transaction_fee,
                )
                scores.append(validation_score(
                    portfolio_values, v_env.total_turnover, benchmark_values
                ))

            agent.policy.train()
            val_score = float(np.mean(scores))

            # Optuna Pruning check
            trial.report(val_score, episode)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Return the validation score of the fully trained agent at the end of 60 episodes,
    # rather than the max seen over time, to ensure Optuna evaluates the final learning capacity.
    return val_score


if __name__ == "__main__":
    print("Initializing Optuna hyperparameter sweep for PPO...")
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    # Run 2 trials to test plumbing. In production, n_trials=50+
    study.optimize(objective, n_trials=2, n_jobs=1)

    print("\n==============================")
    print("Optimization finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best Outperformance (Alpha): +{best_trial.value * 100:.2f}% over Benchmark")
    print("Best Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("==============================\n")
