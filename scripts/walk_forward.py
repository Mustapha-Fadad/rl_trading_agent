# scripts/walk_forward.py

import argparse
import csv
import os
import numpy as np
import pandas as pd
import torch

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from agents.rule_based import RuleBasedAgent
from data.data_loader import load_data
from env.trading_env import TradingEnv
from utils.config import COMMON, DATA_CONFIG, REWARD_CONFIG, TRAIN_CONFIG
from utils.checkpoints import PIPELINE_VERSION
from utils.reproducibility import set_seed
from utils.metrics import buy_and_hold_values, portfolio_statistics


def evaluate_on_env(agent, algo, env):
    """Run a deterministic evaluation on a test environment and calculate key metrics."""
    state = env.reset()
    done = False

    portfolio_values = [env.balance]
    exposures = [env.position_frac]
    rebalances = 0

    # Set to evaluation mode
    if algo == "dqn":
        previous_epsilon = agent.epsilon
        was_training = agent.model.training
        agent.model.eval()
        agent.epsilon = 0.0
    elif algo == "ppo":
        was_training = agent.policy.training
        agent.policy.eval()

    while not done:
        valid_actions = env.get_valid_actions()

        if algo == "dqn":
            action = agent.act(state, valid_actions=valid_actions)
        elif algo == "ppo":
            action = agent.select_action(state, valid_actions=valid_actions, deterministic=True)
        else:
            action = agent.act(env.data.iloc[env.current_step], valid_actions)

        next_state, reward, done = env.step(action)
        after_value = env.balance + env.shares_held * env.prev_price
        portfolio_values.append(float(after_value))
        exposures.append(env.position_frac)
        if abs(env.last_trade_notional) >= 1.0:
            rebalances += 1

        state = next_state

    # Calculate statistics
    initial_val = portfolio_values[0]
    final_val = portfolio_values[-1]
    agent_return = (final_val - initial_val) / initial_val

    agent_stats = portfolio_statistics(portfolio_values)
    benchmark_prices = env.data.iloc[env.window_size - 1 :]["Close"].to_numpy(dtype=float)
    benchmark_values = buy_and_hold_values(
        benchmark_prices, initial_value=initial_val, transaction_fee=env.transaction_fee
    )
    benchmark_stats = portfolio_statistics(benchmark_values)
    bh_return = benchmark_stats["return"]

    if algo == "dqn":
        agent.epsilon = previous_epsilon
        agent.model.train(was_training)
    elif algo == "ppo":
        agent.policy.train(was_training)

    return {
        "agent_return_pct": round(agent_return * 100, 2),
        "bh_return_pct": round(bh_return * 100, 2),
        "alpha_pct": round((agent_return - bh_return) * 100, 2),
        "max_drawdown_pct": round(agent_stats["max_drawdown"] * 100, 2),
        "sharpe_ratio": round(agent_stats["sharpe"], 3),
        "bh_max_drawdown_pct": round(benchmark_stats["max_drawdown"] * 100, 2),
        "bh_sharpe_ratio": round(benchmark_stats["sharpe"], 3),
        "turnover": round(float(env.total_turnover), 3),
        "rebalances": rebalances,
        "average_exposure_pct": round(float(np.mean(exposures)) * 100, 2),
    }


def train_dqn_fold(train_env, episodes=150):
    state_size = train_env.reset().shape[0]
    action_size = COMMON["action_size"]
    window_size = COMMON["window_size"]

    agent = DQNAgent(state_size, action_size=action_size, window_size=window_size)
    for ep in range(episodes):
        state = train_env.reset()
        done = False
        while not done:
            valid_actions = train_env.get_valid_actions()
            action = agent.act(state, valid_actions=valid_actions)
            next_state, reward, done = train_env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            agent.train()
        agent.step_scheduler()

    return agent


def train_ppo_fold(train_env, episodes=200):
    state_size = train_env.reset().shape[0]
    action_size = COMMON["action_size"]
    window_size = COMMON["window_size"]

    agent = PPOAgent(state_size, action_size=action_size, window_size=window_size)
    update_timestep = 128
    timestep = 0

    for ep in range(episodes):
        state = train_env.reset()
        done = False
        while not done:
            valid_actions = train_env.get_valid_actions()
            action, logprob, value = agent.act(state, valid_actions)
            next_state, reward, done = train_env.step(action)
            agent.memory.add(state, action, logprob, reward, value, done, valid_actions)
            state = next_state
            timestep += 1
            if timestep % update_timestep == 0:
                agent.train(next_state=next_state, next_done=done)

        if len(agent.memory):
            agent.train(next_state=state, next_done=True)

    return agent


def run_walk_forward(
    algo="dqn",
    ticker="AAPL",
    train_days=1000,
    test_days=252,
    step_days=252,
    episodes=100,
    seeds=1,
    output_path="reports/walk_forward_results.csv",
    max_folds=None,
):
    df = load_data(ticker, **DATA_CONFIG)
    total_len = len(df)

    reward_kwargs = REWARD_CONFIG.copy()

    folds = []
    start_idx = 0
    fold_id = 1

    while start_idx + train_days + test_days <= total_len:
        train_df = df.iloc[start_idx : start_idx + train_days].copy()
        test_df = df.iloc[start_idx + train_days : start_idx + train_days + test_days].copy()

        train_df.attrs = df.attrs.copy()
        test_df.attrs = df.attrs.copy()

        folds.append((fold_id, train_df, test_df))
        fold_id += 1
        start_idx += step_days

    if not folds:
        print(f"Dataset length ({total_len}) too short for train_days={train_days} + test_days={test_days}")
        return []
    if max_folds is not None:
        folds = folds[:max_folds]

    print(f"\n=======================================================")
    print(f"Starting Walk-Forward Validation | Ticker: {ticker} | Folds: {len(folds)}")
    print(f"Train Window: {train_days} days | Test Window: {test_days} days | Step: {step_days} days")
    print(f"Independent seeds per fold: {seeds}")
    print(f"=======================================================\n")

    algo_groups = {
        "both": ["dqn", "ppo"],
        "baselines": ["sma200", "trend_vol"],
        "all": ["dqn", "ppo", "sma200", "trend_vol"],
    }
    algos_to_run = algo_groups.get(algo.lower(), [algo.lower()])
    results = []
    fieldnames = ["fold", "seed", "algo", "ticker", "pipeline_version", "train_days", "test_days", "agent_return_pct", "bh_return_pct", "alpha_pct", "max_drawdown_pct", "sharpe_ratio", "bh_max_drawdown_pct", "bh_sharpe_ratio", "turnover", "rebalances", "average_exposure_pct"]
    out_path = output_path
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", newline="") as result_file:
        csv.DictWriter(result_file, fieldnames=fieldnames).writeheader()

    for fid, train_df, test_df in folds:
        for current_algo in algos_to_run:
            seed_offsets = range(seeds) if current_algo in ("dqn", "ppo") else range(1)
            for seed_offset in seed_offsets:
                run_seed = COMMON["seed"] + seed_offset
                set_seed(run_seed)
                train_env = TradingEnv(
                    train_df,
                    window_size=COMMON["window_size"],
                    random_start=True,
                    episode_length=min(
                        TRAIN_CONFIG.get("episode_length", 252),
                        train_days - COMMON["window_size"],
                    ),
                    **reward_kwargs,
                )
                test_env = TradingEnv(
                    test_df,
                    window_size=COMMON["window_size"],
                    random_start=False,
                    **reward_kwargs,
                )
                if current_algo in ("dqn", "ppo"):
                    run_description = f"Seed: {run_seed} | Training on {len(train_df)} rows"
                else:
                    run_description = "Deterministic rule evaluation"
                print(
                    f"--> Fold {fid}/{len(folds)} | Algo: {current_algo.upper()} | "
                    f"{run_description}..."
                )

                if current_algo == "dqn":
                    agent = train_dqn_fold(train_env, episodes=episodes)
                elif current_algo == "ppo":
                    agent = train_ppo_fold(train_env, episodes=max(episodes, 200))
                else:
                    agent = RuleBasedAgent(current_algo)

                metrics = evaluate_on_env(agent, current_algo, test_env)

                res_entry = {
                    "fold": fid,
                    "seed": run_seed,
                    "algo": current_algo,
                    "ticker": ticker,
                    "pipeline_version": PIPELINE_VERSION,
                    "train_days": len(train_df),
                    "test_days": len(test_df),
                    **metrics,
                }
                results.append(res_entry)
                with open(out_path, "a", newline="") as result_file:
                    writer = csv.DictWriter(result_file, fieldnames=fieldnames)
                    writer.writerow(res_entry)
                print(
                    f"    Fold {fid} [{current_algo.upper()}] Result: "
                    f"Agent = {metrics['agent_return_pct']}%, "
                    f"B&H = {metrics['bh_return_pct']}%, Alpha = {metrics['alpha_pct']}%"
                )


    print(f"\nSaved Walk-Forward results to {out_path}\n")

    # Print Summary Table
    df_res = pd.DataFrame(results)
    print("=======================================================")
    print("WALK-FORWARD AGGREGATE SUMMARY")
    print("=======================================================")
    for current_algo in algos_to_run:
        sub = df_res[df_res["algo"] == current_algo]
        fold_means = sub.groupby("fold", as_index=False).mean(numeric_only=True)
        mean_agent = fold_means["agent_return_pct"].mean()
        std_agent = fold_means["agent_return_pct"].std()
        mean_bh = fold_means["bh_return_pct"].mean()
        mean_alpha = fold_means["alpha_pct"].mean()
        mean_drawdown = fold_means["max_drawdown_pct"].mean()
        mean_bh_drawdown = fold_means["bh_max_drawdown_pct"].mean()
        mean_sharpe = fold_means["sharpe_ratio"].mean()
        mean_bh_sharpe = fold_means["bh_sharpe_ratio"].mean()
        win_folds = (fold_means["alpha_pct"] > 0).sum()

        print(f"[{current_algo.upper()}] Mean Agent Return: {mean_agent:.2f}% +/- {std_agent:.2f}%")
        print(f"[{current_algo.upper()}] Mean B&H Return:   {mean_bh:.2f}%")
        print(f"[{current_algo.upper()}] Mean Alpha:        {mean_alpha:+.2f}%")
        print(f"[{current_algo.upper()}] Mean Max Drawdown: {mean_drawdown:.2f}% (B&H {mean_bh_drawdown:.2f}%)")
        print(f"[{current_algo.upper()}] Mean Sharpe:       {mean_sharpe:.3f} (B&H {mean_bh_sharpe:.3f})")
        print(f"[{current_algo.upper()}] Folds Outperformed B&H: {win_folds}/{len(fold_means)}")
        print("-------------------------------------------------------")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Walk-Forward Validation for RL Trading Agents")
    parser.add_argument(
        "--algo",
        choices=["dqn", "ppo", "sma200", "trend_vol", "both", "baselines", "all"],
        default="both",
        help="Algorithm or algorithm group to evaluate",
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker to run walk-forward on")
    parser.add_argument("--train-days", type=int, default=1000, help="Number of training days per fold")
    parser.add_argument("--test-days", type=int, default=252, help="Number of test days per fold")
    parser.add_argument("--step-days", type=int, default=252, help="Slide step size in days between folds")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes per fold")
    parser.add_argument("--seeds", type=int, default=1, help="Independent training seeds per fold")
    parser.add_argument(
        "--output",
        default="reports/walk_forward_results.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit folds for a quick ablation run",
    )
    args = parser.parse_args()

    run_walk_forward(
        algo=args.algo,
        ticker=args.ticker,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        episodes=args.episodes,
        seeds=max(1, args.seeds),
        output_path=args.output,
        max_folds=args.max_folds,
    )
