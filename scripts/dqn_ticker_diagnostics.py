import argparse
import csv
import os

import numpy as np
import pandas as pd

from agents.dqn.agent import DQNAgent
from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from utils.checkpoints import PIPELINE_VERSION
from utils.config import COMMON, DATA_CONFIG, REWARD_CONFIG, TRAIN_CONFIG
from utils.metrics import buy_and_hold_values, portfolio_statistics, validation_score
from utils.reproducibility import set_seed
from utils.validation import split_validation_windows


REPORT_VERSION = 1
ALLOCATION_COLUMNS = [
    "allocation_0_pct",
    "allocation_25_pct",
    "allocation_50_pct",
    "allocation_75_pct",
    "allocation_100_pct",
]
FIELDNAMES = [
    "report_version",
    "pipeline_version",
    "ticker",
    "episodes",
    "val_windows",
    "val_window_size",
    "seed",
    "validation_window",
    "validation_start",
    "validation_end",
    "agent_return_pct",
    "bh_return_pct",
    "alpha_pct",
    "max_drawdown_pct",
    "bh_max_drawdown_pct",
    "sharpe_ratio",
    "bh_sharpe_ratio",
    "validation_score",
    "turnover",
    "rebalances",
    "average_exposure_pct",
    *ALLOCATION_COLUMNS,
]


def action_distribution(actions, action_size=COMMON["action_size"]):
    actions = np.asarray(actions, dtype=int)
    if actions.size == 0:
        raise ValueError("at least one action is required")
    counts = np.bincount(actions, minlength=action_size)[:action_size]
    return counts / actions.size * 100.0


def train_ticker_agent(train_data, episodes, seed):
    set_seed(seed)
    env = TradingEnv(
        train_data,
        window_size=COMMON["window_size"],
        random_start=TRAIN_CONFIG["random_start"],
        episode_length=TRAIN_CONFIG["episode_length"],
        **REWARD_CONFIG,
    )
    agent = DQNAgent(
        env.reset().shape[0],
        action_size=COMMON["action_size"],
        window_size=COMMON["window_size"],
    )

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, valid_actions=env.get_valid_actions())
            next_state, reward, done = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            agent.train()
            state = next_state
        agent.step_scheduler()

    return agent


def evaluate_window(
    agent,
    ticker,
    episodes,
    val_windows,
    val_window_size,
    seed,
    window_index,
    data,
):
    env = TradingEnv(
        data,
        window_size=COMMON["window_size"],
        random_start=False,
        **REWARD_CONFIG,
    )
    state = env.reset()
    portfolio_values = [env.balance]
    exposures = [env.position_frac]
    actions = []
    rebalances = 0
    done = False

    previous_epsilon = agent.epsilon
    was_training = agent.model.training
    agent.epsilon = 0.0
    agent.model.eval()
    try:
        while not done:
            action = agent.act(state, valid_actions=env.get_valid_actions())
            state, _, done = env.step(action)
            actions.append(action)
            portfolio_values.append(env.portfolio_value())
            exposures.append(env.position_frac)
            if abs(env.last_trade_notional) >= 1.0:
                rebalances += 1
    finally:
        agent.epsilon = previous_epsilon
        agent.model.train(was_training)

    benchmark_prices = data.iloc[
        env.window_size - 1 :
    ]["Close"].to_numpy(dtype=float)
    benchmark_values = buy_and_hold_values(
        benchmark_prices,
        initial_value=portfolio_values[0],
        transaction_fee=env.transaction_fee,
    )
    agent_stats = portfolio_statistics(portfolio_values)
    benchmark_stats = portfolio_statistics(benchmark_values)
    allocation_pcts = action_distribution(actions)

    result = {
        "report_version": REPORT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "ticker": ticker,
        "episodes": episodes,
        "val_windows": val_windows,
        "val_window_size": val_window_size,
        "seed": seed,
        "validation_window": window_index,
        "validation_start": str(data.index[env.window_size - 1].date()),
        "validation_end": str(data.index[-1].date()),
        "agent_return_pct": agent_stats["return"] * 100.0,
        "bh_return_pct": benchmark_stats["return"] * 100.0,
        "alpha_pct": (agent_stats["return"] - benchmark_stats["return"]) * 100.0,
        "max_drawdown_pct": agent_stats["max_drawdown"] * 100.0,
        "bh_max_drawdown_pct": benchmark_stats["max_drawdown"] * 100.0,
        "sharpe_ratio": agent_stats["sharpe"],
        "bh_sharpe_ratio": benchmark_stats["sharpe"],
        "validation_score": validation_score(
            portfolio_values, env.total_turnover, benchmark_values
        ),
        "turnover": env.total_turnover,
        "rebalances": rebalances,
        "average_exposure_pct": float(np.mean(exposures)) * 100.0,
    }
    result.update(dict(zip(ALLOCATION_COLUMNS, allocation_pcts)))
    return result


def _validate_existing_report(
    report, ticker, episodes, expected_windows, val_window_size
):
    if report.empty:
        return set()
    expected = {
        "report_version": REPORT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "ticker": ticker,
        "episodes": episodes,
        "val_windows": expected_windows,
        "val_window_size": val_window_size,
    }
    for column, value in expected.items():
        actual = set(report[column].tolist())
        if actual != {value}:
            raise ValueError(
                f"existing report has incompatible {column}: {sorted(actual)}"
            )

    counts = report.groupby("seed")["validation_window"].nunique()
    incomplete = counts[counts != expected_windows]
    if not incomplete.empty:
        raise ValueError("existing report contains an incomplete seed; use a new output path")
    return set(int(seed) for seed in counts.index)


def run_diagnostics(
    ticker="AAPL",
    episodes=120,
    seeds=3,
    seed_start=COMMON["seed"],
    val_windows=TRAIN_CONFIG["val_windows"],
    val_window_size=TRAIN_CONFIG["val_window_size"],
    output_path="reports/dqn_AAPL_diagnostics.csv",
):
    data = load_data(ticker, **DATA_CONFIG)
    train_data, _ = train_test_split(data, train_ratio=COMMON["train_split"])
    train_only, windows = split_validation_windows(
        train_data,
        observation_window=COMMON["window_size"],
        num_windows=val_windows,
        window_rows=val_window_size,
    )
    print(
        f"{ticker}: {len(train_only)} ticker-specific training rows / "
        f"{len(windows)} x {val_window_size}-day validation windows"
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        report = pd.read_csv(output_path)
        completed_seeds = _validate_existing_report(
            report, ticker, episodes, val_windows, val_window_size
        )
    else:
        report = pd.DataFrame(columns=FIELDNAMES)
        completed_seeds = set()
        with open(output_path, "w", newline="") as output_file:
            csv.DictWriter(output_file, fieldnames=FIELDNAMES).writeheader()

    for seed_offset in range(seeds):
        seed = seed_start + seed_offset
        if seed in completed_seeds:
            print(f"Seed {seed} already complete; skipping")
            continue

        print(f"Training {ticker}-only DQN | seed {seed} | {episodes} episodes...", flush=True)
        agent = train_ticker_agent(train_only, episodes, seed)
        rows = [
            evaluate_window(
                agent,
                ticker,
                episodes,
                val_windows,
                val_window_size,
                seed,
                index + 1,
                window,
            )
            for index, window in enumerate(windows)
        ]
        with open(output_path, "a", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
            writer.writerows(rows)
        mean_score = float(np.mean([row["validation_score"] for row in rows]))
        mean_alpha = float(np.mean([row["alpha_pct"] for row in rows]))
        print(
            f"  seed {seed}: mean score={mean_score:+.6f}, "
            f"mean alpha={mean_alpha:+.2f}%",
            flush=True,
        )

    report = pd.read_csv(output_path)
    requested_seeds = set(range(seed_start, seed_start + seeds))
    report = report[report["seed"].isin(requested_seeds)]
    seed_scores = report.groupby("seed")["validation_score"].mean()
    mean_score = float(seed_scores.mean())
    score_std = float(seed_scores.std()) if len(seed_scores) > 1 else 0.0
    robust_score = mean_score - 0.5 * score_std

    print("\n========================================")
    print(f"Ticker-specific DQN diagnostic: {ticker}")
    print(f"Completed seeds: {len(seed_scores)}/{seeds}")
    print(f"Mean validation score: {mean_score:+.6f}")
    print(f"Seed score std:       {score_std:.6f}")
    print(f"Robust score:         {robust_score:+.6f}")
    print(f"Mean alpha:           {report['alpha_pct'].mean():+.2f}%")
    print(
        f"Mean drawdown:        {report['max_drawdown_pct'].mean():.2f}% "
        f"(B&H {report['bh_max_drawdown_pct'].mean():.2f}%)"
    )
    print(
        f"Mean Sharpe:          {report['sharpe_ratio'].mean():.3f} "
        f"(B&H {report['bh_sharpe_ratio'].mean():.3f})"
    )
    print(f"Mean exposure:        {report['average_exposure_pct'].mean():.2f}%")
    print(f"Results saved to {output_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and diagnose a ticker-specific DQN on training-only regimes"
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=COMMON["seed"])
    parser.add_argument("--val-windows", type=int, default=TRAIN_CONFIG["val_windows"])
    parser.add_argument(
        "--val-window-size", type=int, default=TRAIN_CONFIG["val_window_size"]
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1 or args.seeds < 1 or args.val_windows < 1:
        raise ValueError("episodes, seeds, and val-windows must be positive")
    run_diagnostics(
        ticker=args.ticker.upper(),
        episodes=args.episodes,
        seeds=args.seeds,
        seed_start=args.seed_start,
        val_windows=args.val_windows,
        val_window_size=args.val_window_size,
        output_path=args.output or f"reports/dqn_{args.ticker.upper()}_diagnostics.csv",
    )


if __name__ == "__main__":
    main()
