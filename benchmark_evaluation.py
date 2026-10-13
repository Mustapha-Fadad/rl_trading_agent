import csv
import os
import numpy as np

from evaluate_agent import evaluate_agent
from utils.config import TRAIN_CONFIG
from utils.reproducibility import set_seed


RANDOM_RUNS = 30


def evaluate_random_baseline(ticker):
    runs = []
    for seed in range(RANDOM_RUNS):
        set_seed(seed)
        runs.append(evaluate_agent(algo="random", ticker=ticker, split="test"))

    result = runs[0].copy()
    result["algo"] = "random_mean"
    result["random_runs"] = RANDOM_RUNS
    for key in (
        "agent_return_pct",
        "max_drawdown_pct",
        "annualized_volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "buy_and_hold_max_drawdown_pct",
        "buy_and_hold_annualized_volatility_pct",
        "buy_and_hold_sharpe_ratio",
        "buy_and_hold_sortino_ratio",
        "total_turnover",
        "rebalance_count",
        "average_exposure_pct",
        "target_0_count",
        "target_25_count",
        "target_50_count",
        "target_75_count",
        "target_100_count",
    ):
        result[key] = round(float(np.mean([run[key] for run in runs])), 3)
    result["return_std_pct"] = round(
        float(np.std([run["agent_return_pct"] for run in runs], ddof=1)), 3
    )
    result["alpha_vs_buy_hold_pct"] = round(
        result["agent_return_pct"] - result["buy_and_hold_return_pct"], 3
    )
    return result


def main():
    os.makedirs("reports", exist_ok=True)

    rows = []

    # ---- Trained agents ----
    for algo in ("dqn", "ppo"):
        for ticker in TRAIN_CONFIG["tickers"]:
            rows.append(evaluate_agent(
                algo=algo,
                model_path=f"models/{algo}.pth",
                ticker=ticker,
                split="test"
            ))

    # ---- Multi-seed random-agent baseline ----
    for ticker in TRAIN_CONFIG["tickers"]:
        rows.append(evaluate_random_baseline(ticker))

    output_path = os.path.join("reports", "evaluation_summary.csv")
    fieldnames = [
        "algo",
        "ticker",
        "split",
        "price_mode",
        "data_source",
        "pipeline_version",
        "checkpoint_pipeline_version",
        "steps",
        "agent_return_pct",
        "return_std_pct",
        "cash_return_pct",
        "buy_and_hold_return_pct",
        "alpha_vs_buy_hold_pct",
        "max_drawdown_pct",
        "annualized_volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "buy_and_hold_max_drawdown_pct",
        "buy_and_hold_annualized_volatility_pct",
        "buy_and_hold_sharpe_ratio",
        "buy_and_hold_sortino_ratio",
        "total_turnover",
        "rebalance_count",
        "average_exposure_pct",
        "target_0_count",
        "target_25_count",
        "target_50_count",
        "target_75_count",
        "target_100_count",
        "random_runs",
        "total_reward",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Saved benchmark results to {output_path}")
    print()
    print(
        "algo,ticker,agent_return_pct,buy_and_hold_return_pct,"
        "max_drawdown_pct,buy_and_hold_max_drawdown_pct,"
        "sharpe_ratio,buy_and_hold_sharpe_ratio,total_turnover"
    )
    for row in rows:
        print(
            f"{row['algo']},{row['ticker']},"
            f"{row['agent_return_pct']},"
            f"{row['buy_and_hold_return_pct']},"
            f"{row['max_drawdown_pct']},"
            f"{row['buy_and_hold_max_drawdown_pct']},"
            f"{row['sharpe_ratio']},"
            f"{row['buy_and_hold_sharpe_ratio']},"
            f"{row['total_turnover']}"
        )


if __name__ == "__main__":
    main()
