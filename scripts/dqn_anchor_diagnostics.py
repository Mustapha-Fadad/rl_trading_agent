import argparse
import csv
import os

import numpy as np
import pandas as pd

from data.data_loader import load_data, train_test_split
from scripts.dqn_ticker_diagnostics import (
    FIELDNAMES as BASE_FIELDNAMES,
    evaluate_window,
    train_ticker_agent,
)
from utils.config import COMMON, DATA_CONFIG, TRAIN_CONFIG
from utils.validation import split_validation_windows


REPORT_VERSION = 1
ANCHOR_FIELDS = [
    "anchor_action",
    "anchor_margin",
    "anchor_bias",
    "underexposure_penalty",
]
FIELDNAMES = BASE_FIELDNAMES + ANCHOR_FIELDS


def _protocol(args):
    return {
        "ticker": args.ticker,
        "episodes": args.episodes,
        "val_windows": args.val_windows,
        "val_window_size": args.val_window_size,
        "anchor_action": 4,
        "anchor_margin": args.anchor_margin,
        "anchor_bias": args.anchor_bias,
        "underexposure_penalty": args.underexposure_penalty,
    }


def _completed_seeds(report, protocol):
    if report.empty:
        return set()
    for column, expected in protocol.items():
        actual = set(report[column].tolist())
        if actual != {expected}:
            raise ValueError(
                f"existing report has incompatible {column}: {sorted(actual)}"
            )
    counts = report.groupby("seed")["validation_window"].nunique()
    incomplete = counts[counts != protocol["val_windows"]]
    if not incomplete.empty:
        raise ValueError("existing report contains an incomplete seed; use a new output path")
    return set(int(seed) for seed in counts.index)


def run_anchor_diagnostics(args):
    protocol = _protocol(args)
    data = load_data(args.ticker, **DATA_CONFIG)
    train_data, _ = train_test_split(data, train_ratio=COMMON["train_split"])
    train_only, windows = split_validation_windows(
        train_data,
        observation_window=COMMON["window_size"],
        num_windows=args.val_windows,
        window_rows=args.val_window_size,
    )
    print(
        f"{args.ticker}: {len(train_only)} anchor-training rows / "
        f"{len(windows)} x {args.val_window_size}-day validation windows"
    )
    print(
        f"Anchor: 100% allocation | margin={args.anchor_margin} | "
        f"bias={args.anchor_bias} | underexposure={args.underexposure_penalty}"
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(args.output):
        report = pd.read_csv(args.output)
        completed = _completed_seeds(report, protocol)
    else:
        completed = set()
        with open(args.output, "w", newline="") as output_file:
            csv.DictWriter(output_file, fieldnames=FIELDNAMES).writeheader()

    config_overrides = {
        "anchor_action": 4,
        "anchor_margin": args.anchor_margin,
        "anchor_bias": args.anchor_bias,
        "anchor_initialize": True,
    }
    reward_overrides = {
        "underexposure_penalty": args.underexposure_penalty,
    }
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if seed in completed:
            print(f"Seed {seed} already complete; skipping")
            continue
        print(
            f"Training anchored {args.ticker} DQN | seed {seed} | "
            f"{args.episodes} episodes...",
            flush=True,
        )
        agent = train_ticker_agent(
            train_only,
            args.episodes,
            seed,
            config_overrides=config_overrides,
            reward_overrides=reward_overrides,
        )
        rows = []
        for index, window in enumerate(windows):
            row = evaluate_window(
                agent,
                args.ticker,
                args.episodes,
                args.val_windows,
                args.val_window_size,
                seed,
                index + 1,
                window,
            )
            row.update(protocol)
            row["report_version"] = REPORT_VERSION
            rows.append(row)
        with open(args.output, "a", newline="") as output_file:
            csv.DictWriter(output_file, fieldnames=FIELDNAMES).writerows(rows)
        print(
            f"  seed {seed}: score="
            f"{np.mean([row['validation_score'] for row in rows]):+.6f}, "
            f"alpha={np.mean([row['alpha_pct'] for row in rows]):+.2f}%",
            flush=True,
        )

    report = pd.read_csv(args.output)
    requested = set(range(args.seed_start, args.seed_start + args.seeds))
    report = report[report["seed"].isin(requested)]
    seed_scores = report.groupby("seed")["validation_score"].mean()
    mean_score = float(seed_scores.mean())
    score_std = float(seed_scores.std()) if len(seed_scores) > 1 else 0.0

    print("\n========================================")
    print(f"Buy & Hold anchored DQN diagnostic: {args.ticker}")
    print(f"Completed seeds: {len(seed_scores)}/{args.seeds}")
    print(f"Mean validation score: {mean_score:+.6f}")
    print(f"Seed score std:       {score_std:.6f}")
    print(f"Robust score:         {mean_score - 0.5 * score_std:+.6f}")
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
    print(f"Results saved to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose a DQN conservatively anchored to Buy & Hold"
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=COMMON["seed"])
    parser.add_argument("--val-windows", type=int, default=2)
    parser.add_argument(
        "--val-window-size", type=int, default=TRAIN_CONFIG["val_window_size"]
    )
    parser.add_argument("--anchor-margin", type=float, default=0.10)
    parser.add_argument("--anchor-bias", type=float, default=0.05)
    parser.add_argument("--underexposure-penalty", type=float, default=0.0001)
    parser.add_argument("--output", default="reports/dqn_AAPL_anchor_pilot.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    args.ticker = args.ticker.upper()
    if args.episodes < 1 or args.seeds < 1 or args.val_windows < 1:
        raise ValueError("episodes, seeds, and val-windows must be positive")
    if min(args.anchor_margin, args.anchor_bias, args.underexposure_penalty) < 0:
        raise ValueError("anchor settings must be non-negative")
    run_anchor_diagnostics(args)


if __name__ == "__main__":
    main()
