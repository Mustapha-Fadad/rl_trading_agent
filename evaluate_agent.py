import numpy as np
import torch

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from agents.random_agent import RandomAgent
from data.data_loader import load_data, train_test_split
from env.trading_env import TradingEnv
from utils.config import COMMON, DATA_CONFIG, REWARD_CONFIG
from utils.checkpoints import PIPELINE_VERSION, load_checkpoint
from utils.metrics import buy_and_hold_values, portfolio_statistics


def _build_agent(algo, state_size, action_size, window_size, model_path):
    algo = algo.lower()

    if algo == "dqn":
        agent = DQNAgent(state_size, action_size=action_size, window_size=window_size)
        state_dict, metadata = load_checkpoint(
            model_path, agent.device, "dqn", allow_legacy=True, return_metadata=True
        )
        agent.model.load_state_dict(state_dict)
        agent.checkpoint_metadata = metadata
        agent.model.eval()
        agent.epsilon = 0.0
        return agent

    if algo == "ppo":
        agent = PPOAgent(state_size, action_size=action_size, window_size=window_size)
        state_dict, metadata = load_checkpoint(
            model_path, agent.device, "ppo", allow_legacy=True, return_metadata=True
        )
        agent.policy.load_state_dict(state_dict)
        agent.checkpoint_metadata = metadata
        agent.policy.eval()
        return agent

    if algo == "random":
        return RandomAgent()

    raise ValueError(f"Unsupported algo '{algo}'. Use 'dqn', 'ppo', or 'random'.")


def _select_action(agent, algo, state, valid_actions, deterministic=True):
    if algo == "dqn":
        return agent.act(state, valid_actions=valid_actions)
    if algo == "random":
        return agent.act(state, valid_actions=valid_actions)
    # ppo
    return agent.select_action(state, valid_actions=valid_actions, deterministic=deterministic)


def evaluate_agent(
    algo="dqn",
    model_path=None,
    ticker="AAPL",
    window_size=None,
    split="test",
    deterministic=True,
):
    """
    Evaluate a trained agent on train, test, or full data.

    Parameters
    ----------
    algo        : 'dqn', 'ppo', or 'random'
    model_path  : path to .pth checkpoint; ignored for 'random'
    ticker      : stock ticker symbol
    window_size : observation window length (defaults to COMMON["window_size"])
    split       : 'train', 'test', or 'all'
    deterministic: for PPO, use argmax instead of sampling
    """
    if window_size is None:
        window_size = COMMON["window_size"]

    data       = load_data(ticker, **DATA_CONFIG)
    price_mode = data.attrs.get("price_mode", "unknown")

    if split in ("train", "test"):
        train_data, test_data = train_test_split(data, train_ratio=COMMON["train_split"])
        eval_data = train_data if split == "train" else test_data
    else:
        eval_data = data

    reward_kwargs = REWARD_CONFIG.copy()
    env = TradingEnv(eval_data, window_size=window_size, **reward_kwargs)

    state_size  = env.reset().shape[0]
    action_size = COMMON["action_size"]

    algo_lower = algo.lower()
    if model_path is None and algo_lower not in ("random",):
        model_path = f"models/{algo_lower}.pth"

    agent = _build_agent(algo_lower, state_size, action_size, window_size, model_path)

    state = env.reset()
    done  = False

    portfolio_values = [env.balance]
    rewards = []
    action_counts = np.zeros(action_size, dtype=int)
    exposures = [env.position_frac]
    rebalance_count = 0

    while not done:
        action = _select_action(agent, algo_lower, state, env.get_valid_actions(), deterministic)
        next_state, reward, done = env.step(action)

        after_value = env.balance + env.shares_held * env.prev_price
        rewards.append(float(reward))
        portfolio_values.append(float(after_value))
        action_counts[action] += 1
        exposures.append(env.position_frac)
        if abs(env.last_trade_notional) >= 1.0:
            rebalance_count += 1

        state = next_state

    initial_value = portfolio_values[0]
    final_value   = portfolio_values[-1]
    agent_stats = portfolio_statistics(portfolio_values)
    benchmark_prices = eval_data.iloc[window_size - 1 :]["Close"].to_numpy(dtype=float)
    benchmark_values = buy_and_hold_values(
        benchmark_prices,
        initial_value=initial_value,
        transaction_fee=env.transaction_fee,
    )
    benchmark_stats = portfolio_statistics(benchmark_values)
    buy_hold_final = float(benchmark_values[-1])

    summary = {
        "algo":                     algo,
        "model_path":               model_path,
        "ticker":                   ticker,
        "split":                    split,
        "price_mode":               price_mode,
        "data_source":              data.attrs.get("source", "unknown"),
        "pipeline_version":         PIPELINE_VERSION,
        "checkpoint_pipeline_version": getattr(agent, "checkpoint_metadata", {}).get(
            "pipeline_version", "n/a"
        ),
        "steps":                    len(rewards),
        "initial_portfolio_value":  round(initial_value, 2),
        "final_portfolio_value":    round(final_value, 2),
        "agent_return_pct":         round(((final_value - initial_value) / initial_value) * 100, 2),
        "cash_return_pct":          0.0,
        "buy_and_hold_final_value": round(buy_hold_final, 2),
        "buy_and_hold_return_pct":  round(((buy_hold_final - initial_value) / initial_value) * 100, 2),
        "alpha_vs_buy_hold_pct":    round((final_value - buy_hold_final) / initial_value * 100, 2),
        "total_reward":             round(float(np.sum(rewards)), 4),
        "mean_step_reward":         round(float(np.mean(rewards)), 6),
        "max_drawdown_pct":         round(agent_stats["max_drawdown"] * 100, 2),
        "annualized_volatility_pct": round(agent_stats["annualized_volatility"] * 100, 2),
        "sharpe_ratio":             round(agent_stats["sharpe"], 3),
        "sortino_ratio":            round(agent_stats["sortino"], 3),
        "buy_and_hold_max_drawdown_pct": round(benchmark_stats["max_drawdown"] * 100, 2),
        "buy_and_hold_annualized_volatility_pct": round(benchmark_stats["annualized_volatility"] * 100, 2),
        "buy_and_hold_sharpe_ratio": round(benchmark_stats["sharpe"], 3),
        "buy_and_hold_sortino_ratio": round(benchmark_stats["sortino"], 3),
        "total_turnover":           round(float(env.total_turnover), 3),
        "rebalance_count":          rebalance_count,
        "average_exposure_pct":     round(float(np.mean(exposures)) * 100, 2),
        "target_0_count":           int(action_counts[0]),
        "target_25_count":          int(action_counts[1]),
        "target_50_count":          int(action_counts[2]),
        "target_75_count":          int(action_counts[3]),
        "target_100_count":         int(action_counts[4]),
    }

    if price_mode != "raw":
        summary["performance_note"] = (
            "Portfolio values are relative to legacy normalised prices, not real market dollars."
        )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",        choices=["dqn", "ppo", "random"], default="ppo")
    parser.add_argument("--model-path",  default=None)
    parser.add_argument("--ticker",      default="AAPL")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--split",       choices=["train", "test", "all"], default="test")
    parser.add_argument("--stochastic",  action="store_true",
                        help="For PPO: sample actions instead of deterministic argmax")
    args = parser.parse_args()

    results = evaluate_agent(
        algo=args.algo,
        model_path=args.model_path,
        ticker=args.ticker,
        window_size=args.window_size,
        split=args.split,
        deterministic=not args.stochastic,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
