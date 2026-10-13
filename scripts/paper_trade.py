import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from dotenv import load_dotenv

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from data.alpaca_loader import load_alpaca_data
from data.features import FEATURE_COLUMNS, feature_vector
from env.trading_env import TARGET_ALLOCATIONS
from utils.checkpoints import PIPELINE_VERSION, load_checkpoint
from utils.config import COMMON, REWARD_CONFIG

load_dotenv()

ACTIONS = {index: f"TARGET_{int(target * 100)}" for index, target in enumerate(TARGET_ALLOCATIONS)}
LOG_PATH = "logs/paper_trading_v6.csv"
STATE_PATH = "logs/paper_trading_state_v6.json"


def get_alpaca_clients():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    return (
        TradingClient(api_key, secret_key, paper=True),
        StockHistoricalDataClient(api_key, secret_key),
    )


def build_state_from_df(
    df,
    window_size=COMMON["window_size"],
    position_frac=0.0,
    balance_norm=1.0,
    current_drawdown=0.0,
    cooldown_progress=1.0,
):
    if len(df) < window_size:
        raise ValueError(f"at least {window_size} completed bars are required")

    window = df.iloc[-window_size:]["Close"].to_numpy(dtype=float)
    current_price = window[-1]
    normalized_window = window / (current_price + 1e-8)
    returns = np.diff(window) / (window[:-1] + 1e-8)
    returns = np.concatenate([[0.0], returns])
    state_parts = [
        normalized_window,
        returns,
        [balance_norm],
        [position_frac],
        [current_drawdown],
        [cooldown_progress],
    ]

    row = df.iloc[-1]
    available_features = [col for col in FEATURE_COLUMNS if col in row]
    state_parts.append(feature_vector(row, current_price, available_features))

    return np.concatenate(state_parts)


def _load_agent(algo, model_path, state_size, allow_legacy=False):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"checkpoint not found: {model_path}")

    if algo == "dqn":
        agent = DQNAgent(state_size, action_size=COMMON["action_size"], window_size=COMMON["window_size"])
        state_dict = load_checkpoint(model_path, agent.device, "dqn", allow_legacy=allow_legacy)
        agent.model.load_state_dict(state_dict)
        agent.model.eval()
        agent.epsilon = 0.0
        return agent

    agent = PPOAgent(state_size, action_size=COMMON["action_size"], window_size=COMMON["window_size"])
    state_dict = load_checkpoint(model_path, agent.device, "ppo", allow_legacy=allow_legacy)
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()
    return agent


def _load_state(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as state_file:
            return json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path, state):
    with open(path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)


def _append_log(timestamp, ticker, algo, price, action, cash, portfolio_value):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if not exists:
            writer.writerow(["timestamp", "ticker", "algo", "price", "action", "cash", "portfolio_value"])
        writer.writerow([timestamp, ticker, algo, price, ACTIONS[action], cash, portfolio_value])


def _get_position(trading_client, ticker):
    try:
        position = trading_client.get_open_position(ticker)
        return float(position.qty), float(position.market_value)
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 404 or "position does not exist" in str(exc).lower():
            return 0.0, 0.0
        raise


def _get_latest_trade_price(data_client, ticker):
    from alpaca.data.requests import StockLatestTradeRequest

    request = StockLatestTradeRequest(symbol_or_symbols=ticker)
    return float(data_client.get_stock_latest_trade(request)[ticker].price)


def _has_open_orders(trading_client, ticker):
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
    return bool(trading_client.get_orders(filter=request))


def _valid_actions(cash, shares_held, current_price, steps_since_rebalance=None):
    if steps_since_rebalance is None:
        steps_since_rebalance = REWARD_CONFIG["min_rebalance_steps"]
    if steps_since_rebalance < REWARD_CONFIG["min_rebalance_steps"]:
        portfolio_value = cash + shares_held * current_price
        allocation = shares_held * current_price / max(portfolio_value, 1e-8)
        nearest = int(np.argmin(np.abs(TARGET_ALLOCATIONS - allocation)))
        return np.array([nearest], dtype=int)
    return np.arange(len(TARGET_ALLOCATIONS), dtype=int)


def _submit_action(
    trading_client,
    ticker,
    action,
    cash,
    shares_held,
    current_price,
    portfolio_value,
    dry_run,
    client_order_id,
    can_rebalance=True,
):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if not can_rebalance:
        return None

    target = float(TARGET_ALLOCATIONS[action])
    current_equity = shares_held * current_price
    current_allocation = current_equity / max(portfolio_value, 1e-8)
    if abs(current_allocation - target) < REWARD_CONFIG["rebalance_threshold"]:
        return None
    desired_equity = target * portfolio_value
    difference = desired_equity - current_equity

    if difference >= current_price:
        budget = min(difference, cash * 0.99)
        quantity = int(budget // current_price)
        side = OrderSide.BUY
    elif difference <= -current_price:
        quantity = min(int(shares_held), max(1, int(round((-difference) / current_price))))
        side = OrderSide.SELL
    else:
        return None

    if quantity < 1:
        return None
    order = MarketOrderRequest(
        symbol=ticker,
        qty=quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )
    return order if dry_run else trading_client.submit_order(order_data=order)


def run_paper_trading(
    algo="dqn",
    ticker="AAPL",
    model_path=None,
    interval_sec=60,
    once=False,
    dry_run=False,
    allow_legacy_checkpoint=False,
):
    trading_client, data_client = get_alpaca_clients()
    account = trading_client.get_account()
    initial_equity = float(account.equity)
    model_path = model_path or f"models/{algo}.pth"
    agent = None
    execution_state = _load_state(STATE_PATH)
    state_key = f"v{PIPELINE_VERSION}:{algo}:{ticker}"

    print(f"Connected to Alpaca paper account {account.id}; using {model_path}")
    while True:
        try:
            clock = trading_client.get_clock()
            if not clock.is_open:
                print(f"Market closed; next open is {clock.next_open}")
                if once:
                    return
                time.sleep(interval_sec)
                continue

            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=420)
            df = load_alpaca_data(
                ticker,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                save=False,
                client=data_client,
            )
            bar_timestamp = df.index[-1].isoformat()
            ticker_state = execution_state.get(state_key, {})
            if ticker_state.get("bar_timestamp") == bar_timestamp:
                print(f"Already processed completed bar {bar_timestamp}")
                if once:
                    return
                time.sleep(interval_sec)
                continue
            if _has_open_orders(trading_client, ticker):
                print(f"Open {ticker} order exists; waiting before making another decision")
                if once:
                    return
                time.sleep(interval_sec)
                continue

            account = trading_client.get_account()
            cash = max(0.0, float(account.cash))
            portfolio_value = float(account.portfolio_value)
            shares_held, market_value = _get_position(trading_client, ticker)
            position_frac = market_value / max(portfolio_value, 1e-8)
            balance_norm = cash / max(initial_equity, 1e-8)
            peak_portfolio_value = max(
                float(ticker_state.get("peak_portfolio_value", portfolio_value)),
                portfolio_value,
            )
            current_drawdown = max(
                0.0,
                (peak_portfolio_value - portfolio_value)
                / max(peak_portfolio_value, 1e-8),
            )
            steps_since_rebalance = int(
                ticker_state.get(
                    "steps_since_rebalance",
                    REWARD_CONFIG["min_rebalance_steps"],
                )
            )
            cooldown_progress = min(
                1.0,
                steps_since_rebalance / max(1, REWARD_CONFIG["min_rebalance_steps"]),
            )
            current_price = _get_latest_trade_price(data_client, ticker)
            observation = build_state_from_df(
                df,
                position_frac=position_frac,
                balance_norm=balance_norm,
                current_drawdown=current_drawdown,
                cooldown_progress=cooldown_progress,
            )

            if agent is None:
                agent = _load_agent(
                    algo,
                    model_path,
                    observation.shape[0],
                    allow_legacy=allow_legacy_checkpoint,
                )
            valid_actions = _valid_actions(
                cash,
                shares_held,
                current_price,
                steps_since_rebalance,
            )
            if algo == "dqn":
                action = agent.act(observation, valid_actions=valid_actions)
            else:
                action = agent.select_action(observation, valid_actions=valid_actions, deterministic=True)

            order_key = f"{algo}:{ticker}:{bar_timestamp}:{action}"
            client_order_id = f"rl-agent-{hashlib.sha256(order_key.encode()).hexdigest()[:24]}"
            result = _submit_action(
                trading_client,
                ticker,
                action,
                cash,
                shares_held,
                current_price,
                portfolio_value,
                dry_run,
                client_order_id,
                can_rebalance=(
                    steps_since_rebalance >= REWARD_CONFIG["min_rebalance_steps"]
                ),
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            mode = "DRY RUN" if dry_run else "PAPER"
            print(f"[{timestamp}] {mode} {ticker} ${current_price:.2f}: {ACTIONS[action]}")
            if result is not None:
                print(f"Order prepared: {result}")

            if not dry_run:
                _append_log(timestamp, ticker, algo, current_price, action, cash, portfolio_value)
                execution_state[state_key] = {
                    "bar_timestamp": bar_timestamp,
                    "peak_portfolio_value": peak_portfolio_value,
                    "steps_since_rebalance": (
                        0
                        if result is not None
                        else min(
                            REWARD_CONFIG["min_rebalance_steps"],
                            steps_since_rebalance + 1,
                        )
                    ),
                }
                os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
                _save_state(STATE_PATH, execution_state)
            if once:
                return
        except KeyboardInterrupt:
            print("Paper trading bot stopped by user.")
            return
        except Exception as exc:
            if once:
                raise
            print(f"Paper trading cycle failed: {exc}")

        time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca daily-model paper trading bot")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Run one eligible decision cycle")
    parser.add_argument("--dry-run", action="store_true", help="Build but do not submit an order")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    run_paper_trading(
        algo=args.algo,
        ticker=args.ticker,
        model_path=args.model_path,
        interval_sec=max(1, args.interval),
        once=args.once,
        dry_run=args.dry_run,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
