import numpy as np
import pandas as pd
import pytest

from env.trading_env import TARGET_ALLOCATIONS, TradingEnv


def _make_data(n=120, seed=0, constant=False):
    rng = np.random.default_rng(seed)
    close = np.full(n, 100.0) if constant else 100.0 + rng.standard_normal(n).cumsum()
    close = np.clip(close, 1.0, None)
    return pd.DataFrame(
        {
            "Close": close,
            "SMA_10": close * 0.99,
            "SMA_30": close * 0.98,
            "SMA_60": close * 0.97,
            "SMA_200": close * 0.95,
            "EMA_10": close * 0.995,
            "RSI_14": np.clip(50 + rng.standard_normal(n) * 10, 0, 100),
            "MACD": rng.standard_normal(n) * 0.5,
            "MACD_Signal": rng.standard_normal(n) * 0.5,
            "Volume_Norm": np.abs(rng.standard_normal(n)),
            "Momentum_20": rng.standard_normal(n) * 0.05,
            "Momentum_60": rng.standard_normal(n) * 0.1,
            "Momentum_120": rng.standard_normal(n) * 0.15,
            "Volatility_20": np.abs(rng.standard_normal(n) * 0.2),
            "Volatility_60": np.abs(rng.standard_normal(n) * 0.2),
            "Price_Drawdown_252": -np.abs(rng.standard_normal(n) * 0.1),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )


@pytest.fixture
def env():
    return TradingEnv(_make_data(), window_size=10)


def test_reset_state_shape(env):
    state = env.reset()
    assert state.ndim == 1
    assert len(state) == 39


def test_all_target_allocations_are_always_valid(env):
    assert env.get_valid_actions().tolist() == [0, 1, 2, 3, 4]
    env.step(4)
    assert env.get_valid_actions().tolist() == [4]


@pytest.mark.parametrize("action,target", enumerate(TARGET_ALLOCATIONS))
def test_action_rebalances_to_target(action, target):
    env = TradingEnv(_make_data(constant=True), window_size=10)
    env.reset()
    env.step(action)
    assert env.position_frac == pytest.approx(target, abs=2e-4)


def test_rebalance_from_full_to_half():
    env = TradingEnv(_make_data(constant=True), window_size=10)
    env.reset()
    env.step(4)
    env.steps_since_rebalance = env.min_rebalance_steps
    env.step(2)
    assert env.position_frac == pytest.approx(0.5, abs=2e-4)


def test_same_target_on_constant_prices_is_a_no_op():
    env = TradingEnv(_make_data(constant=True), window_size=10)
    env.reset()
    env.step(2)
    env.step(2)
    assert env.last_trade_notional == pytest.approx(0.0, abs=1.0)


def test_minimum_holding_period_blocks_early_reallocation():
    env = TradingEnv(_make_data(constant=True), window_size=10, min_rebalance_steps=3)
    env.step(4)
    env.step(0)
    assert env.position_frac == pytest.approx(1.0, abs=2e-4)
    env.step(4)
    env.step(4)
    assert env.get_valid_actions().tolist() == [0, 1, 2, 3, 4]
    env.step(0)
    assert env.position_frac == pytest.approx(0.0, abs=2e-4)


def test_rebalance_band_ignores_small_allocation_drift():
    env = TradingEnv(_make_data(constant=True), window_size=10, rebalance_threshold=0.02)
    env.reset()
    env.step(2)
    env.shares_held *= 1.02
    env.step(2)
    assert env.last_trade_notional == 0.0


def test_transaction_fee_reduces_portfolio_value():
    env = TradingEnv(_make_data(constant=True), window_size=10)
    initial_value = env.portfolio_value()
    env.step(4)
    assert env.portfolio_value() < initial_value


def test_reward_covers_exactly_one_future_bar():
    data = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    env = TradingEnv(
        data,
        window_size=3,
        initial_balance=102.0,
        reward_scale=1.0,
        turnover_penalty=0.0,
        transaction_fee=0.0,
    )
    env.reset()
    env.balance = 0.0
    env.shares_held = 1.0
    env.position_frac = 1.0
    _, reward, _ = env.step(4)
    assert reward == pytest.approx(np.log(103.0 / 102.0))


def test_reward_penalizes_current_drawdown():
    data = pd.DataFrame({"Close": [100.0, 100.0, 100.0, 90.0]})
    env = TradingEnv(
        data,
        window_size=3,
        initial_balance=100.0,
        reward_scale=1.0,
        turnover_penalty=0.0,
        drawdown_penalty=0.1,
        transaction_fee=0.0,
    )
    env.reset()
    env.balance = 0.0
    env.shares_held = 1.0
    env.position_frac = 1.0
    _, reward, _ = env.step(4)
    assert env.current_drawdown == pytest.approx(0.1)
    assert reward == pytest.approx(np.log(0.9) - 0.01)


def test_reward_penalizes_each_meaningful_rebalance():
    data = pd.DataFrame({"Close": [100.0, 100.0, 100.0, 100.0]})
    env = TradingEnv(
        data,
        window_size=3,
        reward_scale=1.0,
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        rebalance_penalty=0.01,
        transaction_fee=0.0,
    )
    _, reward, _ = env.step(4)
    assert reward == pytest.approx(-0.01)


def test_underexposure_penalty_anchors_reward_to_full_investment():
    data = pd.DataFrame({"Close": [100.0, 100.0, 100.0, 100.0]})
    kwargs = {
        "window_size": 3,
        "reward_scale": 1.0,
        "turnover_penalty": 0.0,
        "drawdown_penalty": 0.0,
        "rebalance_penalty": 0.0,
        "transaction_fee": 0.0,
        "underexposure_penalty": 0.001,
    }
    cash_env = TradingEnv(data, **kwargs)
    full_env = TradingEnv(data, **kwargs)
    _, cash_reward, _ = cash_env.step(0)
    _, full_reward, _ = full_env.step(4)
    assert cash_reward == pytest.approx(-0.001)
    assert full_reward == pytest.approx(0.0)


def test_fixed_episode_length():
    env = TradingEnv(
        _make_data(n=300),
        window_size=10,
        random_start=True,
        episode_length=50,
    )
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done = env.step(0)
        steps += 1
    assert steps == 50


def test_turnover_is_recorded(env):
    env.reset()
    env.step(3)
    assert env.last_turnover > 0
    assert env.total_turnover == pytest.approx(env.last_turnover)


def test_invalid_action_is_rejected(env):
    with pytest.raises(ValueError, match="unknown action"):
        env.step(5)
