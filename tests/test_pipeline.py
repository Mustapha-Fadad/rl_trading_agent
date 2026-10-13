import numpy as np
import pandas as pd
import pytest
import torch

from agents.dqn.agent import DQNAgent
from agents.ppo.agent import PPOAgent
from data import data_loader
from data.alpaca_loader import load_alpaca_data
from scripts.paper_trade import _submit_action, _valid_actions, build_state_from_df
from scripts.walk_forward import train_dqn_fold
from env.trading_env import TradingEnv
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.metrics import buy_and_hold_values, portfolio_statistics, validation_score
from utils.validation import split_validation_windows


def _prices(periods=300):
    index = pd.date_range("2024-01-01", periods=periods, tz="UTC")
    return pd.DataFrame(
        {"Close": [100.0 + i for i in range(periods)], "Volume": [1000 + i for i in range(periods)]},
        index=index,
    )


def test_nondefault_date_range_uses_a_distinct_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_loader, "_download_raw_data", lambda *args, **kwargs: _prices())

    first = data_loader.load_data(
        "TEST", start="2024-01-01", end="2024-03-01", save=True, source="yfinance"
    )
    cache = tmp_path / "data" / "processed" / "TEST_2024-01-01_2024-03-01_yfinance.csv"
    assert cache.exists()
    assert isinstance(first.index, pd.DatetimeIndex)

    monkeypatch.setattr(
        data_loader,
        "_download_raw_data",
        lambda *args, **kwargs: pytest.fail("valid cache should be reused"),
    )
    cached = data_loader.load_data(
        "TEST", start="2024-01-01", end="2024-03-01", save=False, source="yfinance"
    )
    assert isinstance(cached.index, pd.DatetimeIndex)
    assert len(cached) == len(first)


def test_alpaca_requests_fully_adjusted_bars():
    from alpaca.data.enums import Adjustment

    class Bars:
        df = _prices().rename(columns={"Close": "close", "Volume": "volume"})

    class Client:
        request = None

        def get_stock_bars(self, request):
            self.request = request
            return Bars()

    client = Client()
    result = load_alpaca_data(
        "TEST", start="2024-01-01", end="2024-03-01", save=False, client=client
    )
    assert client.request.adjustment == Adjustment.ALL
    assert result.attrs["adjustment"] == "all"
    assert isinstance(result.index, pd.DatetimeIndex)


def test_ppo_bootstraps_nonterminal_rollout():
    agent = PPOAgent(state_size=39, action_size=5, window_size=10)
    rewards = torch.tensor([0.0], device=agent.device)
    values = torch.tensor([1.0], device=agent.device)
    masks = torch.tensor([1.0], device=agent.device)

    advantages = agent._compute_advantages(rewards, values, masks, bootstrap_value=2.0)
    assert advantages.item() == pytest.approx(agent.gamma * 2.0 - 1.0)


def test_ppo_updates_and_clears_a_truncated_rollout():
    from data.features import add_features

    env = TradingEnv(add_features(_prices()), window_size=10)
    agent = PPOAgent(state_size=env.reset().shape[0], action_size=5, window_size=10)
    agent.k_epochs = 1
    state = env.reset()
    for _ in range(3):
        valid = env.get_valid_actions()
        action, logprob, value = agent.act(state, valid)
        next_state, reward, done = env.step(action)
        agent.memory.add(state, action, logprob, reward, value, done, valid)
        state = next_state

    loss = agent.train(next_state=state, next_done=False)
    assert np.isfinite(loss)
    assert len(agent.memory) == 0


def test_walk_forward_dqn_uses_current_agent_api():
    from data.features import add_features

    env = TradingEnv(add_features(_prices(320)), window_size=10, random_start=True)
    agent = train_dqn_fold(env, episodes=1)
    assert len(agent.memory) > 0


def test_dqn_performs_an_optimization_step():
    agent = DQNAgent(state_size=39, action_size=5, window_size=10)
    agent.batch_size = 2
    agent.min_replay_size = 2
    state = np.zeros(39, dtype=float)
    agent.memory.add(state, 0, 0.1, state, False)
    agent.memory.add(state, 4, -0.1, state, True)
    agent.train()
    assert agent.optimization_steps == 1


def test_validation_score_penalizes_drawdown_and_turnover():
    smooth = [100.0, 101.0, 102.0]
    volatile = [100.0, 80.0, 102.0]
    assert portfolio_statistics(smooth)["return"] == pytest.approx(0.02)
    assert validation_score(smooth, 0.5) > validation_score(volatile, 0.5)
    assert validation_score(smooth, 0.5) > validation_score(smooth, 10.0)


def test_benchmark_aware_validation_rewards_risk_adjusted_alpha():
    benchmark = [100.0, 80.0, 110.0]
    defensive = [100.0, 95.0, 110.0]
    passive = [100.0, 80.0, 110.0]
    assert validation_score(defensive, 1.0, benchmark) > validation_score(passive, 1.0, benchmark)


def test_validation_windows_are_distinct_and_have_full_episode_length():
    data = _prices(900)
    train, windows = split_validation_windows(data, 10, 3, 252)
    assert len(windows) == 3
    assert all(len(window) == 262 for window in windows)
    assert len(train) == 134
    assert windows[0].index[0] < windows[1].index[0] < windows[2].index[0]


def test_buy_and_hold_path_includes_initial_transaction_fee():
    values = buy_and_hold_values([100.0, 110.0], initial_value=1000.0, transaction_fee=0.001)
    assert values.tolist() == pytest.approx([1000.0, 1098.9])


def test_paper_state_and_action_masks_match_environment_shape():
    from data.features import add_features

    featured = add_features(_prices())
    state = build_state_from_df(featured, position_frac=0.25, balance_norm=0.75)
    assert state.shape == (39,)
    assert _valid_actions(1000.0, 0.0, 100.0).tolist() == [0, 1, 2, 3, 4]
    assert _valid_actions(0.0, 5.0, 100.0).tolist() == [0, 1, 2, 3, 4]


def test_paper_dry_run_builds_an_idempotent_order_without_submitting():
    class TradingClient:
        def submit_order(self, order_data):
            pytest.fail("dry run must not submit an order")

    order = _submit_action(
        TradingClient(),
        "AAPL",
        action=2,
        cash=1000.0,
        shares_held=0.0,
        current_price=100.0,
        portfolio_value=1000.0,
        dry_run=True,
        client_order_id="rl-agent-test",
    )
    assert order.qty == 5
    assert order.client_order_id == "rl-agent-test"


def test_checkpoint_versioning_rejects_legacy_for_execution(tmp_path):
    versioned = tmp_path / "versioned.pth"
    legacy = tmp_path / "legacy.pth"
    weights = {"weight": torch.tensor([1.0])}

    save_checkpoint(versioned, weights, "dqn", state_size=39, window_size=10)
    assert torch.equal(load_checkpoint(versioned, "cpu", "dqn")["weight"], weights["weight"])
    _, metadata = load_checkpoint(versioned, "cpu", "dqn", return_metadata=True)
    assert metadata["pipeline_version"] == 6

    torch.save(weights, legacy)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_checkpoint(legacy, "cpu", "dqn", allow_legacy=False)
