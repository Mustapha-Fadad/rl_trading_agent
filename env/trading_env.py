import numpy as np

from data.features import FEATURE_COLUMNS, feature_vector


TARGET_ALLOCATIONS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)


class TradingEnv:
    """Long-only daily trading environment using target portfolio allocations."""

    def __init__(
        self,
        data,
        window_size=10,
        initial_balance=10_000,
        random_start=False,
        episode_length=None,
        reward_scale=10.0,
        turnover_penalty=0.001,
        drawdown_penalty=0.002,
        rebalance_penalty=0.0005,
        transaction_fee=0.001,
        rebalance_threshold=0.02,
        min_rebalance_steps=20,
        underexposure_penalty=0.0,
    ):
        self.data = data
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.random_start = random_start
        self.episode_length = episode_length
        self.reward_scale = reward_scale
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty
        self.rebalance_penalty = rebalance_penalty
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold
        self.min_rebalance_steps = min_rebalance_steps
        self.underexposure_penalty = float(underexposure_penalty)
        if self.underexposure_penalty < 0.0:
            raise ValueError("underexposure_penalty must be non-negative")

        self.target_allocations = TARGET_ALLOCATIONS.copy()
        self.action_space = np.arange(len(self.target_allocations))
        self.n_actions = len(self.target_allocations)
        self.feature_cols = [col for col in FEATURE_COLUMNS if col in data.columns]

        self.n_steps = len(data)
        if self.n_steps <= self.window_size:
            raise ValueError("data must contain more rows than window_size")
        if not np.isfinite(data["Close"].to_numpy(dtype=float)).all():
            raise ValueError("Close prices must be finite")
        if (data["Close"] <= 0).any():
            raise ValueError("Close prices must be positive")
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be positive")

        self.trades = []
        self.reset()

    def reset(self):
        earliest = self.window_size - 1
        if self.random_start and self.episode_length is not None:
            latest = max(earliest, self.n_steps - self.episode_length - 1)
            self.current_step = np.random.randint(earliest, latest + 1)
        elif self.random_start:
            latest = max(earliest, self.n_steps - 51)
            self.current_step = np.random.randint(earliest, latest + 1)
        else:
            self.current_step = earliest

        if self.episode_length is None:
            self.episode_end = self.n_steps - 1
        else:
            self.episode_end = min(self.n_steps - 1, self.current_step + self.episode_length)

        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.position_frac = 0.0
        self.prev_price = float(self.data.iloc[self.current_step]["Close"])
        self.last_trade_notional = 0.0
        self.last_turnover = 0.0
        self.total_turnover = 0.0
        self.peak_portfolio_value = self.initial_balance
        self.current_drawdown = 0.0
        self.steps_since_rebalance = self.min_rebalance_steps
        self.trades = []
        return self._get_state()

    def portfolio_value(self, price=None):
        if price is None:
            price = self.prev_price
        return self.balance + self.shares_held * price

    def _compute_position_frac(self, price):
        total = self.portfolio_value(price)
        if total < 1e-8:
            return 0.0
        return (self.shares_held * price) / total

    def _get_state(self):
        window = self.data.iloc[
            self.current_step - self.window_size + 1 : self.current_step + 1
        ]["Close"].to_numpy(dtype=float)
        current_price = window[-1]
        normalized_window = window / (current_price + 1e-8)
        returns = np.concatenate(
            [[0.0], np.diff(window) / (window[:-1] + 1e-8)]
        )
        state_parts = [
            normalized_window,
            returns,
            [self.balance / self.initial_balance],
            [self.position_frac],
            [self.current_drawdown],
            [min(1.0, self.steps_since_rebalance / max(1, self.min_rebalance_steps))],
        ]

        row = self.data.iloc[self.current_step]
        state_parts.append(feature_vector(row, current_price, self.feature_cols))
        return np.concatenate(state_parts)

    def get_valid_actions(self):
        """Hold the nearest allocation until the minimum holding period elapses."""
        if self.steps_since_rebalance < self.min_rebalance_steps:
            nearest = int(np.argmin(np.abs(self.target_allocations - self.position_frac)))
            return np.array([nearest], dtype=int)
        return self.action_space.copy()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"unknown action: {action}")

        price = float(self.data.iloc[self.current_step]["Close"])
        previous_value = self.portfolio_value(price)
        target = float(self.target_allocations[action])
        if self.steps_since_rebalance >= self.min_rebalance_steps:
            trade_notional = self._rebalance(price, target)
        else:
            trade_notional = 0.0
        turnover = abs(trade_notional) / max(previous_value, 1e-8)

        self.last_trade_notional = trade_notional
        self.last_turnover = turnover
        self.total_turnover += turnover
        if turnover > 0.0:
            self.steps_since_rebalance = 0
        else:
            self.steps_since_rebalance += 1
        self.trades.append(
            {
                "step": self.current_step,
                "action": int(action),
                "target_allocation": target,
                "price": price,
                "trade_notional": trade_notional,
                "turnover": turnover,
            }
        )

        self.current_step += 1
        done = self.current_step >= self.episode_end
        new_price = float(self.data.iloc[self.current_step]["Close"])
        current_value = self.portfolio_value(new_price)
        self.position_frac = self._compute_position_frac(new_price)
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        self.current_drawdown = max(
            0.0,
            (self.peak_portfolio_value - current_value)
            / max(self.peak_portfolio_value, 1e-8),
        )

        log_return = np.log(max(current_value, 1e-8) / max(previous_value, 1e-8))
        reward = self.reward_scale * (
            log_return
            - self.turnover_penalty * turnover
            - self.rebalance_penalty * float(turnover > 0.0)
            - self.drawdown_penalty * self.current_drawdown
            - self.underexposure_penalty * (1.0 - self.position_frac)
        )

        self.prev_price = new_price
        return self._get_state(), float(reward), done

    def _rebalance(self, price, target):
        """Rebalance to target exposure and return signed traded notional."""
        portfolio = self.portfolio_value(price)
        equity = self.shares_held * price
        current_allocation = equity / max(portfolio, 1e-8)
        if abs(current_allocation - target) < self.rebalance_threshold:
            return 0.0
        desired_equity = target * portfolio
        difference = desired_equity - equity

        if abs(difference) < 1.0:
            return 0.0

        if difference > 0:
            denominator = 1.0 - self.transaction_fee + target * self.transaction_fee
            cash_outlay = min(self.balance, difference / denominator)
            if cash_outlay < 1.0:
                return 0.0
            net_investment = cash_outlay * (1.0 - self.transaction_fee)
            self.shares_held += net_investment / price
            self.balance -= cash_outlay
            return cash_outlay

        denominator = 1.0 - target * self.transaction_fee
        gross_sale = min(equity, (-difference) / denominator)
        if gross_sale < 1.0:
            return 0.0
        self.shares_held -= gross_sale / price
        self.balance += gross_sale * (1.0 - self.transaction_fee)
        return -gross_sale
