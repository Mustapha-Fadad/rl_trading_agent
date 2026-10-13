import numpy as np

from env.trading_env import TARGET_ALLOCATIONS


class RuleBasedAgent:
    """Deterministic long-only baselines using the environment allocation grid."""

    def __init__(self, strategy, target_volatility=0.20):
        if strategy not in ("sma200", "trend_vol"):
            raise ValueError(f"unsupported rule-based strategy: {strategy}")
        self.strategy = strategy
        self.target_volatility = float(target_volatility)

    def target_allocation(self, row):
        close = float(row["Close"])
        sma_200 = float(row["SMA_200"])
        if close <= sma_200:
            return 0.0
        if self.strategy == "sma200":
            return 1.0

        volatility = max(float(row["Volatility_20"]), 1e-8)
        return min(1.0, self.target_volatility / volatility)

    def act(self, row, valid_actions):
        target = self.target_allocation(row)
        valid_actions = np.asarray(valid_actions, dtype=int)
        allocations = TARGET_ALLOCATIONS[valid_actions]
        return int(valid_actions[np.argmin(np.abs(allocations - target))])
