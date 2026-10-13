import numpy as np


def portfolio_statistics(portfolio_values):
    values = np.asarray(portfolio_values, dtype=float)
    if len(values) < 2 or values[0] <= 0:
        raise ValueError("at least two positive portfolio values are required")

    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / np.maximum(peak, 1e-8)
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0

    return {
        "return": float(values[-1] / values[0] - 1.0),
        "max_drawdown": float(np.max(drawdowns)),
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe": float(np.mean(returns) / volatility * np.sqrt(252)) if volatility > 1e-12 else 0.0,
        "sortino": float(np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 1e-12 else 0.0,
    }


def buy_and_hold_values(prices, initial_value=10_000.0, transaction_fee=0.001):
    """Build a passive benchmark path aligned with an agent portfolio path."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 1 or prices[0] <= 0:
        raise ValueError("at least one positive benchmark price is required")
    invested = initial_value * (1.0 - transaction_fee)
    values = invested * prices / prices[0]
    values[0] = initial_value
    return values


def validation_score(portfolio_values, total_turnover, benchmark_values=None):
    """Score return, risk, and turnover, optionally relative to buy-and-hold."""
    stats = portfolio_statistics(portfolio_values)
    if benchmark_values is not None:
        benchmark = portfolio_statistics(benchmark_values)
        return (
            stats["return"]
            - benchmark["return"]
            + 0.5 * (benchmark["max_drawdown"] - stats["max_drawdown"])
            - 0.001 * total_turnover
        )
    return stats["return"] - 0.5 * stats["max_drawdown"] - 0.001 * total_turnover
