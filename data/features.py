# data/features.py

import numpy as np
import pandas as pd


PRICE_RELATIVE_FEATURES = (
    "SMA_10",
    "SMA_30",
    "SMA_60",
    "SMA_200",
    "EMA_10",
    "MACD",
    "MACD_Signal",
)

FEATURE_COLUMNS = (
    "SMA_10",
    "SMA_30",
    "SMA_60",
    "SMA_200",
    "EMA_10",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volume_Norm",
    "Momentum_20",
    "Momentum_60",
    "Momentum_120",
    "Volatility_20",
    "Volatility_60",
    "Price_Drawdown_252",
)


def compute_sma(series, window):
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series, span):
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_rsi(series, period=14):
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def add_features(df):
    """
    Add technical indicators to a DataFrame that has 'Close' and optionally 'Volume' columns.
    Returns a new DataFrame with all features added and NaN rows dropped.
    """
    df = df.copy()

    close = df['Close']

    # Moving averages
    df['SMA_10'] = compute_sma(close, 10)
    df['SMA_30'] = compute_sma(close, 30)
    df['SMA_60'] = compute_sma(close, 60)
    df['SMA_200'] = compute_sma(close, 200)
    df['EMA_10'] = compute_ema(close, 10)

    # RSI
    df['RSI_14'] = compute_rsi(close, 14)

    # MACD
    macd_line, signal_line = compute_macd(close)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line

    # Normalize volume if present
    if 'Volume' in df.columns:
        df['Volume_Norm'] = df['Volume'] / (df['Volume'].rolling(window=20, min_periods=20).mean() + 1e-8)
    else:
        df['Volume_Norm'] = 1.0

    daily_returns = close.pct_change()
    df['Momentum_20'] = close.pct_change(20)
    df['Momentum_60'] = close.pct_change(60)
    df['Momentum_120'] = close.pct_change(120)
    df['Volatility_20'] = daily_returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    df['Volatility_60'] = daily_returns.rolling(60, min_periods=60).std() * np.sqrt(252)
    rolling_peak = close.rolling(252, min_periods=200).max()
    df['Price_Drawdown_252'] = close / (rolling_peak + 1e-8) - 1.0

    # Keep the source timestamp index so splits and reports remain auditable.
    return df.dropna().sort_index()


def feature_vector(row, current_price, columns=FEATURE_COLUMNS):
    """Return consistently scaled scalar features for training and execution."""
    values = []
    for col in columns:
        if col not in row:
            continue
        value = float(row[col])
        if col in PRICE_RELATIVE_FEATURES:
            value /= current_price + 1e-8
        elif col == "RSI_14":
            value /= 100.0
        values.append(value)
    return np.asarray(values, dtype=float)
