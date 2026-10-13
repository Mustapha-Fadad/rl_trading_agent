# data/data_loader.py

import os
import pandas as pd
import yfinance as yf

from data.features import FEATURE_COLUMNS, add_features

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2023-01-01"

def _looks_normalized(series):
    """Detect if Close price values are min-max scaled to [0, 1]."""
    vals = series.dropna()
    if len(vals) == 0:
        return False
    return float(vals.min()) >= 0.0 and float(vals.max()) <= 1.05 and float(vals.mean()) < 1.0


def _download_raw_data(ticker="AAPL", start=DEFAULT_START, end=DEFAULT_END):
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df['Close'] = pd.to_numeric(raw['Close'], errors='coerce')

    if 'Volume' in raw.columns:
        df['Volume'] = pd.to_numeric(raw['Volume'], errors='coerce')
    elif 'Volume' in raw.index.names:
        df['Volume'] = 1.0

    df = df.dropna(subset=['Close'])
    return df


def _cache_path(ticker, start, end):
    safe_ticker = ticker.replace("/", "-")
    if start == DEFAULT_START and end == DEFAULT_END:
        return os.path.join("data", "processed", f"{safe_ticker}.csv")
    return os.path.join("data", "processed", f"{safe_ticker}_{start}_{end}_yfinance.csv")


def _read_cache(file_path):
    df = pd.read_csv(file_path, index_col=0)
    if not pd.api.types.is_numeric_dtype(df.index.dtype):
        parsed = pd.to_datetime(df.index, errors="coerce", utc=True)
        if not parsed.isna().any():
            df.index = parsed
            df.index.name = "Date"
    return df


def load_data(
    ticker="AAPL",
    start=DEFAULT_START,
    end=DEFAULT_END,
    save=True,
    source="yfinance",
    refresh=False,
):
    """
    Load adjusted daily prices with technical indicators.

    The source is explicit so adding Alpaca credentials cannot silently change a
    training or evaluation dataset. Yahoo Finance is the deterministic default.
    """
    source = source.lower()
    if source == "alpaca":
        from data.alpaca_loader import load_alpaca_data

        return load_alpaca_data(ticker, start=start, end=end, save=save)
    if source != "yfinance":
        raise ValueError("source must be 'yfinance' or 'alpaca'")

    file_path = _cache_path(ticker, start, end)

    if os.path.exists(file_path) and not refresh:
        print(f"Loading {ticker} data from {file_path}...")
        df = _read_cache(file_path)

        has_features = all(col in df.columns for col in FEATURE_COLUMNS)

        if has_features and not _looks_normalized(df['Close']):
            df.attrs['price_mode'] = 'raw'
            df.attrs['source'] = 'yfinance'
            df.attrs['start'] = start
            df.attrs['end'] = end
            return df

        if not _looks_normalized(df['Close']):
            print("Local feature cache is outdated. Recomputing features...")
            raw_columns = [col for col in ("Close", "Volume") if col in df.columns]
            df = add_features(df[raw_columns])
            df.attrs['price_mode'] = 'raw'
            df.attrs['source'] = 'yfinance'
            df.attrs['start'] = start
            df.attrs['end'] = end
            if save:
                df.to_csv(file_path)
            return df

        print("Local file is outdated. Re-downloading...")

    print(f"Downloading {ticker} data via yfinance...")
    df = _download_raw_data(ticker, start, end)
    df = add_features(df)
    df.attrs['price_mode'] = 'raw'
    df.attrs['source'] = 'yfinance'
    df.attrs['start'] = start
    df.attrs['end'] = end

    if save:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(file_path)
        print(f"Saved to {file_path}")

    return df


def train_test_split(df, train_ratio=0.8):
    """
    Split data chronologically into train and test sets.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_df.attrs = df.attrs.copy()
    test_df.attrs = df.attrs.copy()

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df
