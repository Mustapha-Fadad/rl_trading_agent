# data/alpaca_loader.py

import os
import pandas as pd
from dotenv import load_dotenv

from data.features import add_features

load_dotenv()


def load_alpaca_data(
    ticker="AAPL",
    start="2020-01-01",
    end="2023-01-01",
    save=True,
    fallback_to_yfinance=False,
    client=None,
):
    """
    Fetch split- and dividend-adjusted daily bars using alpaca-py.

    Fallback is opt-in and bypasses this loader, preventing recursive retries.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if client is None and (not api_key or not secret_key):
        if fallback_to_yfinance:
            from data.data_loader import load_data

            return load_data(ticker, start, end, save=save, source="yfinance")
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        print(f"[Alpaca Loader] Fetching {ticker} bars from Alpaca Market Data API...")
        if client is None:
            client = StockHistoricalDataClient(api_key, secret_key)

        start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
        end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()

        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            adjustment=Adjustment.ALL,
        )

        bars = client.get_stock_bars(request_params)
        df_raw = bars.df

        if df_raw.empty:
            raise ValueError(f"No data returned from Alpaca for ticker {ticker}")

        # If multi-index (symbol, timestamp), reset level
        if isinstance(df_raw.index, pd.MultiIndex):
            df_raw = df_raw.xs(ticker, level=0)

        df = pd.DataFrame(index=df_raw.index)
        df['Close'] = pd.to_numeric(df_raw['close'], errors='coerce')
        if 'volume' in df_raw.columns:
            df['Volume'] = pd.to_numeric(df_raw['volume'], errors='coerce')

        df = df.dropna(subset=['Close']).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Add technical indicators
        df = add_features(df)
        df.attrs['price_mode'] = 'raw'
        df.attrs['source'] = 'alpaca'
        df.attrs['adjustment'] = 'all'
        df.attrs['start'] = start
        df.attrs['end'] = end

        if save:
            os.makedirs("data/processed", exist_ok=True)
            safe_ticker = ticker.replace("/", "-")
            file_path = f"data/processed/{safe_ticker}_{start}_{end}_alpaca.csv"
            df.to_csv(file_path)
            print(f"[Alpaca Loader] Saved Alpaca data to {file_path}")

        return df

    except Exception as exc:
        if fallback_to_yfinance:
            from data.data_loader import load_data

            print(f"[Alpaca Loader] Alpaca failed ({exc}); using yfinance.")
            return load_data(ticker, start, end, save=save, source="yfinance")
        raise RuntimeError(f"Failed to fetch {ticker} from Alpaca") from exc
