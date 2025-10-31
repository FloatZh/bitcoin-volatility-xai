"""
Data Preparation Module
========================

Handles cryptocurrency data fetching from Binance API and basic cleaning operations.

This module provides functions to:
- Fetch historical OHLCV data from Binance public API
- Clean and validate the data
- Cache data locally to avoid repeated API calls
- Detect and report outliers

Example:
    from data_preparation import load_or_fetch_data
    
    df = load_or_fetch_data("BTCUSDT", "2019-01-01")
    print(df.head())
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path


def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_str="2019-01-01", 
                         end_str=None, limit=1000):
    """
    Fetch historical klines (candles) from Binance public API.
    
    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    interval : str
        Kline interval (e.g., '1d' for daily)
    start_str : str
        Start date in 'YYYY-MM-DD' format
    end_str : str, optional
        End date in 'YYYY-MM-DD' format
    limit : int
        Number of records per API call
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    def to_ms(s):
        if s is None:
            return None
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    base = "https://api.binance.com/api/v3/klines"
    start_ms = to_ms(start_str)
    end_ms = to_ms(end_str) if end_str else None

    all_rows = []
    print(f"Fetching data from {start_str} to {end_str or 'present'}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_ms:
            params["startTime"] = start_ms
        if end_ms:
            params["endTime"] = end_ms

        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()

        if not rows:
            break

        all_rows.extend(rows)

        # next page: start after last close time
        last_close = rows[-1][6]
        if end_ms and last_close >= end_ms:
            break
        start_ms = last_close + 1

        time.sleep(0.2)  # rate limit protection

        if len(all_rows) > 100000:
            break

    # Format data
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    
    # Cast numeric columns
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert timestamps to dates
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("UTC").dt.date
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    
    print(f"✓ Fetched {len(df)} daily candles")
    return df


def clean_data(df, outlier_threshold=5.0):
    """
    Clean and prepare raw data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLCV data
    outlier_threshold : float
        Z-score threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    df = df.copy()
    
    # Keep only necessary columns
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠ Found {missing_before} missing values, filling...")
        df = df.ffill().bfill()  # Modern pandas syntax
    
    # Detect outliers using Z-score on returns
    returns = df["close"].pct_change()
    z_scores = np.abs((returns - returns.mean()) / returns.std())
    outliers = z_scores > outlier_threshold
    n_outliers = outliers.sum()
    
    if n_outliers > 0:
        print(f"⚠ Detected {n_outliers} potential outliers (Z-score > {outlier_threshold})")
        print(f"  Dates: {df.loc[outliers, 'date'].tolist()[:5]}{'...' if n_outliers > 5 else ''}")
        # Note: Not removing outliers automatically, just flagging them
    
    # Ensure data is sorted by date
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"✓ Data cleaned: {len(df)} rows")
    return df


def load_or_fetch_data(symbol="BTCUSDT", start_date="2019-01-01", 
                       end_date=None, force_refresh=False):
    """
    Load data from CSV if exists, otherwise fetch from API.
    
    Parameters:
    -----------
    symbol : str
        Trading pair symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    force_refresh : bool
        If True, fetch fresh data even if CSV exists
        
    Returns:
    --------
    pd.DataFrame
        Clean OHLCV data
    """
    out_csv = Path(f"{symbol.lower()}_daily.csv")
    
    if out_csv.exists() and not force_refresh:
        print(f"Loading existing data from {out_csv}...")
        df = pd.read_csv(out_csv, parse_dates=["date"])
        print(f"✓ Loaded {len(df)} rows from CSV")
    else:
        print(f"Downloading {symbol} data from Binance API...")
        df = fetch_binance_klines(symbol, "1d", start_str=start_date, end_str=end_date)
        df.to_csv(out_csv, index=False)
        print(f"✓ Saved to {out_csv}")
    
    # Clean the data
    df = clean_data(df)
    
    return df


if __name__ == "__main__":
    # Test data fetching
    df = load_or_fetch_data("BTCUSDT", "2019-01-01")
    print("\nData Sample:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nPrice Statistics:")
    print(df[["open", "high", "low", "close", "volume"]].describe())

