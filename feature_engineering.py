"""
Feature Engineering Module
===========================

Computes technical indicators and creates features for volatility prediction modeling.

This module generates 50+ technical features from raw OHLCV data including:
- Price features: Returns, Moving Averages, MACD, Bollinger Bands
- Volume features: Volume ratios, changes, and price-volume interactions
- Volatility features: Historical volatility, rolling statistics, regimes
- Momentum features: RSI, ROC, Momentum indicators

Example:
    from feature_engineering import create_all_features, get_feature_columns
    
    df_features = create_all_features(df)
    feature_cols = get_feature_columns()
    print(f"Created {len(feature_cols)} features")
"""

import numpy as np
import pandas as pd


def compute_rsi(close, n=14):
    """
    Compute Relative Strength Index (RSI).
    
    Parameters:
    -----------
    close : pd.Series
        Closing prices
    n : int
        RSI period
        
    Returns:
    --------
    pd.Series
        RSI values
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(close, n=20, num_std=2):
    """
    Compute Bollinger Bands.
    
    Parameters:
    -----------
    close : pd.Series
        Closing prices
    n : int
        Moving average period
    num_std : float
        Number of standard deviations for bands
        
    Returns:
    --------
    tuple
        (middle_band, upper_band, lower_band, bandwidth)
    """
    middle = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle
    return middle, upper, lower, bandwidth


def add_price_features(df):
    """
    Add price-based features (returns, moving averages, etc.).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added price features
    """
    df = df.copy()
    
    # Returns
    df["log_return"] = np.log(df["close"]).diff()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_7d"] = df["close"].pct_change(7)
    
    # Moving averages
    for window in [5, 10, 20, 30, 50]:
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
        df[f"ma_{window}_ratio"] = df["close"] / df[f"ma_{window}"]
        df[f"ma_{window}_slope"] = df[f"ma_{window}"].diff()
    
    # Exponential moving averages
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger_bands(df["close"], 20, 2)
    df["bb_middle"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    
    return df


def add_volume_features(df):
    """
    Add volume-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added volume features
    """
    df = df.copy()
    
    # Volume rolling statistics
    df["vol_roll_5"] = df["volume"].rolling(5).mean()
    df["vol_roll_10"] = df["volume"].rolling(10).mean()
    df["vol_roll_20"] = df["volume"].rolling(20).mean()
    
    # Volume changes
    df["vol_change_1d"] = df["volume"].pct_change()
    df["vol_change_3d"] = df["volume"].pct_change(3)
    df["vol_change_5d"] = df["volume"].pct_change(5)
    
    # Volume ratio to moving average
    df["vol_ratio_5"] = df["volume"] / df["vol_roll_5"]
    df["vol_ratio_20"] = df["volume"] / df["vol_roll_20"]
    
    # Price-volume interaction
    df["price_volume"] = df["close"] * df["volume"]
    df["pv_ratio"] = df["price_volume"] / df["price_volume"].rolling(20).mean()
    
    return df


def add_volatility_features(df):
    """
    Add volatility-based features and target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added volatility features and target
    """
    df = df.copy()
    
    # Intraday volatility (High-Low range)
    df["hl_range"] = (df["high"] - df["low"]) / df["open"]
    df["oc_range"] = abs(df["close"] - df["open"]) / df["open"]
    
    # Realized volatility (our target for day t)
    df["volatility_t"] = df["hl_range"]
    
    # Rolling volatility statistics
    df["volatility_roll_5"] = df["volatility_t"].rolling(5).mean()
    df["volatility_roll_10"] = df["volatility_t"].rolling(10).mean()
    df["volatility_roll_20"] = df["volatility_t"].rolling(20).mean()
    df["volatility_std_20"] = df["volatility_t"].rolling(20).std()
    
    # Return volatility (standard deviation of returns)
    df["return_vol_5"] = df["ret_1d"].rolling(5).std()
    df["return_vol_10"] = df["ret_1d"].rolling(10).std()
    df["return_vol_20"] = df["ret_1d"].rolling(20).std()
    
    # Target variable: next day's volatility
    df["volatility_t_plus_1"] = df["volatility_t"].shift(-1)
    
    # Volatility regime (categorical feature)
    vol_33 = df["volatility_t"].quantile(0.33)
    vol_67 = df["volatility_t"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility_t"],
        bins=[-np.inf, vol_33, vol_67, np.inf],
        labels=["low", "medium", "high"]
    )
    
    return df


def add_momentum_features(df):
    """
    Add momentum and technical indicator features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added momentum features
    """
    df = df.copy()
    
    # RSI (Relative Strength Index)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_7"] = compute_rsi(df["close"], 7)
    df["rsi_21"] = compute_rsi(df["close"], 21)
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = ((df["close"] - df["close"].shift(period)) / 
                                df["close"].shift(period)) * 100
    
    # Momentum
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["momentum_20"] = df["close"] - df["close"].shift(20)
    
    return df


def create_all_features(df):
    """
    Create all features for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clean OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all features
    """
    print("Creating features...")
    
    df = add_price_features(df)
    print("✓ Price features added")
    
    df = add_volume_features(df)
    print("✓ Volume features added")
    
    df = add_volatility_features(df)
    print("✓ Volatility features added")
    
    df = add_momentum_features(df)
    print("✓ Momentum features added")
    
    # Drop rows with NaN values (from rolling calculations)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print(f"✓ Dropped {rows_before - rows_after} rows with NaN values")
    
    print(f"✓ Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def get_feature_columns():
    """
    Return list of feature columns to use for modeling.
    
    Returns:
    --------
    list
        List of feature column names
    """
    feature_cols = [
        # Returns
        "log_return", "ret_1d", "ret_3d", "ret_5d", "ret_7d",
        
        # Moving averages and ratios
        "ma_5", "ma_10", "ma_20", "ma_30", "ma_50",
        "ma_5_ratio", "ma_10_ratio", "ma_20_ratio",
        "ma_5_slope", "ma_10_slope", "ma_20_slope",
        
        # EMA and MACD
        "ema_12", "ema_26", "macd", "macd_signal", "macd_diff",
        
        # Bollinger Bands
        "bb_width", "bb_position",
        
        # Volume
        "vol_roll_5", "vol_roll_20",
        "vol_change_1d", "vol_change_3d",
        "vol_ratio_5", "vol_ratio_20",
        "pv_ratio",
        
        # Volatility (historical)
        "volatility_t", "volatility_roll_5", "volatility_roll_10",
        "volatility_roll_20", "volatility_std_20",
        "return_vol_5", "return_vol_10", "return_vol_20",
        "hl_range", "oc_range",
        
        # Momentum
        "rsi_14", "rsi_7", "rsi_21",
        "roc_5", "roc_10", "roc_20",
        "momentum_10", "momentum_20",
    ]
    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from data_preparation import load_or_fetch_data
    
    df = load_or_fetch_data("BTCUSDT", "2019-01-01")
    df_features = create_all_features(df)
    
    print("\nFeature columns:")
    features = get_feature_columns()
    print(f"Total features: {len(features)}")
    print(features[:10], "...")
    
    print("\nSample data:")
    print(df_features[["date", "close", "volatility_t", "volatility_t_plus_1"]].tail())

