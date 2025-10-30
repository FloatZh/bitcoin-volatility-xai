# start.py
# Minimal end-to-end pipeline: data -> features -> model -> evaluation (+ optional SHAP)
# Works out of the box with public Binance API (no key).

import math
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to use XGBoost; if unavailable, fall back to RandomForest
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False

# ---------------------------
# 1) Data download (Binance)
# ---------------------------
def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_str="2019-01-01", end_str=None, limit=1000):
    """
    Fetch historical klines (candles) from Binance public API.
    Returns a pandas DataFrame with columns: open_time, open, high, low, close, volume, close_time, ...
    """
    def to_ms(s):
        if s is None:
            return None
        # interpret YYYY-MM-DD as UTC midnight
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    base = "https://api.binance.com/api/v3/klines"
    start_ms = to_ms(start_str)
    end_ms = to_ms(end_str) if end_str else None

    all_rows = []
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
        # stop if we've reached or exceeded end bound or no movement
        if end_ms and last_close >= end_ms:
            break
        # increment start to one ms after last close
        start_ms = last_close + 1

        # polite sleep to avoid rate limits
        time.sleep(0.2)

        # Safety break if too many rows
        if len(all_rows) > 100000:
            break

    # format
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    # cast numeric
    for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time to datetime
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("UTC").dt.date
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df = df.reset_index(drop=True)
    return df

# ---------------------------
# 2) Feature engineering
# ---------------------------
def compute_rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing (EMA-ish) is common, but simple rolling mean is ok to start
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"]).diff()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)

    # Moving averages and slopes
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"ma_{w}_slope"] = df[f"ma_{w}"].diff()

    # MACD-ish signals (fast EMA - slow EMA); use simple EMA starter
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # Volume features
    df["vol_roll_5"] = df["volume"].rolling(5).mean()
    df["vol_change_3d"] = df["volume"].pct_change(3)

    # Realized volatility proxy for day t (used to create t+1 target)
    # simple: (High - Low) / Open
    df["volatility_t"] = (df["high"] - df["low"]) / df["open"]

    # Target = next day volatility
    df["volatility_t_plus_1"] = df["volatility_t"].shift(-1)

    # Drop early NA rows
    df = df.dropna().reset_index(drop=True)
    return df

# ---------------------------
# 3) Train / evaluate
# ---------------------------
def time_split(df, test_size=0.2):
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]

def evaluate_regression(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    out_csv = Path("btc_daily.csv")

    if out_csv.exists():
        df = pd.read_csv(out_csv, parse_dates=["date"])
    else:
        print("Downloading BTC/USDT daily candles from Binance...")
        df = fetch_binance_klines("BTCUSDT", "1d", start_str="2019-01-01")
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} rows to {out_csv}")

    # Keep only what's needed
    df = df[["date","open","high","low","close","volume"]].copy()

    # Add features & target
    df_feat = add_features(df)

    feature_cols = [
        "log_return","ret_1d","ret_5d",
        "ma_5","ma_10","ma_20",
        "ma_5_slope","ma_10_slope","ma_20_slope",
        "ema_12","ema_26","macd",
        "rsi_14",
        "vol_roll_5","vol_change_3d",
        "volatility_t"
    ]

    X = df_feat[feature_cols]
    y = df_feat["volatility_t_plus_1"]

    # Time-ordered split
    train_df, test_df = time_split(df_feat, test_size=0.2)
    X_train, y_train = train_df[feature_cols], train_df["volatility_t_plus_1"]
    X_test, y_test = test_df[feature_cols], test_df["volatility_t_plus_1"]

    # Baseline: persistence (predict next-day vol = today’s vol)
    baseline_pred = test_df["volatility_t"].values
    print("\n=== Baseline (Persistence) ===")
    evaluate_regression(y_test.values, baseline_pred, label="Baseline")

    # Model
    print("\n=== Training Model ===")
    if USE_XGB:
        model = XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== Test Performance ===")
    evaluate_regression(y_test.values, pred, label="XGBoost" if USE_XGB else "RandomForest")

    # Optional: quick feature importance
    try:
        import matplotlib.pyplot as plt
        if USE_XGB and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        elif hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        else:
            imp = None

        if imp is not None:
            print("\nTop features:")
            print(imp.head(10))
            imp.head(15).plot(kind="barh", title="Top Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"(Skipping importance plot: {e})")

    # Optional: SHAP (works best with tree models)
    try:
        import shap
        print("\n=== SHAP Summary (optional) ===")
        # Use a small background to keep it fast
        background = X_train.tail(800)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background)
        shap.summary_plot(shap_values, background, show=True)
    except Exception as e:
        print(f"(Skipping SHAP: {e})")

if __name__ == "__main__":
    main()
