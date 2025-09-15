import pandas as pd
import numpy as np


def clean_crypto_df(df):
    # Ensure 'date' is datetime and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    symbols = df["symbol"].unique()
    cleaned_list = []

    for symbol in symbols:
        sub_df = df[df["symbol"] == symbol].copy()

        # Full daily date range for this symbol
        full_range = pd.DataFrame(
            {
                "date": pd.date_range(
                    sub_df["date"].min(), sub_df["date"].max(), freq="D"
                )
            }
        )
        full_range["symbol"] = symbol

        # Merge to show missing dates as NaN
        merged = pd.merge(full_range, sub_df, on=["symbol", "date"], how="left")

        # Gap detection
        day_diff = merged["date"].diff().dt.days

        # Forward fill for 1-day gaps (only numeric columns)
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        ffill_mask = day_diff.eq(1)
        merged.loc[ffill_mask, numeric_cols] = merged.loc[ffill_mask, numeric_cols].ffill()  # type: ignore

        # Interpolate for longer gaps (>1 day) only on numeric columns
        merged[numeric_cols] = (
            merged[numeric_cols]
            .infer_objects(copy=False)  # Fixes future warning # type: ignore
            .interpolate(method="linear", limit_direction="forward", axis=0)
        )

        cleaned_list.append(merged)

    cleaned_df = pd.concat(cleaned_list, ignore_index=True)
    return cleaned_df


def addReturns(df):
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df["return_target"] = df["return"].shift(1)  # 1-day lagged target
    df = df.dropna().reset_index(drop=True)  # remove NaNs and reset index
    return df


def add_ohlc_features(df, window=14):
    # Simple returns
    df["return"] = df["close"].pct_change()

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Momentum
    df["momentum"] = df["close"] / df["close"].shift(window) - 1

    # Reversal (short-term reversal proxy = negative of momentum)
    df["reversal"] = -df["momentum"]

    # Moving Average
    df["ma"] = df["close"].rolling(window).mean()

    # Volatility (std dev of returns)
    df["volatility"] = df["return"].rolling(window).std()

    # Downside volatility (only negative returns)
    downside = df["return"].where(df["return"] < 0, 0)
    df["downside_vol"] = downside.rolling(window).std()

    # Max Drawdown
    running_max = df["close"].cummax()
    df["drawdown"] = (running_max - df["close"]) / running_max
    df["max_drawdown"] = df["drawdown"].cummax()
    df = df.dropna().reset_index(drop=True)  # remove NaNs and reset index
    return df


def add_volume_and_technical_features(df, window=14, bollinger_k=2):
    # Volume-based features
    df["volume_ratio"] = df["usd_volume"] / df["usd_volume"].rolling(window).mean()
    df["intraday_range"] = (df["high"] - df["low"]) / df["low"]
    df["close_to_high"] = (df["high"] - df["close"]) / df["high"]
    df["close_to_low"] = (df["close"] - df["low"]) / df["low"]

    # RSI Calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()

    df["bollinger_middle"] = rolling_mean
    df["bollinger_upper"] = rolling_mean + bollinger_k * rolling_std
    df["bollinger_lower"] = rolling_mean - bollinger_k * rolling_std
    df = df.dropna().reset_index(drop=True)  # remove NaNs and reset index
    return df


def getSymbolToDf(path: str, threshold: int = 100):
    df = pd.read_csv(path)
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")
    symbol_to_df = {}
    for symbol, mini_df in df.groupby(by="symbol"):
        if len(mini_df) < threshold:
            continue
        mini_df.sort_values(by="date", inplace=True)
        mini_df = addReturns(mini_df)
        mini_df = add_ohlc_features(mini_df)
        mini_df = add_volume_and_technical_features(mini_df)

        symbol_to_df[symbol] = clean_crypto_df(mini_df.reset_index(drop=True))

    return symbol_to_df
