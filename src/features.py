import pandas as pd
import numpy as np


def calculate_rsi(data, window=14):
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_technical_indicators(df):
    df = df.copy()

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # Moving averages
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()

    # Volatility
    df["Volatility_7"] = df["Daily_Return"].rolling(7).std()
    df["Volatility_30"] = df["Daily_Return"].rolling(30).std()

    # Drawdown
    df["Rolling_Max"] = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / df["Rolling_Max"] - 1

    # RSI
    df["RSI_14"] = calculate_rsi(df)

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    # Sharpe Ratio
    df["Rolling_Return_Mean_30"] = df["Daily_Return"].rolling(30).mean()
    df["Rolling_Return_Std_30"] = df["Daily_Return"].rolling(30).std()
    df["Sharpe_30"] = df["Rolling_Return_Mean_30"] / df["Rolling_Return_Std_30"]

    df = df.dropna().reset_index(drop=True)

    return df


def create_binary_target(df):
    df = df.copy()

    df["Future_Volatility_7"] = df["Volatility_7"].shift(-7)

    df = df.dropna().reset_index(drop=True)

    threshold = df["Future_Volatility_7"].quantile(0.66)

    df["High_Risk"] = (df["Future_Volatility_7"] > threshold).astype(int)

    return df
