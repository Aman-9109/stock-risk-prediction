import yfinance as yf
import pandas as pd
import os


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")


def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance
    """

    print(f"Downloading data for {ticker}...")

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # If MultiIndex columns exist → flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.reset_index(inplace=True)

    return df


def save_raw_data(df, ticker):
    """
    Saves raw data to project root data/raw folder
    """

    os.makedirs(DATA_RAW_PATH, exist_ok=True)

    file_path = os.path.join(DATA_RAW_PATH, f"{ticker}.csv")

    df.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")


if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "GOOGL"]

    start_date = "2015-01-01"
    end_date = "2024-01-01"

    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date)
        save_raw_data(df, ticker)
