import pandas as pd
import joblib
from features import add_technical_indicators, create_binary_target


def predict_latest_risk(ticker="AAPL"):

    print("Loading model...")
    model = joblib.load("models/xgb_model.pkl")

    print("Loading raw data...")
    df = pd.read_csv(f"data/raw/{ticker}.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print("Generating features...")
    df = add_technical_indicators(df)
    df = create_binary_target(df)

    feature_columns = [
        "Close",
        "Volume",
        "Daily_Return",
        "MA_7",
        "MA_30",
        "Volatility_7",
        "Volatility_30",
        "Drawdown",
        "RSI_14",
        "BB_Upper",
        "BB_Lower",
        "Sharpe_30"
    ]

    latest_row = df.iloc[-1]

    X_latest = latest_row[feature_columns].values.reshape(1, -1)

    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0][1]

    print("\n===== Latest Risk Prediction =====")

    print("Date:", latest_row["Date"])
    print("Predicted High Risk:", prediction)
    print("Probability of High Risk:", round(probability, 4))

    if prediction == 1:
        print("⚠️ Market entering HIGH RISK regime.")
    else:
        print("✅ Market in normal risk regime.")


if __name__ == "__main__":
    predict_latest_risk("AAPL")
