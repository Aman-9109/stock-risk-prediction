import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from features import add_technical_indicators, create_binary_target


def train_model():
    print("Training started...\n")

 

    tickers = ["AAPL", "MSFT", "GOOGL"]

    all_data = []

    for ticker in tickers:
        print(f"Loading {ticker}...")

        df = pd.read_csv(f"data/raw/{ticker}.csv")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)

        # Add ticker column
        df["Ticker"] = ticker

        # Feature Engineering
        df = add_technical_indicators(df)

        # Target Creation
        df = create_binary_target(df)

        all_data.append(df)

    # Combine all stocks
    df = pd.concat(all_data, ignore_index=True)

    print("\nTotal combined rows:", len(df))

   

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

    X = df[feature_columns]
    y = df["High_Risk"]



    print("\nRunning TimeSeries Cross Validation...")

    tscv = TimeSeriesSplit(n_splits=5)

    model_cv = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )

    cv_scores = cross_val_score(model_cv, X, y, cv=tscv, scoring="accuracy")

    print("TimeSeries CV Scores:", cv_scores)
    print("Average CV Accuracy:", round(cv_scores.mean(), 4))


    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Class Weight for Imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\nTraining Final Weighted Model...")

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nFinal Test Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # Save Model
    joblib.dump(model, "models/xgb_model.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()
