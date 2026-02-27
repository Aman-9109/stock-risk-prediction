import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt

# Add src folder to path
sys.path.append(os.path.abspath("../src"))

from features import add_technical_indicators, create_binary_target

st.set_page_config(page_title="Stock Risk Dashboard", layout="wide")

st.title("📊 Stock Risk Prediction Dashboard")
st.markdown("Predict High Volatility Risk using XGBoost")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("../models/xgb_model.pkl")

model = load_model()

# -----------------------------
# Select Ticker
# -----------------------------
ticker = st.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL"])

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(f"../data/raw/{ticker}.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

df = add_technical_indicators(df)
df = create_binary_target(df)

# -----------------------------
# Feature Selection
# -----------------------------
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

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📅 Latest Date")
st.write(latest_row["Date"])

st.subheader("⚠️ Risk Prediction")

if prediction == 1:
    st.error(f"High Risk Detected (Probability: {probability:.2f})")
else:
    st.success(f"Normal Risk (Probability: {probability:.2f})")

# -----------------------------
# Volatility Chart
# -----------------------------
st.subheader("📈 30-Day Rolling Volatility")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Date"], df["Volatility_30"])
ax.set_title("Volatility Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")

st.pyplot(fig)

st.markdown("---")
st.caption("Built with XGBoost + TimeSeries Cross Validation")
