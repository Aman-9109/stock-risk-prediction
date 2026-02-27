# 📊 Multi-Asset Stock Risk Prediction System

## Overview
Built a production-ready stock risk prediction system using XGBoost with TimeSeries cross-validation to detect high-volatility regimes across multiple assets (AAPL, MSFT, GOOGL).

## Features
- Multi-asset training
- Financial feature engineering (RSI, Bollinger Bands, Volatility, Sharpe Ratio)
- Binary volatility regime classification
- TimeSeries cross-validation
- Weighted XGBoost
- SHAP explainability
- Streamlit dashboard for live prediction

## Project Structure
src/
  - features.py
  - train.py
  - predict.py
app/
  - app.py
models/
data/

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python src/train.py

### 3. Run dashboard
cd app
streamlit run app.py

## Results
- ~67% average CV accuracy
- 74% recall for High Risk regime
- Multi-asset generalization

## Author
Your Name