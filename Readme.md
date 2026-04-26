# Tesla Stock Sentiment Analyzer 📈📰

A machine learning project that predicts **Tesla (TSLA) stock movement** by combining **historical stock market data** with **financial news sentiment analysis**.

## Project Overview

This project uses:

* **Tesla historical stock data (2021–2025)**
* **Financial news articles related to Tesla**
* **Sentiment scores from Alpha Vantage API**
* **Technical indicators**
* **Machine Learning models**
* **Trading signal generation + backtesting**

The goal is to analyze whether **news sentiment + technical indicators** can help predict stock movement.

---

## Features

✔ Historical stock data preprocessing
✔ News sentiment extraction using Alpha Vantage API
✔ Sentiment classification (Positive / Neutral / Negative)
✔ Feature engineering with technical indicators:

* RSI
* MACD
* ATR
* Bollinger Bands
* OBV
* Moving averages
* Lag features
* Momentum features
* Volatility features

✔ Multiple ML models:

* Logistic Regression
* Random Forest
* SVM
* XGBoost
* KNN

✔ Ensemble prediction
✔ Trading signal generation
✔ Backtesting
✔ Visualization of results

---

## Dataset

### Stock Data

Historical Tesla stock OHLCV data:

* Open
* High
* Low
* Close
* Volume

Period:
**Jan 2021 – Jan 2025**

### News Data

Fetched using **Alpha Vantage News API**

Includes:

* Headline
* Publication date
* Overall sentiment score
* Ticker-specific sentiment score
* Sentiment label

---

## Methodology

1. Data Collection
2. Data Cleaning
3. Sentiment Analysis
4. Feature Engineering
5. Train/Test Split (Chronological)
6. Model Training
7. Model Evaluation
8. Ensemble Prediction
9. Trading Signal Generation
10. Backtesting & Visualization

---

## Results

Best model:
**SVM**

Metrics:

* Accuracy: ~48–52%
* F1 Score: ~0.55
* Positive backtest return achieved

This highlights the challenge of financial forecasting while showing meaningful predictive patterns.

---

## Installation

```bash
pip install -r requirements.txt
```

Install OpenMP (Mac only):

```bash
brew install libomp
```

---

## Run Project

```bash
python3 main.py
```

---

## Project Structure

```text
Stock-Sentiment-Analyser/
│── data/
│── models/
│── outputs/
│── src/
│── main.py
│── requirements.txt
│── README.md
```

---

## Future Improvements

* Hyperparameter tuning
* Deep learning models (LSTM/Transformer)
* Multi-stock support
* Real-time prediction dashboard
* Better trading strategy optimization

---

## Author
Developed as part of the **AI & ML Lab Project**  
B.Tech Mathematics & Computing
