import os
import requests
import yfinance as yf

TICKER = "TSLA"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

ALPHA_VANTAGE_API_KEY = "5LIXX5SPS31TOT3N"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def fetch_stock_data():
    print("Downloading stock data...")

    df = yf.download(TICKER, start=START_DATE, end=END_DATE)

    file_path = os.path.join(DATA_DIR, "tsla_stock_raw.csv")
    df.to_csv(file_path)

    print(f"Stock data saved to {file_path}")

def fetch_news_data():
    print("Fetching raw news data...")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": TICKER,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    file_path = os.path.join(DATA_DIR, "tsla_news_raw.json")

    with open(file_path, "w") as f:
        import json
        json.dump(data, f, indent=4)

    print(f"Raw news saved to {file_path}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Saving to:", DATA_DIR)

    fetch_stock_data()
    fetch_news_data()


if __name__ == "__main__":
    main()