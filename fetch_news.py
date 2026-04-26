import requests
import json
import time
import os

API_KEY = "KK16A65906QXVHWU"
TICKER = "TSLA"

periods = [
    ("20210101T0000", "20211231T2359"),
    ("20220101T0000", "20221231T2359"),
    ("20230101T0000", "20231231T2359"),
    ("20240101T0000", "20241231T2359"),
    ("20250101T0000", "20250101T2359"),
]

all_feed = []

print("\nDownloading historical Tesla news...\n")

for start, end in periods:

    print(f"Fetching {start} → {end}")

    url = (
        f"https://www.alphavantage.co/query?"
        f"function=NEWS_SENTIMENT"
        f"&tickers={TICKER}"
        f"&time_from={start}"
        f"&time_to={end}"
        f"&limit=1000"
        f"&apikey={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if "feed" in data:
        all_feed.extend(data["feed"])
        print("✓ Articles fetched:", len(data["feed"]))
    else:
        print("✗ No feed found")
        print(data)

    time.sleep(15)

final_json = {
    "feed": all_feed
}

os.makedirs("data", exist_ok=True)

with open("data/tsla_news_raw.json", "w", encoding="utf-8") as f:
    json.dump(final_json, f, indent=4)

print("\n✓ Saved successfully")
print("Location: data/tsla_news_raw.json")
print("Total articles:", len(all_feed))