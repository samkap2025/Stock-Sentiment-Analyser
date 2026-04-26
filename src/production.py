import sys

sys.path.append('src/')

# Import all modules
from preprocessing import load_stock_data
from sentiment_analysis import analyze_sentiment_vader
from feature_engineering import add_technical_indicators
from model_training import train_all_models
from backtesting import simple_backtest
from visualisation import (
    plot_price_trends, plot_sentiment_trends, plot_trading_signals
)


def predict_tomorrow(ticker):
    '''Predict price movement for tomorrow'''

    # Fetch latest data
    latest_stock = fetch_stock_data(ticker, days=60)
    latest_news = fetch_news(ticker, days=7)

    # Prepare features
    X_latest = engineer_features(latest_stock)
    X_scaled = scaler.transform(X_latest[-1:])

    # Get predictions
    models = load_models()
    ensemble_pred = ensemble_weighted_average(models, X_scaled)
    confidence = add_confidence_scores(models, X_scaled)

    # Get sentiment
    sentiment = analyze_latest_sentiment(latest_news)

    # Generate signal
    signal = generate_trading_signal(ensemble_pred[0], sentiment, confidence[0])

    return {
        'ticker': ticker,
        'prediction': 'UP' if ensemble_pred[0] == 1 else 'DOWN',
        'signal': signal,
        'confidence': float(confidence[0]),
        'sentiment': float(sentiment)
    }

def main():
    print('=' * 50)
    print('Stock Price Predictor Pipeline')
    print('=' * 50)

    # Step 1: Collect data
    print('\nStep 1: Collecting data...')
    stock_df = fetch_stock_data('AAPL', '2022-01-01', '2024-01-01')
    news_df = fetch_news('AAPL')

    # Step 2: Clean data
    print('Step 2: Cleaning data...')
    stock_df = clean_stock_data(stock_df)
    news_df['text'] = news_df['headline'].apply(clean_news_text)

    # Step 3: Sentiment analysis
    print('Step 3: Analyzing sentiment...')
    news_df['sentiment'] = news_df['text'].apply(
        lambda x: analyze_sentiment_vader(x)[0]
    )

    # Step 4: Feature engineering
    print('Step 4: Engineering features...')
    stock_df = add_technical_indicators(stock_df)

    # Step 5: Train models
    print('Step 5: Training models...')
    models = train_all_models(X_train, y_train)

    # Step 6: Backtest
    print('Step 6: Backtesting strategy...')
    backtest_df = simple_backtest(test_df)

    # Step 7: Visualize
    print('Step 7: Creating visualizations...')
    plot_price_trends(stock_df)
    plot_sentiment_trends(stock_df)
    plot_trading_signals(test_df)

    print('\nPipeline complete!')


if __name__ == '__main__':
    main()