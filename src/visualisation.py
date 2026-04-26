import matplotlib.pyplot as plt
import numpy as np


def plot_price_trends(df):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price', linewidth=2)

        if 'SMA_20' in df.columns:
            plt.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)

        if 'SMA_50' in df.columns:
            plt.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)

        plt.title('Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/price_trends.png', dpi=100)
        plt.close()

    except Exception as e:
        print(f"Error plotting price trends: {e}")


def plot_sentiment_trends(df):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        if 'sentiment_score' in df.columns:
            ax1.bar(df.index, df['sentiment_score'], alpha=0.6)
            ax1.axhline(y=0, linestyle='--')
            ax1.set_title('Daily Sentiment Score')
            ax1.grid(True, alpha=0.3)

        colors = ['green' if s > 0 else 'red' for s in df['sentiment_score']]
        ax2.scatter(df.index, df['close'], c=colors, alpha=0.5)
        ax2.plot(df.index, df['close'], alpha=0.3)

        ax2.set_title('Price with Sentiment')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/sentiment_trends.png', dpi=100)
        plt.close()

    except Exception as e:
        print(f"Error plotting sentiment trends: {e}")


def plot_trading_signals(df):
    try:
        plt.figure(figsize=(14, 6))

        plt.plot(df.index, df['close'], label='Close Price', linewidth=2)

        if 'signal' in df.columns:

            buy_idx = df[df['signal'].isin(['BUY', 'STRONG_BUY'])].index
            if len(buy_idx):
                plt.scatter(
                    buy_idx,
                    df.loc[buy_idx, 'close'],
                    marker='^',
                    s=100,
                    label='BUY'
                )

            sell_idx = df[df['signal'].isin(['SELL', 'STRONG_SELL'])].index
            if len(sell_idx):
                plt.scatter(
                    sell_idx,
                    df.loc[sell_idx, 'close'],
                    marker='v',
                    s=100,
                    label='SELL'
                )

        plt.title('Trading Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/trading_signals.png', dpi=100)
        plt.close()

    except Exception as e:
        print(f"Error plotting trading signals: {e}")


def plot_backtest_results(df):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        if 'cumulative_return' in df.columns:
            ax.plot(
                df.index,
                (df['cumulative_return'] - 1) * 100,
                label='Strategy'
            )

        if 'buyhold_cumulative' in df.columns:
            ax.plot(
                df.index,
                (df['buyhold_cumulative'] - 1) * 100,
                label='Buy & Hold'
            )

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/backtest_results.png', dpi=100)
        plt.close()

    except Exception as e:
        print(f"Error plotting backtest results: {e}")


def plot_model_comparison(evaluation_results):
    try:
        models = list(evaluation_results.keys())
        accuracies = [evaluation_results[m]['accuracy'] for m in models]

        plt.figure(figsize=(10, 5))
        plt.bar(models, accuracies)

        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=100)
        plt.close()

    except Exception as e:
        print(f"Error plotting model comparison: {e}")