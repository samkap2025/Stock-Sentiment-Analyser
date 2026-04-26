import matplotlib.pyplot as plt
import numpy as np


def plot_price_trends(df):
    """Plot stock price with moving averages"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)

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
    """Plot sentiment score over time"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Sentiment score over time
        if 'sentiment_score' in df.columns:
            ax1.bar(df.index, df['sentiment_score'], alpha=0.6, color='steelblue')
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_title('Daily Sentiment Score')
            ax1.set_ylabel('Sentiment')
            ax1.grid(True, alpha=0.3)

        # Close price with sentiment coloring
        if 'sentiment_score' in df.columns:
            colors = ['green' if s > 0 else 'red' for s in df['sentiment_score']]
        else:
            colors = 'steelblue'

        ax2.scatter(df.index, df['Close'], c=colors, alpha=0.5)
        ax2.plot(df.index, df['Close'], alpha=0.3)
        ax2.set_title('Price with Sentiment')
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/sentiment_trends.png', dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error plotting sentiment trends: {e}")


def plot_trading_signals(df):
    """Plot trading signals on price chart"""
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')

        # Buy signals
        if 'signal' in df.columns:
            buy_idx = df[df['signal'].isin(['BUY', 'STRONG_BUY'])].index
            if len(buy_idx) > 0:
                plt.scatter(buy_idx, df.loc[buy_idx, 'Close'],
                            marker='^', color='green', s=100, label='Buy Signal', zorder=5)

            # Sell signals
            sell_idx = df[df['signal'].isin(['SELL', 'STRONG_SELL'])].index
            if len(sell_idx) > 0:
                plt.scatter(sell_idx, df.loc[sell_idx, 'Close'],
                            marker='v', color='red', s=100, label='Sell Signal', zorder=5)

        plt.title('Trading Signals on Price Chart')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/trading_signals.png', dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error plotting trading signals: {e}")


def plot_backtest_results(df):
    """Plot backtest performance"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Cumulative returns
        if 'cumulative_return' in df.columns:
            ax1.plot(df.index, (df['cumulative_return'] - 1) * 100,
                     label='Strategy', linewidth=2, color='blue')

        if 'buyhold_cumulative' in df.columns:
            ax1.plot(df.index, (df['buyhold_cumulative'] - 1) * 100,
                     label='Buy & Hold', linewidth=2, alpha=0.7, color='green')

        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        try:
            if 'strategy_return' in df.columns:
                cumsum = (1 + df['strategy_return'].fillna(0)).cumprod()
                running_max = cumsum.expanding().max()
                drawdown = (cumsum - running_max) / running_max * 100
                ax2.fill_between(df.index, drawdown, alpha=0.3, color='red')
                ax2.set_title('Strategy Drawdown')
                ax2.set_ylabel('Drawdown (%)')
        except:
            pass

        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/backtest_results.png', dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error plotting backtest results: {e}")


def plot_model_comparison(evaluation_results):
    """Plot model performance comparison"""
    try:
        models = list(evaluation_results.keys())
        accuracies = [evaluation_results[m]['accuracy'] for m in models]
        precisions = [evaluation_results[m]['precision'] for m in models]
        recalls = [evaluation_results[m]['recall'] for m in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, accuracies, width, label='Accuracy', color='steelblue')
        ax.bar(x, precisions, width, label='Precision', color='orange')
        ax.bar(x + width, recalls, width, label='Recall', color='green')

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error plotting model comparison: {e}")