import matplotlib.pyplot as plt

def plot_price_trends(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    plt.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
    plt.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
    plt.title('Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/price_trends.png')
    plt.show()


def plot_sentiment_trends(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Sentiment score over time
    ax1.bar(df.index, df['sentiment_score'], alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_title('Daily Sentiment Score')
    ax1.set_ylabel('Sentiment')
    ax1.grid(True, alpha=0.3)

    # Close price with sentiment coloring
    colors = ['green' if s > 0 else 'red' for s in df['sentiment_score']]
    ax2.scatter(df.index, df['Close'], c=colors, alpha=0.5)
    ax2.plot(df.index, df['Close'], alpha=0.3)
    ax2.set_title('Price with Sentiment Background')
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/sentiment_trends.png')
    plt.show()


def plot_trading_signals(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=2)

    # Buy signals
    buy_idx = df[df['signal'].isin(['BUY', 'STRONG_BUY'])].index
    plt.scatter(buy_idx, df.loc[buy_idx, 'Close'],
                marker='^', color='green', s=100, label='Buy Signal')

    # Sell signals
    sell_idx = df[df['signal'].isin(['SELL', 'STRONG_SELL'])].index
    plt.scatter(sell_idx, df.loc[sell_idx, 'Close'],
                marker='v', color='red', s=100, label='Sell Signal')

    plt.title('Trading Signals on Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/trading_signals.png')
    plt.show()


def plot_backtest_results(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Cumulative returns
    ax1.plot(df.index, (df['cumulative_return'] - 1) * 100,
             label='Strategy', linewidth=2)
    ax1.plot(df.index, (df['buyhold_return'] - 1) * 100,
             label='Buy & Hold', linewidth=2, alpha=0.7)
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True)

    # Drawdown
    cumsum = (1 + df['strategy_return']).cumprod()
    running_max = cumsum.expanding().max()
    drawdown = (cumsum - running_max) / running_max * 100
    ax2.fill_between(df.index, drawdown, alpha=0.3)
    ax2.set_title('Strategy Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/backtest_results.png')
    plt.show()


def plot_model_comparison(evaluation_results):
    models = list(evaluation_results.keys())
    accuracies = [evaluation_results[m]['accuracy'] for m in models]
    precisions = [evaluation_results[m]['precision'] for m in models]
    recalls = [evaluation_results[m]['recall'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, accuracies, width, label='Accuracy')
    ax.bar(x, precisions, width, label='Precision')
    ax.bar(x + width, recalls, width, label='Recall')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    plt.show()

