import numpy as np

def simple_backtest(df):
    df['position'] = 0
    df.loc[df['signal'] == 'BUY', 'position'] = 1
    df.loc[df['signal'] == 'SELL', 'position'] = -1

    # Calculate returns
    df['daily_return'] = df['Close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    return df


def calculate_backtest_metrics(df, risk_free_rate=0.02):
    strategy_returns = df['strategy_return'].dropna()
    buy_hold_returns = df['daily_return'].dropna()

    # Total return
    total_return = (df['cumulative_return'].iloc[-1] - 1) * 100

    # Annual return
    num_trading_days = len(strategy_returns)
    annual_return = (1 + strategy_returns.mean()) ** 252 - 1

    # Annual volatility
    annual_volatility = strategy_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    # Maximum Drawdown
    cumsum = (1 + strategy_returns).cumprod()
    running_max = cumsum.expanding().max()
    drawdown = (cumsum - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win Rate
    wins = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    win_rate = wins / total_trades if total_trades > 0 else 0

    metrics = {
        'total_return_%': total_return,
        'annual_return_%': annual_return * 100,
        'annual_volatility_%': annual_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_%': max_drawdown * 100,
        'win_rate_%': win_rate * 100
    }

    return metrics


def compare_with_buyhold(df):
    # Buy-and-Hold strategy
    df['buyhold_return'] = (1 + df['daily_return']).cumprod()

    # Compare final values
    strategy_final = df['cumulative_return'].iloc[-1]
    buyhold_final = df['buyhold_return'].iloc[-1]

    print(f'Strategy Final Value: {strategy_final:.2f}')
    print(f'Buy-Hold Final Value: {buyhold_final:.2f}')
    print(f'Strategy Outperformance: {(strategy_final - buyhold_final) * 100:.2f}%')