import numpy as np
import pandas as pd


def simple_backtest(df):
    """
    Simple backtest of trading strategy
    """
    df_backtest = df.copy()

    # Convert signals to positions
    df_backtest['position'] = 0
    df_backtest.loc[df_backtest['signal'] == 'BUY', 'position'] = 1
    df_backtest.loc[df_backtest['signal'] == 'STRONG_BUY', 'position'] = 1
    df_backtest.loc[df_backtest['signal'] == 'SELL', 'position'] = -1
    df_backtest.loc[df_backtest['signal'] == 'STRONG_SELL', 'position'] = -1

    # lowercase close column
    df_backtest['daily_return'] = df_backtest['close'].pct_change()

    df_backtest['strategy_return'] = (
        df_backtest['position'].shift(1)
        * df_backtest['daily_return']
    )

    df_backtest['cumulative_return'] = (
        1 + df_backtest['strategy_return'].fillna(0)
    ).cumprod()

    return df_backtest


def calculate_backtest_metrics(df, risk_free_rate=0.02):
    strategy_returns = df['strategy_return'].dropna()

    if len(strategy_returns) == 0:
        return {
            'total_return_%': 0,
            'annual_return_%': 0,
            'annual_volatility_%': 0,
            'sharpe_ratio': 0,
            'max_drawdown_%': 0,
            'win_rate_%': 0
        }

    final_value = df['cumulative_return'].iloc[-1]
    total_return = (final_value - 1) * 100

    annual_return = (1 + strategy_returns.mean()) ** 252 - 1
    annual_volatility = strategy_returns.std() * np.sqrt(252)

    if annual_volatility > 0:
        sharpe_ratio = (
            annual_return - risk_free_rate
        ) / annual_volatility
    else:
        sharpe_ratio = 0

    cumsum = (1 + strategy_returns).cumprod()
    running_max = cumsum.expanding().max()
    drawdown = (cumsum - running_max) / running_max
    max_drawdown = drawdown.min()

    wins = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        'total_return_%': total_return,
        'annual_return_%': annual_return * 100,
        'annual_volatility_%': annual_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_%': max_drawdown * 100,
        'win_rate_%': win_rate * 100
    }


def compare_with_buyhold(df):
    df['buyhold_return'] = df['close'].pct_change()

    df['buyhold_cumulative'] = (
        1 + df['buyhold_return'].fillna(0)
    ).cumprod()

    strategy_final = df['cumulative_return'].iloc[-1]
    buyhold_final = df['buyhold_cumulative'].iloc[-1]

    print(f"Strategy Final Value: {strategy_final:.2f}")
    print(f"Buy-Hold Final Value: {buyhold_final:.2f}")
    print(
        f"Strategy Outperformance: "
        f"{(strategy_final-buyhold_final)*100:.2f}%"
    )

    return df