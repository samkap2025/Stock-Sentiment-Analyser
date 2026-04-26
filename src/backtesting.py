import numpy as np
import pandas as pd


def simple_backtest(df):
    """
    Simple backtest of trading strategy

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'signal' and 'Close' columns

    Returns:
    --------
    pd.DataFrame
        DataFrame with backtest results
    """
    df_backtest = df.copy()

    # Convert signals to positions
    df_backtest['position'] = 0
    df_backtest.loc[df_backtest['signal'] == 'BUY', 'position'] = 1
    df_backtest.loc[df_backtest['signal'] == 'STRONG_BUY', 'position'] = 1
    df_backtest.loc[df_backtest['signal'] == 'SELL', 'position'] = -1
    df_backtest.loc[df_backtest['signal'] == 'STRONG_SELL', 'position'] = -1

    # Calculate returns
    df_backtest['daily_return'] = df_backtest['Close'].pct_change()
    df_backtest['strategy_return'] = df_backtest['position'].shift(1) * df_backtest['daily_return']
    df_backtest['cumulative_return'] = (1 + df_backtest['strategy_return'].fillna(0)).cumprod()

    return df_backtest


def calculate_backtest_metrics(df, risk_free_rate=0.02):
    """
    Calculate backtest performance metrics

    Parameters:
    -----------
    df : pd.DataFrame
        Backtest DataFrame with returns
    risk_free_rate : float
        Risk-free rate for Sharpe ratio

    Returns:
    --------
    dict : Performance metrics
    """
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

    # Total return
    final_value = df['cumulative_return'].iloc[-1] if 'cumulative_return' in df.columns else 1
    total_return = (final_value - 1) * 100

    # Annual return
    annual_return = (1 + strategy_returns.mean()) ** 252 - 1

    # Annual volatility
    annual_volatility = strategy_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    if annual_volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe_ratio = 0

    # Maximum Drawdown
    try:
        cumsum = (1 + strategy_returns).cumprod()
        running_max = cumsum.expanding().max()
        drawdown = (cumsum - running_max) / running_max
        max_drawdown = drawdown.min()
    except:
        max_drawdown = 0

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
    """
    Compare strategy with buy-and-hold

    Parameters:
    -----------
    df : pd.DataFrame
        Backtest DataFrame
    """
    # Buy-and-Hold strategy
    df['buyhold_return'] = df['Close'].pct_change()
    df['buyhold_cumulative'] = (1 + df['buyhold_return'].fillna(0)).cumprod()

    # Compare final values
    strategy_final = df['cumulative_return'].iloc[-1]
    buyhold_final = df['buyhold_cumulative'].iloc[-1]

    print(f'Strategy Final Value: {strategy_final:.2f}')
    print(f'Buy-Hold Final Value: {buyhold_final:.2f}')
    print(f'Strategy Outperformance: {(strategy_final - buyhold_final) * 100:.2f}%')

    return df