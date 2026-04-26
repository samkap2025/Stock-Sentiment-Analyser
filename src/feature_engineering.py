import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

def add_technical_indicators(df):
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # Momentum Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])[0]
    df['MACD_signal'] = ta.macd(df['Close'])[1]

    # Volatility Indicators
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['Bollinger_High'] = ta.bbands(df['Close'])[0]
    df['Bollinger_Low'] = ta.bbands(df['Close'])[1]

    # Volume Indicators
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    return df


def prepare_features(df):
    # Fill NaN values with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Drop any remaining NaN
    df = df.dropna()

    # Normalize features to 0-1 range
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != 'target']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler


def add_price_features(df):
    # Daily returns
    df['daily_return'] = df['Close'].pct_change()

    # Price change
    df['price_change'] = df['Close'] - df['Open']

    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']

    # Volume change
    df['volume_change'] = df['Volume'].pct_change()

    return df