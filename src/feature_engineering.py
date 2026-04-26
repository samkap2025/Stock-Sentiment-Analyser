import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Engineer features for machine learning models
    """

    def __init__(self, df):
        """
        Initialize feature engineer

        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe with stock and sentiment data
        """
        self.df = df.copy()
        self.scaler = None

    def add_technical_indicators(self):
        """Add technical indicators to dataframe"""
        print("Adding technical indicators...")
        df = self.df

        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # Momentum Indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)

        macd_result = ta.macd(df['Close'])
        df['MACD'] = macd_result.iloc[:, 0] if isinstance(macd_result, pd.DataFrame) else macd_result

        # Volatility Indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])

        bb_result = ta.bbands(df['Close'])
        if isinstance(bb_result, pd.DataFrame):
            df['Bollinger_High'] = bb_result.iloc[:, 0]
            df['Bollinger_Low'] = bb_result.iloc[:, 2]

        # Volume Indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        print("✓ Technical indicators added")
        return df

    def add_price_features(self):
        """Add price-based features"""
        print("Adding price features...")
        df = self.df

        # Daily returns
        df['daily_return'] = df['Close'].pct_change()

        # Price change
        df['price_change'] = df['Close'] - df['Open']

        # High-Low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']

        # Volume change
        df['volume_change'] = df['Volume'].pct_change()

        print("✓ Price features added")
        return df

    def handle_missing_values(self):
        """Handle missing values in features"""
        print("Handling missing values...")
        df = self.df

        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Drop any remaining NaN
        initial_rows = len(df)
        df = df.dropna()
        removed = initial_rows - len(df)

        if removed > 0:
            print(f"  Removed {removed} rows with NaN values")

        print("✓ Missing values handled")
        return df

    def normalize_features(self):
        """Normalize features using StandardScaler"""
        print("Normalizing features...")
        df = self.df

        # Exclude target and date-related columns
        exclude_cols = ['target', 'signal', 'position', 'Close', 'Open', 'High', 'Low', 'Volume']
        feature_cols = [col for col in df.columns if
                        col not in exclude_cols and col not in df.select_dtypes(include=[np.datetime64]).columns]

        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        print(f"✓ {len(feature_cols)} features normalized")
        return df

    def create_all_features(self):
        """Execute complete feature engineering pipeline"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60 + "\n")

        self.df = self.add_price_features()
        self.df = self.add_technical_indicators()
        self.df = self.handle_missing_values()
        self.df = self.normalize_features()

        print(f"\n✓ Feature engineering complete")
        print(f"  Total features: {len(self.df.columns)}")
        print(f"  Total rows: {len(self.df)}")

        return self.df