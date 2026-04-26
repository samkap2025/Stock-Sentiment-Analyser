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
        self.df = df.copy()
        self.scaler = None

    def add_technical_indicators(self):
        """Add technical indicators"""
        print("Adding technical indicators...")
        df = self.df

        # Moving averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()

        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)

        # MACD
        macd_result = ta.macd(df['close'])
        if isinstance(macd_result, pd.DataFrame):
            df['MACD'] = macd_result.iloc[:, 0]
        else:
            df['MACD'] = macd_result

        # ATR
        df['ATR'] = ta.atr(
            df['high'],
            df['low'],
            df['close']
        )

        # Bollinger Bands
        bb_result = ta.bbands(df['close'])
        if isinstance(bb_result, pd.DataFrame):
            df['Bollinger_High'] = bb_result.iloc[:, 0]
            df['Bollinger_Low'] = bb_result.iloc[:, 2]

        # OBV
        df['OBV'] = ta.obv(
            df['close'],
            df['volume']
        )

        print("✓ Technical indicators added")
        return df

    def add_price_features(self):
        """Add price-based features"""
        print("Adding price features...")
        df = self.df

        df['daily_return'] = df['close'].pct_change()
        df['price_change'] = df['close'] - df['open']
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['volume_change'] = df['volume'].pct_change()

        print("✓ Price features added")
        return df

    def handle_missing_values(self):
        """Handle missing values"""
        print("Handling missing values...")
        df = self.df

        df = df.ffill().bfill()

        initial_rows = len(df)
        df = df.dropna()
        removed = initial_rows - len(df)

        if removed > 0:
            print(f"Removed {removed} rows with NaN values")

        print("✓ Missing values handled")
        return df

    def normalize_features(self):
        """Normalize only numeric features"""
        print("Normalizing features...")
        df = self.df

        exclude_cols = [
            'target',
            'signal',
            'position',
            'close',
            'open',
            'high',
            'low',
            'volume',
            'price'
        ]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        print(f"✓ {len(feature_cols)} features normalized")
        return df

    def create_all_features(self):
        """Complete feature engineering pipeline"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60 + "\n")

        self.df = self.add_price_features()
        self.df = self.add_technical_indicators()
        self.df = self.handle_missing_values()
        self.df = self.normalize_features()

        print("\n✓ Feature engineering complete")
        print(f"Total features: {len(self.df.columns)}")
        print(f"Total rows: {len(self.df)}")

        return self.df