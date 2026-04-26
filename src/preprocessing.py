import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')


class StockDataPreprocessor:
    """
    Preprocesses stock price data and news data for the stock price predictor.
    Handles cleaning, validation, and feature preparation.
    """

    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.stock_df = None
        self.news_df = None
        self.combined_df = None

    def load_stock_data(self, stock_df):
        """
        Load stock data from a pandas DataFrame.

        Parameters:
        -----------
        stock_df : pd.DataFrame
            Stock data with columns: Date/Index, Open, High, Low, Close, Volume

        Returns:
        --------
        pd.DataFrame
            Loaded stock data
        """
        print(f"Loading stock data...")

        try:
            df = stock_df.copy()

            # If Date is a column, set it as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                # Try to parse the index as datetime
                df.index = pd.to_datetime(df.index)

            # Convert column names to lowercase
            df.columns = df.columns.str.lower()

            self.stock_df = df

            print(f"✓ Stock data loaded: {len(self.stock_df)} rows")
            print(f"  Columns: {list(self.stock_df.columns)}")
            print(f"  Date range: {self.stock_df.index.min()} to {self.stock_df.index.max()}")

            return self.stock_df

        except Exception as e:
            print(f"✗ Error loading stock data: {str(e)}")
            return None

    def load_news_data(self, news_df):
        """
        Load news data from a pandas DataFrame.

        Parameters:
        -----------
        news_df : pd.DataFrame
            News data with columns: date, headline, source (optional), sentiment (optional), sentiment_score (optional)

        Returns:
        --------
        pd.DataFrame
            Loaded news data
        """
        print(f"Loading news data...")

        try:
            df = news_df.copy()

            # Ensure date column exists and is datetime
            if 'date' not in df.columns:
                print("✗ Error: 'date' column is required")
                return None

            df['date'] = pd.to_datetime(df['date'])

            # Ensure headline column exists
            if 'headline' not in df.columns and 'title' in df.columns:
                df['headline'] = df['title']
            elif 'headline' not in df.columns:
                print("✗ Error: 'headline' or 'title' column is required")
                return None

            # Add sentiment column if missing
            if 'sentiment' not in df.columns:
                df['sentiment'] = 'NEUTRAL'

            # Add sentiment_score if missing
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = 0.0

            # Add source if missing
            if 'source' not in df.columns:
                df['source'] = 'Unknown'

            # Sort by date
            df = df.sort_values('date')

            self.news_df = df

            print(f"✓ News data loaded: {len(self.news_df)} articles")
            print(f"  Date range: {self.news_df['date'].min()} to {self.news_df['date'].max()}")
            print(f"  Sentiments: {self.news_df['sentiment'].value_counts().to_dict()}")

            return self.news_df

        except Exception as e:
            print(f"✗ Error loading news data: {str(e)}")
            return None

    def clean_stock_data(self):
        """
        Clean stock data:
        - Remove missing values
        - Remove duplicates
        - Handle outliers
        - Sort by date

        Returns:
        --------
        pd.DataFrame
            Cleaned stock data
        """
        if self.stock_df is None:
            print("✗ Error: Stock data not loaded. Call load_stock_data() first.")
            return None

        print("\nCleaning stock data...")

        df = self.stock_df.copy()

        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"✗ Error: Missing columns: {missing_cols}")
            print(f"  Available columns: {list(df.columns)}")
            return None

        initial_rows = len(df)

        # 1. Remove rows with missing values
        df = df.dropna()
        removed_na = initial_rows - len(df)
        if removed_na > 0:
            print(f"  - Removed {removed_na} rows with missing values")

        # 2. Remove duplicates (keep first occurrence)
        df = df[~df.index.duplicated(keep='first')]

        # 3. Ensure numeric columns
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows where conversion failed
        df = df.dropna()

        # 4. Remove obvious outliers (price spikes > 50% in one day)
        df['price_change_pct'] = df['close'].pct_change().abs()
        outlier_threshold = 0.50  # 50% change
        outliers = (df['price_change_pct'] > outlier_threshold).sum()

        if outliers > 0:
            print(f"  - Found {outliers} potential outliers (>50% daily change)")
            print(f"    Note: These may be legitimate stock splits or major events")

        df = df.drop('price_change_pct', axis=1)

        # 5. Ensure data is sorted by date
        df = df.sort_index()

        # 6. Validate OHLC relationship
        invalid_ohlc = ((df['high'] < df['low']) |
                        (df['high'] < df['open']) |
                        (df['high'] < df['close']) |
                        (df['low'] > df['open']) |
                        (df['low'] > df['close'])).sum()

        if invalid_ohlc > 0:
            print(f"  - Found {invalid_ohlc} rows with invalid OHLC relationships")
            df = df[~((df['high'] < df['low']) |
                      (df['high'] < df['open']) |
                      (df['high'] < df['close']) |
                      (df['low'] > df['open']) |
                      (df['low'] > df['close']))]

        self.stock_df = df

        print(f"✓ Stock data cleaned: {len(df)} rows remaining")
        print(f"  Initial rows: {initial_rows}, Final rows: {len(df)}")
        print(f"  Data type summary:\n{df.dtypes}")

        return self.stock_df

    def clean_news_text(self, text):
        """
        Clean individual news headline/text.

        Parameters:
        -----------
        text : str
            Raw text to clean

        Returns:
        --------
        str
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\?\!\-\'"]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove very short text (likely noise)
        if len(text) < 5:
            return ""

        return text

    def clean_news_data(self):
        """
        Clean news data:
        - Remove missing headlines
        - Clean text
        - Remove duplicates
        - Filter invalid dates

        Returns:
        --------
        pd.DataFrame
            Cleaned news data
        """
        if self.news_df is None:
            print("✗ Error: News data not loaded. Call load_news_data() first.")
            return None

        print("\nCleaning news data...")

        df = self.news_df.copy()
        initial_rows = len(df)

        # 1. Remove rows with missing headlines
        df = df.dropna(subset=['headline'])
        removed_na = initial_rows - len(df)
        if removed_na > 0:
            print(f"  - Removed {removed_na} rows with missing headlines")

        # 2. Remove empty headlines
        df['headline'] = df['headline'].astype(str)
        df = df[df['headline'].str.strip() != '']

        # 3. Clean headline text
        df['headline_cleaned'] = df['headline'].apply(self.clean_news_text)

        # 4. Remove rows where cleaning resulted in empty text
        df = df[df['headline_cleaned'].str.len() > 0]

        # 5. Clean summary if it exists
        if 'summary' in df.columns:
            df['summary'] = df['summary'].fillna('')
            df['summary_cleaned'] = df['summary'].apply(self.clean_news_text)

        # 6. Remove duplicate headlines (case-insensitive)
        df['headline_lower'] = df['headline_cleaned'].str.lower()
        df = df.drop_duplicates(subset=['headline_lower'], keep='first')
        df = df.drop('headline_lower', axis=1)

        # 7. Validate and filter dates
        df = df[df['date'] >= pd.Timestamp('2021-01-01')]  # Filter out very old news

        # 8. Ensure sentiment is one of expected values
        valid_sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        df['sentiment'] = df['sentiment'].fillna('NEUTRAL')
        df['sentiment'] = df['sentiment'].str.upper()
        df = df[df['sentiment'].isin(valid_sentiments)]

        # 9. Ensure sentiment_score is numeric
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

        self.news_df = df.sort_values('date')

        print(f"✓ News data cleaned: {len(self.news_df)} articles remaining")
        print(f"  Initial articles: {initial_rows}, Final articles: {len(self.news_df)}")
        print(f"  Sentiment distribution:")
        for sentiment, count in self.news_df['sentiment'].value_counts().items():
            print(f"    - {sentiment}: {count}")

        return self.news_df

    def create_target_variable(self):
        """
        Create binary target variable for classification:
        - 1: Stock price goes UP (close > close_previous_day)
        - 0: Stock price goes DOWN (close <= close_previous_day)

        The target is SHIFTED so that we predict TOMORROW's movement using TODAY's data.

        Returns:
        --------
        pd.DataFrame
            Stock data with target variable
        """
        if self.stock_df is None:
            print("✗ Error: Stock data not loaded. Call load_stock_data() first.")
            return None

        print("\nCreating target variable...")

        df = self.stock_df.copy()

        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()

        # Create binary target: 1 if UP, 0 if DOWN
        # This represents what HAPPENED on that day
        df['target_actual'] = (df['daily_return'] > 0).astype(int)

        # Shift target UP by 1 row so that:
        # Row N's features predict Row N+1's movement
        # This ensures we're predicting TOMORROW using TODAY's data
        df['target'] = df['target_actual'].shift(-1)

        # Drop the last row (it has no target value for tomorrow)
        df = df[:-1]

        # Drop the temporary column
        df = df.drop('target_actual', axis=1)

        self.stock_df = df

        # Print target distribution
        target_counts = df['target'].value_counts()
        print(f"✓ Target variable created")
        print(f"  - UP (1): {target_counts.get(1, 0)} days")
        print(f"  - DOWN (0): {target_counts.get(0, 0)} days")
        print(f"  - Missing (NaN): {df['target'].isna().sum()} days")

        # Calculate percentage
        total_valid = target_counts.get(1, 0) + target_counts.get(0, 0)
        if total_valid > 0:
            up_pct = (target_counts.get(1, 0) / total_valid) * 100
            print(f"  - Class balance: {up_pct:.2f}% UP, {100 - up_pct:.2f}% DOWN")

        return self.stock_df

    def align_stock_and_news(self):
        """
        Align stock data with news data by date.
        Creates columns for daily sentiment aggregation.

        Returns:
        --------
        pd.DataFrame
            Combined dataframe with stock and aggregated news sentiment
        """
        if self.stock_df is None or self.news_df is None:
            print("✗ Error: Both stock and news data must be loaded and cleaned first.")
            return None

        print("\nAligning stock data with news sentiment...")

        df = self.stock_df.copy()
        news = self.news_df.copy()

        # Create date column for news (just the date part, not time)
        news['date_only'] = news['date'].dt.date

        # Create date column for stock (from index)
        df['date_only'] = df.index.date

        # Initialize sentiment columns
        df['sentiment_score'] = 0.0
        df['positive_count'] = 0
        df['negative_count'] = 0
        df['neutral_count'] = 0
        df['total_articles'] = 0

        # For each stock date, aggregate all news from that date
        for stock_date in df['date_only'].unique():
            news_that_day = news[news['date_only'] == stock_date]

            if len(news_that_day) > 0:
                # Calculate average sentiment score
                avg_sentiment = news_that_day['sentiment_score'].mean()

                # Count sentiments
                sentiment_counts = news_that_day['sentiment'].value_counts()

                # Update DataFrame
                mask = df['date_only'] == stock_date
                df.loc[mask, 'sentiment_score'] = avg_sentiment
                df.loc[mask, 'positive_count'] = sentiment_counts.get('POSITIVE', 0)
                df.loc[mask, 'negative_count'] = sentiment_counts.get('NEGATIVE', 0)
                df.loc[mask, 'neutral_count'] = sentiment_counts.get('NEUTRAL', 0)
                df.loc[mask, 'total_articles'] = len(news_that_day)

        # Drop temporary date column
        df = df.drop('date_only', axis=1)

        self.combined_df = df

        # Print alignment statistics
        days_with_news = (df['total_articles'] > 0).sum()
        print(f"✓ Data aligned")
        print(f"  - Stock data days: {len(df)}")
        print(f"  - Days with news: {days_with_news}")
        print(f"  - Days without news: {len(df) - days_with_news}")
        print(f"  - Average articles per day (with news): {df[df['total_articles'] > 0]['total_articles'].mean():.2f}")

        return self.combined_df

    def handle_missing_values(self, method='forward_fill'):
        """
        Handle missing values in the dataset.

        Parameters:
        -----------
        method : str
            Method to use: 'forward_fill', 'backward_fill', or 'drop'

        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        if self.combined_df is None:
            print("✗ Error: Combined data not created. Call align_stock_and_news() first.")
            return None

        print("\nHandling missing values...")

        df = self.combined_df.copy()
        missing_before = df.isna().sum().sum()

        if method == 'forward_fill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')  # For any remaining NaNs at the start

        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
            df = df.fillna(method='ffill')  # For any remaining NaNs at the end

        elif method == 'drop':
            df = df.dropna()

        missing_after = df.isna().sum().sum()

        print(f"✓ Missing values handled")
        print(f"  - Missing before: {missing_before}")
        print(f"  - Missing after: {missing_after}")
        print(f"  - Rows remaining: {len(df)}")

        self.combined_df = df
        return self.combined_df

    def save_cleaned_data(self, output_path=None):
        """
        Save cleaned and preprocessed data to CSV (optional).

        Parameters:
        -----------
        output_path : str
            Path to save the file. If None, data is only returned in-memory.

        Returns:
        --------
        pd.DataFrame or str
            If output_path provided: returns path where data was saved
            If output_path is None: returns the DataFrame
        """
        if self.combined_df is None:
            print("✗ Error: No processed data to save. Run preprocessing pipeline first.")
            return None

        if output_path is None:
            print("✓ Processed data ready (not saved to file)")
            print(f"  - Rows: {len(self.combined_df)}")
            print(f"  - Columns: {len(self.combined_df.columns)}")
            return self.combined_df

        print(f"\nSaving cleaned data to {output_path}...")

        try:
            self.combined_df.to_csv(output_path)
            print(f"✓ Data saved successfully")
            print(f"  - Rows: {len(self.combined_df)}")
            print(f"  - Columns: {len(self.combined_df.columns)}")
            return output_path

        except Exception as e:
            print(f"✗ Error saving data: {str(e)}")
            return None

    def get_summary_statistics(self):
        """
        Print summary statistics of the processed data.
        """
        if self.combined_df is None:
            print("✗ Error: No processed data. Run preprocessing pipeline first.")
            return

        df = self.combined_df

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        print("\nStock Price Data:")
        print(f"  - Date range: {df.index.min()} to {df.index.max()}")
        print(f"  - Total trading days: {len(df)}")
        print(f"\n  Price Statistics (in USD):")
        print(f"    Close: Min={df['close'].min():.2f}, Max={df['close'].max():.2f}, Mean={df['close'].mean():.2f}")
        print(f"    Volume: Min={df['volume'].min():.0f}, Max={df['volume'].max():.0f}, Mean={df['volume'].mean():.0f}")

        print(f"\nNews Sentiment Data:")
        print(f"  - Total articles: {df['total_articles'].sum():.0f}")
        print(f"  - Days with news: {(df['total_articles'] > 0).sum()}")
        print(f"  - Average sentiment score: {df['sentiment_score'].mean():.4f}")
        print(f"  - Sentiment range: [{df['sentiment_score'].min():.4f}, {df['sentiment_score'].max():.4f}]")

        print(f"\nTarget Variable Distribution:")
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            total = len(df[df['target'].notna()])
            print(f"  - UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0) / total * 100:.1f}%)")
            print(f"  - DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0) / total * 100:.1f}%)")

        print("\nData Quality:")
        print(f"  - Missing values: {df.isna().sum().sum()}")
        print(f"  - Duplicate dates: {df.index.duplicated().sum()}")

        print("=" * 60 + "\n")


def main(stock_df, news_df):
    """
    Main preprocessing pipeline.

    Parameters:
    -----------
    stock_df : pd.DataFrame
        Stock data with columns: Date/Index, Open, High, Low, Close, Volume
    news_df : pd.DataFrame
        News data with columns: date, headline, sentiment (optional), sentiment_score (optional)

    Returns:
    --------
    pd.DataFrame
        Processed data ready for feature engineering
    """
    print("\n" + "=" * 60)
    print("STOCK PRICE PREDICTOR - DATA PREPROCESSING")
    print("=" * 60)

    # Initialize preprocessor
    preprocessor = StockDataPreprocessor(ticker="TSLA")

    # Step 1: Load data from DataFrames
    print("\nStep 1: Loading data...")
    stock_data = preprocessor.load_stock_data(stock_df)
    news_data = preprocessor.load_news_data(news_df)

    if stock_data is None or news_data is None:
        print("\n✗ Failed to load data. Exiting.")
        return None

    # Step 2: Clean stock data
    print("\nStep 2: Cleaning stock data...")
    preprocessor.clean_stock_data()

    # Step 3: Clean news data
    print("\nStep 3: Cleaning news data...")
    preprocessor.clean_news_data()

    # Step 4: Create target variable
    print("\nStep 4: Creating target variable...")
    preprocessor.create_target_variable()

    # Step 5: Align stock and news data
    print("\nStep 5: Aligning stock and news data...")
    preprocessor.align_stock_and_news()

    # Step 6: Handle missing values
    print("\nStep 6: Handling missing values...")
    preprocessor.handle_missing_values(method='forward_fill')

    # Step 7: Get processed data (no file saving)
    print("\nStep 7: Finalizing processed data...")
    processed_data = preprocessor.save_cleaned_data()

    # Step 8: Print summary
    print("\nStep 8: Summary statistics...")
    preprocessor.get_summary_statistics()

    return processed_data


if __name__ == "__main__":
    # Example usage - you would load your CSV files here
    print("\nTo use preprocessing.py, do the following:")
    print("\n1. Load your data:")
    print("   stock_df = pd.read_csv('data/tsla_stock_raw.csv')")
    print("   news_df = pd.read_json('data/tsla_news_raw.json')")
    print("\n2. Run preprocessing:")
    print("   from preprocessing import main")
    print("   processed_data = main(stock_df, news_df)")
    print("\n3. The preprocessed data is now ready for feature engineering!")