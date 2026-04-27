import pandas as pd
import numpy as np
import nltk
import ssl
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        pass

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        print("⚠ Warning: VADER lexicon download failed. Will use alternative method.")


class SentimentAnalyzer:
    def __init__(self, method='vader'):
        self.method = method.lower()
        self.sia = None

        if self.method == 'vader':
            try:
                self.sia = SentimentIntensityAnalyzer()
                print("✓ VADER sentiment analyzer initialized")
            except Exception as e:
                print(f"⚠ Warning: VADER initialization failed: {str(e)}")
                print("  Falling back to TextBlob")
                self.method = 'textblob'
                print("✓ TextBlob sentiment analyzer initialized")
        elif self.method == 'textblob':
            print("✓ TextBlob sentiment analyzer initialized")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'vader' or 'textblob'")

    def analyze_sentiment_vader(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 'NEUTRAL', 0.0

        try:
            if self.sia is None:
                # Fallback to TextBlob if VADER not available
                return self.analyze_sentiment_textblob(text)

            scores = self.sia.polarity_scores(text)
            compound = scores['compound']  # -1 to 1

            if compound >= 0.05:
                return 'POSITIVE', compound
            elif compound <= -0.05:
                return 'NEGATIVE', compound
            else:
                return 'NEUTRAL', compound

        except Exception as e:
            # Fallback to TextBlob
            try:
                return self.analyze_sentiment_textblob(text)
            except:
                return 'NEUTRAL', 0.0

    def analyze_sentiment_textblob(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 'NEUTRAL', 0.0

        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            if polarity > 0.1:
                return 'POSITIVE', polarity
            elif polarity < -0.1:
                return 'NEGATIVE', polarity
            else:
                return 'NEUTRAL', polarity

        except Exception as e:
            print(f"Error in TextBlob analysis: {str(e)}")
            return 'NEUTRAL', 0.0

    def analyze_sentiment(self, text):
        if self.method == 'vader':
            return self.analyze_sentiment_vader(text)
        elif self.method == 'textblob':
            return self.analyze_sentiment_textblob(text)

    def analyze_news_dataframe(self, news_df, headline_column='headline'):
        print(f"\nAnalyzing sentiment for {len(news_df)} articles...")

        df = news_df.copy()

        # Use Alpha Vantage sentiment already extracted in preprocessing
        if 'sentiment' in df.columns and 'sentiment_score' in df.columns:
            print("✓ Using Alpha Vantage sentiment scores")
            return df

        sentiments = []
        scores = []

        for idx, row in df.iterrows():
            text = row.get(headline_column, '')
            sentiment, score = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            scores.append(score)

            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  - Processed {idx + 1} articles...")

        df['sentiment'] = sentiments
        df['sentiment_score'] = scores

        print(f"✓ Sentiment analysis complete")
        print(f"  - POSITIVE: {(np.array(sentiments) == 'POSITIVE').sum()}")
        print(f"  - NEGATIVE: {(np.array(sentiments) == 'NEGATIVE').sum()}")
        print(f"  - NEUTRAL: {(np.array(sentiments) == 'NEUTRAL').sum()}")

        return df

    def aggregate_daily_sentiment(self, news_df, stock_df):
        print("\nAggregating daily sentiment...")

        df = stock_df.copy()
        news = news_df.copy()

        # Ensure date columns are in the right format
        news['date'] = pd.to_datetime(news['date']).dt.date
        df['date'] = df.index.date

        # Group news by date
        daily_sentiment = news.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'sentiment': 'count'
        }).reset_index()

        # Flatten column names
        daily_sentiment.columns = ['date', 'sentiment_score', 'sentiment_std',
                                   'sentiment_min', 'sentiment_max', 'article_count']

        # Fill NaN sentiment_std with 0
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)

        # Count sentiments by type
        sentiment_counts = news.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

        # Merge sentiment counts
        daily_sentiment = daily_sentiment.merge(
            sentiment_counts.reset_index(),
            on='date',
            how='left'
        )

        # Merge with stock data
        df = df.merge(
            daily_sentiment,
            on='date',
            how='left'
        )

        # Fill missing sentiment scores with 0
        df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        df['sentiment_std'] = df['sentiment_std'].fillna(0.0)
        df['sentiment_min'] = df['sentiment_min'].fillna(0.0)
        df['sentiment_max'] = df['sentiment_max'].fillna(0.0)
        df['article_count'] = df['article_count'].fillna(0)

        # Fill missing sentiment counts with 0
        for col in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Drop temporary date column
        df = df.drop('date', axis=1)

        print(f"✓ Daily sentiment aggregated")
        print(f"  - Days with news: {(df['article_count'] > 0).sum()}")
        print(f"  - Days without news: {(df['article_count'] == 0).sum()}")
        print(f"  - Avg sentiment score: {df['sentiment_score'].mean():.4f}")

        return df

    def add_sentiment_features(self, df):
        print("\nCreating sentiment features...")

        df_feat = df.copy()

        # Sentiment moving averages
        df_feat['sentiment_ma_5'] = df_feat['sentiment_score'].rolling(window=5, min_periods=1).mean()
        df_feat['sentiment_ma_10'] = df_feat['sentiment_score'].rolling(window=10, min_periods=1).mean()
        df_feat['sentiment_ma_20'] = df_feat['sentiment_score'].rolling(window=20, min_periods=1).mean()

        # Sentiment change
        df_feat['sentiment_change'] = df_feat['sentiment_score'].diff()
        df_feat['sentiment_change_ma'] = df_feat['sentiment_change'].rolling(window=5, min_periods=1).mean()

        # Sentiment volatility (std dev)
        df_feat['sentiment_volatility'] = df_feat['sentiment_score'].rolling(window=5, min_periods=1).std()

        # Positive/negative trend
        df_feat['positive_momentum'] = (df_feat['sentiment_score'] > 0).astype(int).rolling(window=5).mean()
        df_feat['negative_momentum'] = (df_feat['sentiment_score'] < 0).astype(int).rolling(window=5).mean()

        # Sentiment trend (increasing or decreasing)
        df_feat['sentiment_trend'] = np.sign(df_feat['sentiment_ma_5'] - df_feat['sentiment_ma_10'])

        # Days since positive news
        positive_mask = df_feat['sentiment_score'] > 0
        df_feat['days_since_positive'] = (~positive_mask).cumsum() - (~positive_mask).cumsum()[positive_mask].max()
        df_feat['days_since_positive'] = df_feat['days_since_positive'].fillna(method='ffill').fillna(0)

        # Days since negative news
        negative_mask = df_feat['sentiment_score'] < 0
        df_feat['days_since_negative'] = (~negative_mask).cumsum() - (~negative_mask).cumsum()[negative_mask].max()
        df_feat['days_since_negative'] = df_feat['days_since_negative'].fillna(method='ffill').fillna(0)

        # Article count features
        df_feat['article_count_ma'] = df_feat['article_count'].rolling(window=5, min_periods=1).mean()
        df_feat['high_news_day'] = (df_feat['article_count'] > df_feat['article_count'].quantile(0.75)).astype(int)

        # Fill NaN values
        df_feat = df_feat.fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"✓ Sentiment features created")
        print(f"  - New features: {len(df_feat.columns) - len(df.columns)}")

        return df_feat

    def get_sentiment_summary(self, news_df):
        print("\n" + "=" * 60)
        print("SENTIMENT SUMMARY")
        print("=" * 60)

        print("\nSentiment Distribution:")
        sentiment_counts = news_df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(news_df)) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")

        print("\nSentiment Score Statistics:")
        scores = news_df['sentiment_score']
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
        print(f"  Median: {scores.median():.4f}")

        print("\n" + "=" * 60 + "\n")


def main(news_df, stock_df, method='vader'):
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)

    # Initialize analyzer
    analyzer = SentimentAnalyzer(method=method)

    # Analyze sentiment for all headlines
    print("\nStep 1: Analyzing sentiment for news headlines...")
    news_analyzed = analyzer.analyze_news_dataframe(news_df)

    # Show sentiment summary
    analyzer.get_sentiment_summary(news_analyzed)

    # Aggregate daily sentiment
    print("Step 2: Aggregating sentiment by date...")
    stock_with_sentiment = analyzer.aggregate_daily_sentiment(news_analyzed, stock_df)

    # Create sentiment features
    print("Step 3: Creating sentiment features...")
    stock_with_features = analyzer.add_sentiment_features(stock_with_sentiment)

    print("\n✓ Sentiment analysis complete!")
    print(f"  - Input: {len(news_df)} articles")
    print(f"  - Output: {len(stock_with_features)} trading days")
    print(f"  - Features added: {len(stock_with_features.columns) - len(stock_df.columns)}")

    return stock_with_features


if __name__ == "__main__":
    print("Sentiment Analysis Module")
    print("Usage:")
    print("  from sentiment_analysis import main")
    print("  processed_data = main(news_df, stock_df, method='vader')")