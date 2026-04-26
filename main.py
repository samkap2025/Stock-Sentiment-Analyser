"""
TESLA Stock Price Predictor - Main Pipeline
Complete workflow from data loading to predictions
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocessing import StockDataPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.model_training import DataPreparer, ModelTrainer
from src.ensemble_prediction import ensemble_weighted_average
from src.trading_signals import generate_trading_signals, add_confidence_scores
from src.backtesting import simple_backtest, calculate_backtest_metrics
from src.visualisation import (
    plot_price_trends, plot_sentiment_trends,
    plot_trading_signals, plot_backtest_results
)
from src.model_persistence import save_models, save_scaler


def main():
    """
    Complete stock price prediction pipeline for TESLA
    """
    print("\n" + "=" * 80)
    print("TESLA STOCK PRICE PREDICTOR - COMPLETE PIPELINE")
    print("=" * 80)

    # ========== STEP 1: LOAD RAW DATA ==========
    print("\n\n" + "=" * 80)
    print("STEP 1: LOADING RAW DATA FROM DATA FOLDER")
    print("=" * 80)

    try:
        # Load stock data
        print("\nLoading stock data from: data/tsla_stock_raw.csv")
        stock_df = pd.read_csv('data/tsla_stock_raw.csv', index_col=0)
        stock_df.index = pd.to_datetime(stock_df.index)
        print(f"✓ Stock data loaded: {len(stock_df)} rows")

        # Load news data
        print("\nLoading news data from: data/tsla_news_raw.json")
        news_df = pd.read_json('data/tsla_news_raw.json')
        print(f"✓ News data loaded: {len(news_df)} articles")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Make sure data/tsla_stock_raw.csv and data/tsla_news_raw.json exist")
        return

    # ========== STEP 2: PREPROCESS DATA ==========
    print("\n\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)

    preprocessor = StockDataPreprocessor(ticker="TSLA")
    preprocessor.load_stock_data(stock_df)
    preprocessor.load_news_data(news_df)
    preprocessor.clean_stock_data()
    preprocessor.clean_news_data()
    preprocessor.create_target_variable()
    preprocessor.align_stock_and_news()
    preprocessor.handle_missing_values(method='forward_fill')
    processed_df = preprocessor.combined_df

    print(f"\n✓ Preprocessing complete: {len(processed_df)} rows, {len(processed_df.columns)} columns")

    # ========== STEP 3: SENTIMENT ANALYSIS ==========
    print("\n\n" + "=" * 80)
    print("STEP 3: SENTIMENT ANALYSIS")
    print("=" * 80)

    sentiment_analyzer = SentimentAnalyzer(method='vader')
    news_analyzed = sentiment_analyzer.analyze_news_dataframe(preprocessor.news_df)
    sentiment_analyzer.get_sentiment_summary(news_analyzed)

    print(f"\n✓ Sentiment analysis complete")

    # ========== STEP 4: FEATURE ENGINEERING ==========
    print("\n\n" + "=" * 80)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 80)

    feature_engineer = FeatureEngineer(processed_df)
    final_df = feature_engineer.create_all_features()

    print(f"\n✓ Feature engineering complete: {len(final_df.columns)} features")

    # ========== STEP 5: DATA PREPARATION FOR TRAINING ==========
    print("\n\n" + "=" * 80)
    print("STEP 5: TRAIN-TEST SPLIT")
    print("=" * 80)

    preparer = DataPreparer(final_df, train_size=0.8)
    X_train, X_test, y_train, y_test, scaler = preparer.prepare()

    # ========== STEP 6: MODEL TRAINING ==========
    print("\n\n" + "=" * 80)
    print("STEP 6: MODEL TRAINING")
    print("=" * 80)

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    models = trainer.train_all()

    # ========== STEP 7: MODEL EVALUATION ==========
    print("\n\n" + "=" * 80)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 80)

    results = trainer.evaluate_all()
    trainer.print_results()

    best_name, best_model, best_metrics = trainer.get_best_model(metric='f1')

    # ========== STEP 8: ENSEMBLE PREDICTIONS ==========
    print("\n\n" + "=" * 80)
    print("STEP 8: ENSEMBLE PREDICTIONS")
    print("=" * 80)

    print("\nGenerating ensemble predictions...")
    ensemble_pred = ensemble_weighted_average(models, X_test)
    confidence = add_confidence_scores(models, X_test)
    print(f"✓ Ensemble predictions generated: {len(ensemble_pred)} predictions")

    # ========== STEP 9: TRADING SIGNALS ==========
    print("\n\n" + "=" * 80)
    print("STEP 9: TRADING SIGNAL GENERATION")
    print("=" * 80)

    # Get test data for signals
    test_df = final_df.iloc[-len(X_test):].copy()
    test_df['ensemble_prediction'] = ensemble_pred
    test_df['confidence'] = confidence

    # Generate signals
    sentiment_scores = test_df['sentiment_score'].values
    test_df = generate_trading_signals(test_df, ensemble_pred, sentiment_scores)

    print(f"\n✓ Trading signals generated")
    signal_counts = test_df['signal'].value_counts()
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count}")

    # ========== STEP 10: BACKTESTING ==========
    print("\n\n" + "=" * 80)
    print("STEP 10: BACKTESTING")
    print("=" * 80)

    backtest_df = simple_backtest(test_df.copy())
    metrics = calculate_backtest_metrics(backtest_df)

    print(f"\n✓ Backtesting complete")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # ========== STEP 11: VISUALIZATIONS ==========
    print("\n\n" + "=" * 80)
    print("STEP 11: CREATING VISUALIZATIONS")
    print("=" * 80)

    os.makedirs('outputs', exist_ok=True)

    print("\nGenerating plots...")
    try:
        plot_price_trends(final_df)
        print("  ✓ Price trends plot saved")
    except Exception as e:
        print(f"  ✗ Price trends plot failed: {e}")

    try:
        plot_sentiment_trends(final_df)
        print("  ✓ Sentiment trends plot saved")
    except Exception as e:
        print(f"  ✗ Sentiment trends plot failed: {e}")

    try:
        plot_trading_signals(backtest_df)
        print("  ✓ Trading signals plot saved")
    except Exception as e:
        print(f"  ✗ Trading signals plot failed: {e}")

    try:
        plot_backtest_results(backtest_df)
        print("  ✓ Backtest results plot saved")
    except Exception as e:
        print(f"  ✗ Backtest results plot failed: {e}")

    # ========== STEP 12: SAVE MODELS ==========
    print("\n\n" + "=" * 80)
    print("STEP 12: SAVING MODELS")
    print("=" * 80)

    os.makedirs('models', exist_ok=True)

    print("\nSaving trained models...")
    save_models(models, path='models/')
    save_scaler(scaler, path='models/scaler.pkl')
    print("✓ Models saved to models/ folder")

    # ========== SUMMARY ==========
    print("\n\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"""

    ✓ Stock Data: {len(final_df)} trading days
    ✓ News Articles: {len(news_analyzed)} articles analyzed
    ✓ Features: {len(final_df.columns)} features created
    ✓ Models Trained: {len(models)} models
    ✓ Best Model: {best_name.upper()}
    ✓ Trading Signals: {len(test_df)} signals generated

    Results:
    - Strategy Return: {metrics['total_return_%']:.2f}%
    - Annual Return: {metrics['annual_return_%']:.2f}%
    - Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
    - Max Drawdown: {metrics['max_drawdown_%']:.2f}%
    - Win Rate: {metrics['win_rate_%']:.2f}%

    Output Files:
    - Plots: outputs/
    - Models: models/
    - Data: (in memory)

    Next Steps:
    1. Check plots in outputs/
    2. Use models/ for production predictions
    3. Run main.py again for updated predictions

    """)

    return {
        'models': models,
        'scaler': scaler,
        'results': results,
        'best_model': (best_name, best_model),
        'backtest_metrics': metrics,
        'test_df': backtest_df
    }


if __name__ == "__main__":
    pipeline_results = main()