import pandas as pd
import numpy as np
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing import StockDataPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.model_training import DataPreparer, ModelTrainer
from src.ensemble_prediction import ensemble_weighted_average
from src.trading_signals import generate_trading_signals, add_confidence_scores
from src.backtesting import simple_backtest, calculate_backtest_metrics
from src.visualisation import (
    plot_price_trends,
    plot_sentiment_trends,
    plot_trading_signals,
    plot_backtest_results,
    plot_model_comparison
)
from src.model_persistence import save_models, save_scaler


def main():
    print("\n" + "=" * 80)
    print("TESLA STOCK PRICE PREDICTOR - COMPLETE PIPELINE")
    print("=" * 80)

    stock_df = pd.read_csv("data/tsla_stock_raw.csv")
    news_df = pd.read_json("data/tsla_news_raw.json")

    # PREPROCESS
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)

    preprocessor = StockDataPreprocessor(ticker="TSLA")
    preprocessor.load_stock_data(stock_df)
    preprocessor.load_news_data(news_df)
    preprocessor.clean_stock_data()
    preprocessor.clean_news_data()
    preprocessor.create_target_variable()
    preprocessor.align_stock_and_news()
    preprocessor.handle_missing_values("forward_fill")

    processed_df = preprocessor.combined_df
    print(f"\n✓ Preprocessing complete: {len(processed_df)} rows")

    # SENTIMENT
    print("\n" + "=" * 80)
    print("STEP 3: SENTIMENT ANALYSIS")
    print("=" * 80)

    sentiment_analyzer = SentimentAnalyzer(method="vader")
    news_analyzed = sentiment_analyzer.analyze_news_dataframe(preprocessor.news_df)
    sentiment_analyzer.get_sentiment_summary(news_analyzed)

    # FEATURES
    print("\n" + "=" * 80)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 80)

    feature_engineer = FeatureEngineer(processed_df)
    final_df = feature_engineer.create_all_features()

    # TRAIN TEST
    print("\n" + "=" * 80)
    print("STEP 5: TRAIN TEST SPLIT")
    print("=" * 80)

    preparer = DataPreparer(final_df)
    X_train, X_test, y_train, y_test, scaler = preparer.prepare()

    # TRAIN
    print("\n" + "=" * 80)
    print("STEP 6: MODEL TRAINING")
    print("=" * 80)

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    models = trainer.train_all()

    # EVAL
    print("\n" + "=" * 80)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 80)

    results = trainer.evaluate_all()
    trainer.print_results()

    best_name, best_model, _ = trainer.get_best_model()

    # ENSEMBLE
    ensemble_pred = ensemble_weighted_average(models, X_test)
    confidence = add_confidence_scores(models, X_test)

    # SIGNALS
    print("\n" + "=" * 80)
    print("STEP 8: SIGNALS")
    print("=" * 80)

    test_df = final_df.iloc[-len(X_test):].copy()
    test_df["confidence"] = confidence

    sentiment_scores = test_df["sentiment_score"].values
    test_df = generate_trading_signals(
        test_df,
        ensemble_pred,
        sentiment_scores
    )

    print(test_df["signal"].value_counts())

    # BACKTEST
    print("\n" + "=" * 80)
    print("STEP 9: BACKTEST")
    print("=" * 80)

    backtest_df = simple_backtest(test_df)
    metrics = calculate_backtest_metrics(backtest_df)

    for k, v in metrics.items():
        print(k, ":", round(v, 4))

    # PLOTS
    print("\n" + "=" * 80)
    print("STEP 10: PLOTS")
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)

    evaluation_results = results

    plot_price_trends(final_df)
    plot_sentiment_trends(final_df)
    plot_trading_signals(backtest_df)
    plot_backtest_results(backtest_df)
    plot_model_comparison(evaluation_results)

    print("✓ Plots saved in outputs/")

    # auto-open on Mac
    try:
        subprocess.run(["open", "outputs"])
        print("✓ Opened outputs folder")
    except Exception as e:
        print("Could not auto-open:", e)

    # SAVE
    os.makedirs("models", exist_ok=True)
    save_models(models, "models/")
    save_scaler(scaler, "models/scaler.pkl")

    print("\n✓ COMPLETE")
    print("Best model:", best_name.upper())


if __name__ == "__main__":
    main()