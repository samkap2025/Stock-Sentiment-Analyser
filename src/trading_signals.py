import numpy as np


def generate_trading_signals(df, predictions, sentiment_scores):
    """
    Generate trading signals based on predictions and sentiment

    Parameters:
    -----------
    df : pd.DataFrame
        Test dataframe
    predictions : array
        Binary predictions (1=UP, 0=DOWN)
    sentiment_scores : array
        Sentiment scores (-1 to 1)

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'signal' column
    """
    df_signals = df.copy()
    df_signals['prediction'] = predictions
    df_signals['signal'] = 'HOLD'

    # Ensure sentiment_scores is numpy array
    if not isinstance(sentiment_scores, np.ndarray):
        sentiment_scores = np.array(sentiment_scores)

    # Buy: Prediction UP + Positive sentiment
    buy_condition = (predictions == 1) & (sentiment_scores > 0.05)
    df_signals.loc[buy_condition, 'signal'] = 'BUY'

    # Sell: Prediction DOWN + Negative sentiment
    sell_condition = (predictions == 0) & (sentiment_scores < -0.05)
    df_signals.loc[sell_condition, 'signal'] = 'SELL'

    # Strong signals for extreme conditions
    strong_buy = (predictions == 1) & (sentiment_scores > 0.3)
    df_signals.loc[strong_buy, 'signal'] = 'STRONG_BUY'

    strong_sell = (predictions == 0) & (sentiment_scores < -0.3)
    df_signals.loc[strong_sell, 'signal'] = 'STRONG_SELL'

    return df_signals


def add_confidence_scores(models, X_test):
    """
    Calculate confidence scores across all models

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array
        Test features

    Returns:
    --------
    array
        Average confidence scores
    """
    confidence_scores = []

    for model_name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without probability (SVM with decision_function)
                try:
                    proba = np.abs(model.decision_function(X_test))
                    # Normalize to 0-1 range
                    proba = (proba - proba.min()) / (proba.max() - proba.min())
                except:
                    proba = np.abs(model.predict(X_test))

            confidence_scores.append(proba)
        except Exception as e:
            print(f"Warning: Could not get probabilities for {model_name}: {e}")
            continue

    if confidence_scores:
        # Average confidence across models
        avg_confidence = np.mean(confidence_scores, axis=0)
    else:
        avg_confidence = np.zeros(len(X_test))

    return avg_confidence