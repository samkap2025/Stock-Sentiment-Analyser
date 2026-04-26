import numpy as np


def generate_trading_signals(df, predictions, sentiment_scores):
    """
    Generate trading signals using model prediction + sentiment score
    """
    df_signals = df.copy()
    df_signals["prediction"] = predictions
    df_signals["signal"] = "HOLD"

    # convert to numpy
    if not isinstance(sentiment_scores, np.ndarray):
        sentiment_scores = np.array(sentiment_scores)

    # BUY / SELL
    buy_condition = (predictions == 1) & (sentiment_scores > 0.0)
    sell_condition = (predictions == 0) & (sentiment_scores < 0.0)

    df_signals.loc[buy_condition, "signal"] = "BUY"
    df_signals.loc[sell_condition, "signal"] = "SELL"

    # Strong signals
    strong_buy = (predictions == 1) & (sentiment_scores > 0.25)
    strong_sell = (predictions == 0) & (sentiment_scores < -0.25)

    df_signals.loc[strong_buy, "signal"] = "STRONG_BUY"
    df_signals.loc[strong_sell, "signal"] = "STRONG_SELL"

    return df_signals


def add_confidence_scores(models, X_test):
    """
    Calculate average confidence across all models
    """
    confidence_scores = []

    for model_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]

            elif hasattr(model, "decision_function"):
                proba = np.abs(model.decision_function(X_test))
                rng = proba.max() - proba.min()
                if rng != 0:
                    proba = (proba - proba.min()) / rng

            else:
                proba = np.abs(model.predict(X_test))

            confidence_scores.append(proba)

        except Exception as e:
            print(f"Warning: Could not get confidence for {model_name}: {e}")

    if confidence_scores:
        avg_confidence = np.mean(confidence_scores, axis=0)
    else:
        avg_confidence = np.zeros(len(X_test))

    return avg_confidence