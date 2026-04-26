import numpy as np


def generate_trading_signals(df, predictions, sentiment_scores):
    """
    Generate balanced trading signals
    """
    df_signals = df.copy()
    df_signals["prediction"] = predictions
    df_signals["signal"] = "HOLD"

    if not isinstance(sentiment_scores, np.ndarray):
        sentiment_scores = np.array(sentiment_scores)

    # normalize sentiment
    sentiment_scores = np.nan_to_num(sentiment_scores)

    # BUY logic
    buy = (
        (predictions == 1) &
        (sentiment_scores >= -0.05)
    )

    # SELL logic
    sell = (
        (predictions == 0) &
        (sentiment_scores <= 0.05)
    )

    df_signals.loc[buy, "signal"] = "BUY"
    df_signals.loc[sell, "signal"] = "SELL"

    # strong buy
    strong_buy = (
        (predictions == 1) &
        (sentiment_scores > 0.20)
    )

    # strong sell
    strong_sell = (
        (predictions == 0) &
        (sentiment_scores < -0.20)
    )

    df_signals.loc[strong_buy, "signal"] = "STRONG_BUY"
    df_signals.loc[strong_sell, "signal"] = "STRONG_SELL"

    # uncertain cases
    uncertain = np.abs(sentiment_scores) < 0.03
    df_signals.loc[uncertain, "signal"] = "HOLD"

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