def generate_trading_signals(df, predictions, sentiment_scores):
    df['prediction'] = predictions  # 1 = UP, 0 = DOWN
    df['signal'] = 'HOLD'

    # Buy: Prediction UP + Positive sentiment
    buy_condition = (df['prediction'] == 1) & (sentiment_scores > 0.05)
    df.loc[buy_condition, 'signal'] = 'BUY'

    # Sell: Prediction DOWN + Negative sentiment
    sell_condition = (df['prediction'] == 0) & (sentiment_scores < -0.05)
    df.loc[sell_condition, 'signal'] = 'SELL'

    # Strong signals for extreme conditions
    strong_buy = (df['prediction'] == 1) & (sentiment_scores > 0.3)
    df.loc[strong_buy, 'signal'] = 'STRONG_BUY'

    strong_sell = (df['prediction'] == 0) & (sentiment_scores < -0.3)
    df.loc[strong_sell, 'signal'] = 'STRONG_SELL'

    return df


def add_confidence_scores(models, X_test):
    confidence_scores = []

    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            # For models without probability
            proba = np.abs(model.decision_function(X_test))
        confidence_scores.append(proba)

    # Average confidence across models
    avg_confidence = np.mean(confidence_scores, axis=0)
    return avg_confidence