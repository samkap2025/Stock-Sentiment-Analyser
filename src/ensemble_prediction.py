def ensemble_majority_vote(models, X_test):
    predictions = []

    for model_name, model in models.items():
        pred = model.predict(X_test)
        predictions.append(pred)

    # Stack predictions and get majority vote
    predictions = np.array(predictions)
    ensemble_pred = (predictions.mean(axis=0) > 0.5).astype(int)

    return ensemble_pred


def ensemble_weighted_average(models, X_test, weights=None):
    if weights is None:
        # Equal weights for all models
        weights = {k: 1.0 for k in models.keys()}

    total_weight = sum(weights.values())
    weighted_pred = np.zeros(X_test.shape[0])

    for model_name, model in models.items():
        pred_proba = model.predict_proba(X_test)[:, 1]
        weighted_pred += (weights[model_name] / total_weight) * pred_proba

    ensemble_pred = (weighted_pred > 0.5).astype(int)
    return ensemble_pred