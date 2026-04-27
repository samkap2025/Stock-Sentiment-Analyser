import numpy as np


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
        try:
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba, use predictions
                pred_proba = model.predict(X_test).astype(float)

            weighted_pred += (weights[model_name] / total_weight) * pred_proba
        except Exception as e:
            print(f"Warning: Could not get predictions from {model_name}: {e}")
            continue

    ensemble_pred = (weighted_pred > 0.5).astype(int)
    return ensemble_pred


def ensemble_stacking(models, X_test, meta_model=None):
    meta_features = []

    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_test)
        else:
            pred = model.predict(X_test).reshape(-1, 1)

        meta_features.append(pred)

    # Stack meta features
    X_meta = np.hstack(meta_features)

    # Use weighted average if no meta model provided
    if meta_model is None:
        ensemble_pred = (X_meta.mean(axis=1) > 0.5).astype(int)
    else:
        ensemble_pred = meta_model.predict(X_meta)

    return ensemble_pred