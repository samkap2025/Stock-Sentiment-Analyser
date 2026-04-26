from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def evaluate_all_models(models, X_test, y_test):
    results = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }

    return results

def perform_cross_validation(model, X_train, y_train, cv=5):
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f'Cross-val scores: {scores}')
    print(f'Mean score: {scores.mean():.4f} (+/- {scores.std():.4f})')


def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 15]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5
    )
    grid_search.fit(X_train, y_train)

    print(f'Best params: {grid_search.best_params_}')
    return grid_search.best_estimator_