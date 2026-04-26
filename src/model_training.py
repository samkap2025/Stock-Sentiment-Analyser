import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.utils import class_weight
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class DataPreparer:
    """
    Prepares data for model training using time-series split.
    """

    def __init__(self, df, train_size=0.8):
        """
        Initialize data preparer.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe with 'target' column
        train_size : float
            Proportion of data for training (default: 0.8)
        """
        self.df = df.copy()
        self.train_size = train_size
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_cols = None

    def prepare(self):
        """
        Prepare data for training using chronological split.

        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scaler)
        """
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        # Step 1: Remove rows with missing target
        print("\nStep 1: Removing missing target values...")
        df = self.df.dropna(subset=['target']).copy()
        removed = len(self.df) - len(df)
        print(f"✓ Removed {removed} rows with missing target")
        print(f"  Rows remaining: {len(df)}")

        # Step 2: Identify feature columns (exclude target)
        print("\nStep 2: Identifying feature columns...")
        self.feature_cols = [col for col in df.columns if col != 'target']
        print(f"✓ Found {len(self.feature_cols)} features")
        print(f"  Features: {self.feature_cols[:5]}..." if len(
            self.feature_cols) > 5 else f"  Features: {self.feature_cols}")

        # Step 3: Extract X and y
        print("\nStep 3: Extracting features and target...")
        X = df[self.feature_cols]
        y = df['target']

        # Check for NaN in features
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠ Warning: Found {nan_count} NaN values in features")
            print("  Filling with forward fill then backward fill...")
            X = X.fillna(method='ffill').fillna(method='bfill')

        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")

        # Step 4: Chronological split (NO RANDOM SHUFFLE)
        print("\nStep 4: Chronological train-test split...")
        split_point = int(len(X) * self.train_size)

        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        print(f"✓ Split at point: {split_point}")
        print(f"  Train size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
        print(f"  Test size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")
        print(f"  Train date range: {X_train.index[0]} to {X_train.index[-1]}")
        print(f"  Test date range: {X_test.index[0]} to {X_test.index[-1]}")

        # Step 5: Standardize features
        print("\nStep 5: Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"✓ Features standardized")
        print(f"  Mean (train): {X_train_scaled.mean():.4f}")
        print(f"  Std (train): {X_train_scaled.std():.4f}")

        # Step 6: Check class balance
        print("\nStep 6: Checking class balance...")
        up_count = (y_train == 1).sum()
        down_count = (y_train == 0).sum()
        total = len(y_train)

        print(f"✓ Training set:")
        print(f"  UP (1): {up_count} ({up_count / total * 100:.1f}%)")
        print(f"  DOWN (0): {down_count} ({down_count / total * 100:.1f}%)")

        up_test = (y_test == 1).sum()
        down_test = (y_test == 0).sum()
        total_test = len(y_test)

        print(f"✓ Test set:")
        print(f"  UP (1): {up_test} ({up_test / total_test * 100:.1f}%)")
        print(f"  DOWN (0): {down_test} ({down_test / total_test * 100:.1f}%)")

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values

        return self.X_train, self.X_test, self.y_train, self.y_test, self.scaler

    def get_class_weights(self):
        """
        Calculate class weights for handling imbalance.

        Returns:
        --------
        dict : class weights {0: weight, 1: weight}
        """
        weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(self.y_train),
            self.y_train
        )
        return dict(enumerate(weights))


class ModelTrainer:
    """
    Trains multiple machine learning models for stock price prediction.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize model trainer.

        Parameters:
        -----------
        X_train : array-like
            Training features
        X_test : array-like
            Test features
        y_train : array-like
            Training target
        y_test : array-like
            Test target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}

    def train_logistic_regression(self):
        """Train logistic regression model."""
        print("\n  Training Logistic Regression...")

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        self.models['lr'] = model
        print("  ✓ Logistic Regression trained")
        return model

    def train_random_forest(self):
        """Train random forest model."""
        print("  Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(self.X_train, self.y_train)

        self.models['rf'] = model
        print("  ✓ Random Forest trained")
        return model

    def train_svm(self):
        """Train Support Vector Machine model."""
        print("  Training SVM...")

        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True,
            verbose=0
        )
        model.fit(self.X_train, self.y_train)

        self.models['svm'] = model
        print("  ✓ SVM trained")
        return model

    def train_xgboost(self):
        """Train XGBoost model."""
        print("  Training XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)

        self.models['xgb'] = model
        print("  ✓ XGBoost trained")
        return model

    def train_knn(self):
        """Train K-Nearest Neighbors model."""
        print("  Training KNN...")

        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        self.models['knn'] = model
        print("  ✓ KNN trained")
        return model

    def train_all(self):
        """Train all models."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        self.train_knn()

        print("\n✓ All models trained")
        return self.models

    def evaluate_model(self, model_name, model=None):
        """
        Evaluate a single model.

        Parameters:
        -----------
        model_name : str
            Name of model ('lr', 'rf', 'svm', 'xgb', 'knn')
        model : sklearn model, optional
            If None, uses model from self.models

        Returns:
        --------
        dict : evaluation metrics
        """
        if model is None:
            model = self.models.get(model_name)

        if model is None:
            print(f"✗ Model {model_name} not found")
            return None

        # Make predictions
        y_pred = model.predict(self.X_test)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(self.X_test)[:, 1]
            auc_roc = roc_auc_score(self.y_test, y_proba)
        else:
            y_proba = None
            auc_roc = None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        self.results[model_name] = metrics
        return metrics

    def evaluate_all(self):
        """Evaluate all trained models."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        for model_name in self.models.keys():
            print(f"\nEvaluating {model_name.upper()}...")
            self.evaluate_model(model_name)

        return self.results

    def print_results(self):
        """Print evaluation results in table format."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        if not self.results:
            print("No results to display. Run evaluate_all() first.")
            return

        # Print table header
        print(f"\n{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
        print("-" * 70)

        # Print results
        for model_name, metrics in self.results.items():
            acc = metrics['accuracy']
            prec = metrics['precision']
            rec = metrics['recall']
            f1 = metrics['f1']
            auc = metrics['auc_roc']

            auc_str = f"{auc:.4f}" if auc is not None else "N/A"

            print(f"{model_name:<10} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {auc_str:<12}")

        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1'])
        print("\n" + "-" * 70)
        print(f"Best Model: {best_model[0].upper()} (F1: {best_model[1]['f1']:.4f})")
        print("=" * 60)

    def get_best_model(self, metric='f1'):
        """
        Get best performing model.

        Parameters:
        -----------
        metric : str
            Metric to use for ranking ('f1', 'accuracy', 'auc_roc')

        Returns:
        --------
        tuple : (model_name, model, metrics)
        """
        if not self.results:
            print("No results. Run evaluate_all() first.")
            return None

        best_name = max(self.results.items(), key=lambda x: x[1].get(metric, 0))[0]
        best_metrics = self.results[best_name]
        best_model = self.models[best_name]

        return best_name, best_model, best_metrics


def main(processed_data, train_size=0.8):
    """
    Main training pipeline.

    Parameters:
    -----------
    processed_data : pd.DataFrame
        Processed data from sentiment_analysis module
    train_size : float
        Proportion of data for training

    Returns:
    --------
    tuple : (trainer, models, results)
    """
    print("\n" + "=" * 60)
    print("STOCK PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Prepare data
    preparer = DataPreparer(processed_data, train_size=train_size)
    X_train, X_test, y_train, y_test, scaler = preparer.prepare()

    # Step 2: Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    models = trainer.train_all()

    # Step 3: Evaluate models
    results = trainer.evaluate_all()

    # Step 4: Print results
    trainer.print_results()

    # Step 5: Get best model
    best_name, best_model, best_metrics = trainer.get_best_model(metric='f1')
    print(f"\nBest Model: {best_name.upper()}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    if best_metrics['auc_roc']:
        print(f"  AUC-ROC: {best_metrics['auc_roc']:.4f}")

    return trainer, models, results


if __name__ == "__main__":
    print("Model Training Module")
    print("Usage:")
    print("  from model_training import main")
    print("  trainer, models, results = main(processed_data)")