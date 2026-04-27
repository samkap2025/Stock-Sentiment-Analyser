import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class DataPreparer:
    def __init__(self, df, train_size=0.8):
        self.df = df.copy()
        self.train_size = train_size
        self.scaler = None
        self.feature_cols = None

    def prepare(self):
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        print("\nStep 1: Removing missing target values...")
        df = self.df.dropna(subset=['target']).copy()
        removed = len(self.df) - len(df)

        print(f"✓ Removed {removed} rows")
        print(f"Rows remaining: {len(df)}")

        print("\nStep 2: Identifying numeric feature columns...")

        exclude_cols = ['target', 'price']

        self.feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        print(f"✓ Found {len(self.feature_cols)} numeric features")
        print(self.feature_cols)

        print("\nStep 3: Extracting X and y...")
        X = df[self.feature_cols].copy()
        y = df['target'].copy()

        X = X.ffill().bfill()

        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")

        print("\nStep 4: Chronological split...")
        split_point = int(len(X) * self.train_size)

        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")

        print("\nStep 5: Standardizing...")
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("✓ Features standardized")

        return (
            X_train_scaled,
            X_test_scaled,
            y_train.values,
            y_test.values,
            self.scaler
        )


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}

    def train_logistic_regression(self):
        print("Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['lr'] = model

    def train_random_forest(self):
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['rf'] = model

    def train_svm(self):
        print("Training SVM...")
        model = SVC(probability=True, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['svm'] = model

    def train_xgboost(self):
        print("Training XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models['xgb'] = model

    def train_knn(self):
        print("Training KNN...")
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        self.models['knn'] = model

    def train_all(self):
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        self.train_knn()

        print("✓ All models trained")
        return self.models

    def evaluate_model(self, model_name):
        model = self.models[model_name]

        y_pred = model.predict(self.X_test)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_prob)
        else:
            auc = None

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc_roc': auc
        }

        self.results[model_name] = metrics
        return metrics

    def evaluate_all(self):
        for name in self.models:
            self.evaluate_model(name)
        return self.results

    def print_results(self):
        print("\nRESULTS")
        for model, metrics in self.results.items():
            print(model.upper(), metrics)

    def get_best_model(self, metric='f1'):
        best_name = max(
            self.results,
            key=lambda x: self.results[x][metric]
        )
        return best_name, self.models[best_name], self.results[best_name]