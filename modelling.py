"""
Baseline modelling script for Credit Card Fraud Detection.
Loads preprocessed data, trains an XGBoost classifier, and tracks with MLflow (autolog).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import dagshub
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Membangun-model", "creditcardfraud_preprocessing.csv"
)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    if "is_fraud" not in df.columns:
        raise KeyError("Column 'is_fraud' not found in dataset")
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y


def train_baseline(X: pd.DataFrame, y: pd.Series) -> None:
    dagshub.init(repo_owner="Dekhsa", repo_name="SMSML_Muhamad-Dekhsa-Afnan", mlflow=True)
    mlflow.autolog()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="Baseline_Model"):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        _ = model.predict(X_test)
        print("Baseline model training completed.")


def main() -> None:
    try:
        X, y = load_data(DATA_PATH)
        train_baseline(X, y)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Error during baseline modelling: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
