"""
Advanced modelling script with hyperparameter tuning and manual MLflow logging.
Trains an optimized XGBoost classifier on the credit card fraud dataset.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Membangun-model", "creditcardfraud_preprocessing.csv"
)
TEST_SIZE = 0.2
RANDOM_STATE = 42
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    if "is_fraud" not in df.columns:
        raise KeyError("Column 'is_fraud' not found in dataset")
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y


def plot_confusion_matrix(cm: np.ndarray, labels: Tuple[str, str], path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, path: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR Curve (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def tune_and_log(X: pd.DataFrame, y: pd.Series) -> None:
    dagshub.init(repo_owner="Dekhsa", repo_name="SMSML_Muhamad-Dekhsa-Afnan", mlflow=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "scale_pos_weight": [1.0, 5.0],
    }

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="Optimized_Model_Tuning"):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_proba)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auprc", auprc)

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        pr_path = os.path.join(ARTIFACT_DIR, "precision_recall_curve.png")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=("No Fraud", "Fraud"), path=cm_path)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        plot_pr_curve(prec_curve, rec_curve, auprc, path=pr_path)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pr_path)

        print("Best Params:", grid.best_params_)
        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUPRC: {auprc:.4f}")


def main() -> None:
    try:
        X, y = load_data(DATA_PATH)
        tune_and_log(X, y)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Error during model tuning: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
