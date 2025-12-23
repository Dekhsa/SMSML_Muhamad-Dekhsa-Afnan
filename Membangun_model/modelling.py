"""
Baseline modelling script for Credit Card Fraud Detection.
Loads preprocessed data, trains an XGBoost classifier, and tracks with MLflow (autolog).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tempfile
import shutil

import dagshub
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "creditcardfraud_preprocessing.csv"
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
        # Use a simple pipeline to avoid data leakage on scaling
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Compute metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_proba)

        # Log metrics explicitly (autolog won't record test metrics by default)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("auprc", auprc)

        # Create and log artifacts (confusion matrix and PR curve)
        artifact_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        cm_path = os.path.join(artifact_dir, "baseline_confusion_matrix.png")
        pr_path = os.path.join(artifact_dir, "baseline_precision_recall_curve.png")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=("No Fraud", "Fraud"), yticklabels=("No Fraud", "Fraud"))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Baseline Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_curve, prec_curve, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Baseline Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150)
        plt.close()

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pr_path)

        # Log the trained pipeline as an MLflow model (ensures MLmodel + model.pkl saved)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Save to a temporary directory to avoid collisions across runs, then upload folder
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_dir = os.path.join(tmpdir, "model")
            mlflow.sklearn.save_model(pipeline, path=local_model_dir)
            mlflow.log_artifacts(local_model_dir, artifact_path="model")

        # Save a stable local copy for inference service
        stable_model_dir = os.path.join(os.path.dirname(__file__), "model")
        try:
            if os.path.exists(stable_model_dir):
                shutil.rmtree(stable_model_dir)
            mlflow.sklearn.save_model(pipeline, path=stable_model_dir)
            print(f"Saved local model copy for inference at: {stable_model_dir}")
        except Exception as _e:
            print(f"Warning: could not save stable local model copy: {_e}")

        print(
            f"Baseline metrics — F1: {f1:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, AUPRC: {auprc:.4f}"
        )


def main() -> None:
    try:
        X, y = load_data(DATA_PATH)
        train_baseline(X, y)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"Error during baseline modelling: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
