"""
Inference API for Credit Card Fraud Detection Model.
Serves predictions via Flask and loads model from MLflow using run ID.
Logs metrics for Prometheus exporter.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request

import mlflow
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Global state for metrics (shared with exporter via file or queue)
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")

# MLflow configuration
MLFLOW_RUN_ID = "54050a2976204c79bae1a103b41e6753"  # Hardcoded run ID
MLFLOW_TRACKING_URI = "https://dagshub.com/Dekhsa/SMSML_Muhamad-Dekhsa-Afnan.mlflow"

# Fallback to local model path
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Membangun_model", "model"
)

# In-memory metrics state
metrics_state = {
    "total_predictions": 0,
    "total_fraud_detected": 0,
    "total_errors": 0,
    "prediction_latencies": [],
    "last_confidence_score": 0.0,
    "transaction_amounts": [],
    "active_users": 0,
    "startup_time": time.time(),
}


def load_model():
    """Load the trained XGBoost pipeline model from MLflow or local fallback."""
    global model

    # Try loading from MLflow first
    if MLFLOW_RUN_ID:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            print(f"Loading model from MLflow run: {MLFLOW_RUN_ID}")
            model_uri = f"runs:/{MLFLOW_RUN_ID}/model"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✓ Model loaded from MLflow run {MLFLOW_RUN_ID}")
            return model
        except Exception as e:
            print(
                f"Warning: could not load model from MLflow ({e}). Falling back to local.",
                file=sys.stderr,
            )

    # Fallback to local model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.sklearn.load_model(LOCAL_MODEL_PATH)
            print(f"✓ Model loaded from local path: {LOCAL_MODEL_PATH}")
            return model
        except Exception as e:
            print(f"Error loading local model: {e}", file=sys.stderr)
            raise

    raise FileNotFoundError(
        f"Model not found. Set MLFLOW_RUN_ID or ensure model exists at {LOCAL_MODEL_PATH}"
    )


def save_metrics():
    """Persist metrics to JSON file for exporter to read."""
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics_state, f)
    except Exception as e:
        print(f"Error saving metrics: {e}", file=sys.stderr)


def update_metrics(
    prediction: int,
    confidence: float,
    latency: float,
    amount: float,
    is_error: bool = False,
):
    """Update in-memory metrics."""
    metrics_state["total_predictions"] += 1
    if is_error:
        metrics_state["total_errors"] += 1
    else:
        if prediction == 1:
            metrics_state["total_fraud_detected"] += 1
        metrics_state["prediction_latencies"].append(latency)
        metrics_state["last_confidence_score"] = float(confidence)
        metrics_state["transaction_amounts"].append(float(amount))
    save_metrics()


@app.before_request
def track_active_users():
    """Track concurrent active users."""
    metrics_state["active_users"] = max(
        1, metrics_state.get("active_users", 0)
    )  # Simplified: 1+ active


@app.after_request
def release_active_users(response):
    """Release active user tracking."""
    metrics_state["active_users"] = max(0, metrics_state.get("active_users", 1) - 1)
    return response


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "uptime_seconds": time.time() - metrics_state["startup_time"],
                "total_predictions": metrics_state["total_predictions"],
            }
        ),
        200,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud for a transaction.
    
    Expected JSON payload:
    {
        "amount": 100.5,
        "transaction_hour": 10,
        "foreign_transaction": 0,
        "location_mismatch": 0,
        "device_trust_score": 0.8,
        "velocity_last_24h": 2,
        "cardholder_age": 35,
        "merchant_category_encoded": 5,
        "amount_bin_encoded": 3,
        "age_group_encoded": 2,
        "time_period_encoded": 1
    }
    """
    start_time = time.time()
    try:
        # Validate request
        data = request.get_json()
        if not data:
            raise ValueError("No JSON payload provided")

        # Expected features (from the preprocessing CSV)
        expected_features = [
            "amount",
            "transaction_hour",
            "foreign_transaction",
            "location_mismatch",
            "device_trust_score",
            "velocity_last_24h",
            "cardholder_age",
            "merchant_category_encoded",
            "amount_bin_encoded",
            "age_group_encoded",
            "time_period_encoded",
        ]

        # Validate all features are present
        missing = [f for f in expected_features if f not in data]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Extract features in the correct order
        features = np.array([data[f] for f in expected_features]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))

        latency = time.time() - start_time
        amount = float(data.get("amount", 0))

        # Update metrics
        update_metrics(
            prediction=int(prediction),
            confidence=confidence,
            latency=latency,
            amount=amount,
            is_error=False,
        )

        return (
            jsonify(
                {
                    "prediction": int(prediction),
                    "fraud": bool(prediction),
                    "confidence": confidence,
                    "probability_no_fraud": float(proba[0]),
                    "probability_fraud": float(proba[1]),
                    "latency_seconds": latency,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        latency = time.time() - start_time
        update_metrics(
            prediction=0, confidence=0.0, latency=latency, amount=0.0, is_error=True
        )
        return (
            jsonify(
                {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            400,
        )


@app.route("/metrics/state", methods=["GET"])
def get_metrics_state():
    """Return current metrics state (for debugging)."""
    avg_latency = (
        np.mean(metrics_state["prediction_latencies"])
        if metrics_state["prediction_latencies"]
        else 0.0
    )
    avg_amount = (
        np.mean(metrics_state["transaction_amounts"])
        if metrics_state["transaction_amounts"]
        else 0.0
    )
    uptime = time.time() - metrics_state["startup_time"]

    return jsonify(
        {
            "total_predictions": metrics_state["total_predictions"],
            "total_fraud_detected": metrics_state["total_fraud_detected"],
            "total_errors": metrics_state["total_errors"],
            "avg_prediction_latency_seconds": avg_latency,
            "avg_transaction_amount": avg_amount,
            "last_confidence_score": metrics_state["last_confidence_score"],
            "active_users": metrics_state["active_users"],
            "uptime_seconds": uptime,
        }
    )


if __name__ == "__main__":
    try:
        # Load model on startup
        print("Loading model...")
        print(f"  MLFLOW_RUN_ID: {MLFLOW_RUN_ID}")
        print(f"  MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
        model = load_model()

        # Start Flask app
        print("Starting Flask inference API on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"Error starting inference API: {e}", file=sys.stderr)
        sys.exit(1)
