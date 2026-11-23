"""
train.py — MLflow experiment tracking and model registry for MP126.

Usage (from project root):
    python -m mp126.train

What it does:
  1. Runs the cleaning pipeline on train.csv (read-only original).
  2. Encodes categoricals and builds feature/target arrays.
  3. Runs 3+ training experiments with different model settings,
     logging every run to MLflow (SQLite backend: mlflow.db).
  4. In every run logs:
       - param  ``data_sha256``  (SHA-256 of original train.csv)
       - metric ``val_mae``      (MAE on a held-out validation split)
  5. Picks the best run (lowest val_mae), registers the model as
     version 1 and sets the alias ``champion``.

Handbook requirements met:
  - sqlite:///mlflow.db  (committed to git)
  - 3+ runs
  - data_sha256 param logged
  - mae metric logged
  - registered model with @champion alias

Author: G224 — Parv (S24CSEU1347)
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Allow running as python -m mp126.train from project root
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from mp126.config import get_params, project_root
from mp126.clean import run_cleaning_pipeline


# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "mp126-plates-sold"
EXPERIMENT_NAME = "mp126-canteen-rush"
FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
    "day_type",
    "block",
]
TARGET = "plates_sold"
CAT_COLS = ["day_type", "block"]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_features(df: pd.DataFrame) -> tuple[np.ndarray, OrdinalEncoder]:
    """Ordinal-encode categorical columns; return feature matrix and fitted encoder."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df = df.copy()
    df[CAT_COLS] = enc.fit_transform(df[CAT_COLS])
    return df[FEATURES].values, enc


def encode_features_with(df: pd.DataFrame, enc: OrdinalEncoder) -> np.ndarray:
    """Apply a fitted encoder to new data."""
    df = df.copy()
    df[CAT_COLS] = enc.transform(df[CAT_COLS])
    return df[FEATURES].values


def run_experiments(params: dict) -> None:
    """Train 3 models, log every run, register the best as @champion."""
    root = project_root()
    train_csv_path = root / params["paths"]["train_csv"]
    data_sha256 = _sha256(train_csv_path)
    random_state = params.get("model", {}).get("random_state", 42)

    # Set MLflow tracking to SQLite in project root
    mlflow_db = root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Clean the data ---
    df = run_cleaning_pipeline(params)
    X_all, encoder = encode_features(df)
    y_all = df[TARGET].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=params.get("model", {}).get("test_size", 0.2),
        random_state=random_state,
    )

    # --- Define 3+ experiments ---
    experiments = [
        {
            "name": "ridge_baseline",
            "model": Ridge(alpha=1.0),
            "params": {"model_type": "Ridge", "alpha": 1.0},
        },
        {
            "name": "random_forest_100",
            "model": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "params": {"model_type": "RandomForest", "n_estimators": 100},
        },
        {
            "name": "random_forest_200",
            "model": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state),
            "params": {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10},
        },
        {
            "name": "gradient_boosting",
            "model": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=random_state),
            "params": {"model_type": "GradientBoosting", "n_estimators": 150, "learning_rate": 0.1},
        },
    ]

    best_mae = float("inf")
    best_run_id = None
    best_model = None

    print(f"\nMLflow tracking URI: sqlite:///{mlflow_db}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"data_sha256: {data_sha256}\n")

    for exp in experiments:
        with mlflow.start_run(run_name=exp["name"]) as run:
            # Log data hash in EVERY run (required by handbook)
            mlflow.log_param("data_sha256", data_sha256)

            # Log model hyperparams
            for k, v in exp["params"].items():
                mlflow.log_param(k, v)

            # Train
            model = exp["model"]
            model.fit(X_train, y_train)

            # Evaluate
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_mae = mean_absolute_error(y_train, model.predict(X_train))

            # Log metrics (name must contain 'mae')
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("train_mae", train_mae)

            # Log model artifact
            try:
                mlflow.sklearn.log_model(model, name="model")
            except Exception:
                mlflow.sklearn.log_model(
                    model, name="model",
                    serialization_format="cloudpickle"
                )

            print(f"  Run: {exp['name']:30s}  val_mae={val_mae:.2f}  run_id={run.info.run_id[:8]}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_run_id = run.info.run_id
                best_model = model

    # --- Register the best model ---
    print(f"\nBest run: {best_run_id}  val_mae={best_mae:.2f}")

    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = registered.version

    # Set @champion alias
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Registered model '{MODEL_NAME}' version {version} → @champion")

    # Save champion model to models/ for the FastAPI service
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "model.joblib")
    joblib.dump(encoder, models_dir / "encoder.joblib")
    print(f"Saved model to models/model.joblib")

    # Update findings.yaml with MLflow details
    _update_findings(params, EXPERIMENT_NAME, best_run_id, MODEL_NAME, version)
    print("findings.yaml tracking section updated.")


def _update_findings(
    params: dict,
    experiment_name: str,
    best_run_id: str,
    model_name: str,
    champion_version: str,
) -> None:
    """Write tracking info into reports/findings.yaml."""
    import yaml
    root = project_root()
    findings_path = root / params["paths"]["findings_yaml"]
    with open(findings_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    doc["tracking"] = {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": model_name,
        "champion_version": str(champion_version),
    }

    with open(findings_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    params = get_params()
    run_experiments(params)

