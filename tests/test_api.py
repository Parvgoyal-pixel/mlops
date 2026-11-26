"""
test_api.py — TestClient tests for the FastAPI prediction service.

Tests do NOT need CSV files. They use the TestClient against the
running app, which loads model.joblib and encoder.joblib at import time.

Run with:  pytest -q tests/test_api.py

Author: G224 — Yash Mudgal (S24CSEU0246)
"""

import sys
import pathlib
import pytest

# Ensure repo root and src/ are on the path
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── A valid request payload ──────────────────────────────────────────────────

VALID_PAYLOAD = {
    "students_on_campus": 1200,
    "menu_price_rs": 45.0,
    "temp_c": 35.0,
    "exam_days_left": 10,
    "special_dishes": 3,
    "queue_minutes_yesterday": 8.5,
    "day_type": "weekday",
    "block": "A-Block",
}


# ── Health endpoint ──────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ── Predict endpoint — happy path ────────────────────────────────────────────

class TestPredictHappy:
    def test_predict_returns_200(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_has_prediction_key(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "prediction" in data

    def test_prediction_is_number(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(resp.json()["prediction"], (int, float))

    def test_prediction_is_positive(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["prediction"] > 0


# ── Predict endpoint — validation rules ──────────────────────────────────────

class TestPredictValidation:
    def test_temp_c_above_75_rejected(self):
        """Client rule 1: temp_c above 75 → 422."""
        payload = {**VALID_PAYLOAD, "temp_c": 76.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_temp_c_exactly_75_accepted(self):
        """Boundary: temp_c == 75 is valid."""
        payload = {**VALID_PAYLOAD, "temp_c": 75.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_queue_minutes_below_1_rejected(self):
        """Client rule 2: queue_minutes_yesterday below 1 → 422."""
        payload = {**VALID_PAYLOAD, "queue_minutes_yesterday": 0.5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_queue_minutes_exactly_1_accepted(self):
        """Boundary: queue_minutes_yesterday == 1 is valid."""
        payload = {**VALID_PAYLOAD, "queue_minutes_yesterday": 1.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_missing_field_rejected(self):
        """Missing required field → 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "block"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422
