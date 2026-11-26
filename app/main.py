"""
main.py — FastAPI prediction service for MP126 College Canteen Lunch Rush.

Endpoints:
  GET  /health   → {"status": "ok"}
  POST /predict  → {"prediction": <float>}

Pydantic validation rules (from PROBLEM.md section 5):
  - temp_c must be ≤ 75  (above 75 is impossible)
  - queue_minutes_yesterday must be ≥ 1  (below 1 is a typo)

JSON structured logging: one JSON line per prediction on stdout.

Author: G224 — Ekta Mishra (S24CSEU1362)
"""

from __future__ import annotations

import logging
import json
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── JSON structured logging ──────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record on a single line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)


logger = logging.getLogger("mp126_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# ── Load model artefacts at startup ──────────────────────────────────────────

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODEL = joblib.load(_ROOT / "models" / "model.joblib")
_ENCODER = joblib.load(_ROOT / "models" / "encoder.joblib")

# Feature order must match training (see src/mp126/train.py)
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
CAT_COLS = ["day_type", "block"]

# ── Pydantic request schema ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    students_on_campus: int
    menu_price_rs: float
    temp_c: float = Field(..., le=75, description="Temperature at noon (°C). Max 75.")
    exam_days_left: int
    special_dishes: int
    queue_minutes_yesterday: float = Field(
        ..., ge=1, description="Avg queue time yesterday (min). Min 1."
    )
    day_type: str
    block: str

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="MP126 Canteen Prediction API", version="1.0")


@app.get("/health")
def health():
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Return predicted plates_sold for the given input."""
    import pandas as pd

    # Build a single-row DataFrame with the same column order as training
    row = {
        "students_on_campus": req.students_on_campus,
        "menu_price_rs": req.menu_price_rs,
        "temp_c": req.temp_c,
        "exam_days_left": req.exam_days_left,
        "special_dishes": req.special_dishes,
        "queue_minutes_yesterday": req.queue_minutes_yesterday,
        "day_type": req.day_type,
        "block": req.block,
    }
    df = pd.DataFrame([row])

    # Encode categoricals with the fitted encoder
    df[CAT_COLS] = _ENCODER.transform(df[CAT_COLS])

    # Predict
    X = df[FEATURES].values
    prediction = float(_MODEL.predict(X)[0])

    # Structured JSON log (never log personal details)
    logger.info(
        "prediction",
        extra={
            "extra": {
                "event": "prediction",
                "input": {k: v for k, v in row.items()},
                "prediction": prediction,
            }
        },
    )

    return {"prediction": prediction}
