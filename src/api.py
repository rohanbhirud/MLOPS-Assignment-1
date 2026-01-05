"""
FastAPI inference service for the Heart Disease model.

Usage (with existing trained model in models/log_reg_pipeline.joblib):
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_PATH: path to a joblib file containing the trained pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, conlist
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pandas as pd

# Feature ordering expected by the model
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/log_reg_pipeline.joblib")


class HeartRecord(BaseModel):
    # Use loose typing (float) to allow ints or floats in the payload.
    values: conlist(float, min_length=len(FEATURE_COLUMNS), max_length=len(FEATURE_COLUMNS)) = Field(
        ..., description=f"Feature values ordered as {FEATURE_COLUMNS}"
    )


class PredictRequest(BaseModel):
    records: List[HeartRecord]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path.resolve()}")
    return joblib.load(path)


def preprocess_payload(payload: PredictRequest) -> pd.DataFrame:
    # Convert to DataFrame aligned with FEATURE_COLUMNS
    data = [rec.values for rec in payload.records]
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Basic request/response logging
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency per endpoint",
    ["endpoint"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(duration)
    logger.info(
        "Handled %s %s -> %s in %.3fs",
        request.method,
        endpoint,
        response.status_code,
        duration,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    X = preprocess_payload(payload)
    preds = model.predict(X)
    # Use predict_proba to get probability of positive class (index 1).
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = [0.0] * len(preds)
    return PredictResponse(predictions=preds.tolist(), probabilities=[float(p) for p in probs])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
