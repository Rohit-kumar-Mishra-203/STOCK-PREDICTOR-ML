"""
FastAPI serving layer for the trained stock direction prediction models.

Design boundary worth being explicit about (see project README for full
discussion): this API serves predictions using ALREADY-COMPUTED features
for a given date (either passed directly, or pulled from our last
processed dataset run). It does not perform live feature computation
from a fresh market data pull on every request - that would require a
scheduled data-refresh job (see paths.md/README for how this would be
extended with Airflow/cron in a fuller production deployment). This is
a deliberate, stated scope boundary, not an oversight.

Why models load once at startup (not per-request): loading a model from
disk is I/O - repeating that on every request would add unnecessary
latency and disk load under any real traffic. Loading once into memory
at startup and reusing across requests is standard practice.
"""

from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger

# Populated at startup, read during requests - see lifespan() below
MODELS: dict = {}
FEATURE_COLS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook - runs once when the server starts (and once
    more on shutdown, though we don't need cleanup logic here).

    Why lifespan over the older @app.on_event("startup") decorator:
    that pattern is deprecated in current FastAPI versions - lifespan
    is the recommended, modern approach for startup/shutdown logic.
    """
    cfg = get_config()
    model_dir = resolve_path(cfg["paths"]["models"]) / "production"
    tickers: list[str] = cfg["data"]["tickers"]

    for ticker in tickers:
        model_path = model_dir / f"{ticker}_xgboost.joblib"
        features_path = model_dir / f"{ticker}_features.joblib"

        if not model_path.exists():
            logger.warning(f"No production model found for {ticker} at {model_path} - skipping")
            continue

        MODELS[ticker] = joblib.load(model_path)
        FEATURE_COLS[ticker] = joblib.load(features_path)
        logger.info(f"Loaded production model for {ticker}")

    logger.info(f"API ready. Models loaded for: {list(MODELS.keys())}")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Stock Movement Prediction API",
    description=(
        "Serves next-day price direction predictions from a trained "
        "XGBoost model. See README for the honest scope/limitations of "
        "this prediction task (see the project's evaluation section)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """
    Request contract for a prediction.

    Why 'features' is a dict rather than fixed named fields: the exact
    feature set is defined by the training pipeline (see build_dataset.py)
    and could evolve over time. A dict keeps the API contract flexible to
    that, while /model-info exposes the exact expected keys for a given
    ticker so callers aren't guessing.
    """
    ticker: str = Field(..., examples=["AAPL"])
    features: dict[str, float] = Field(
        ..., description="Feature values matching the model's expected feature set. See /model-info/{ticker}."
    )


class PredictionResponse(BaseModel):
    ticker: str
    predicted_direction: str  # "up" or "down"
    probability_up: float
    model_version: str = "xgboost_baseline_v1"


@app.get("/")
def root():
    return {
        "service": "Stock Movement Prediction API",
        "status": "running",
        "available_tickers": list(MODELS.keys()),
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """
    Basic health/readiness check - the kind of endpoint a real deployment
    (Docker healthcheck, load balancer, uptime monitor) would poll.
    """
    return {
        "status": "healthy" if MODELS else "degraded",
        "models_loaded": list(MODELS.keys()),
    }


@app.get("/model-info/{ticker}")
def model_info(ticker: str):
    """
    Exposes exactly which features a given ticker's model expects, and
    in what order - lets an API consumer construct a valid request
    without needing to read the training code.
    """
    ticker = ticker.upper()
    if ticker not in FEATURE_COLS:
        raise HTTPException(status_code=404, detail=f"No model found for ticker '{ticker}'")

    return {
        "ticker": ticker,
        "expected_features": FEATURE_COLS[ticker],
        "n_features": len(FEATURE_COLS[ticker]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict next-day price direction given a feature vector.

    Validates that the request's features exactly match what the model
    was trained on (same set, right order enforced via DataFrame column
    selection) - failing loudly with a clear error rather than silently
    passing mismatched/missing features into the model, which would
    produce a confident-looking but meaningless prediction.
    """
    ticker = request.ticker.upper()

    if ticker not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"No model available for ticker '{ticker}'. Available: {list(MODELS.keys())}",
        )

    expected_features = FEATURE_COLS[ticker]
    missing = set(expected_features) - set(request.features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features for {ticker}: {sorted(missing)}",
        )

    # Explicit column order via DataFrame construction from a dict, then
    # column selection matching the training-time order exactly - this is
    # the runtime enforcement of the 'same columns, same order' contract
    # described in save_model_for_serving() in train_baseline.py.
    input_df = pd.DataFrame([request.features])[expected_features]

    model = MODELS[ticker]
    proba = model.predict_proba(input_df)[0]
    prob_up = float(proba[1])  # class 1 = "up", per build_target() in build_dataset.py

    return PredictionResponse(
        ticker=ticker,
        predicted_direction="up" if prob_up > 0.5 else "down",
        probability_up=round(prob_up, 4),
    )


@app.get("/predict-latest/{ticker}", response_model=PredictionResponse)
def predict_latest(ticker: str):
    """
    Convenience endpoint: predicts using the most recent row already
    present in our processed dataset (data/processed/{ticker}_final.csv),
    rather than requiring the caller to supply features manually.

    Honest limitation stated here directly: this uses whatever the last
    pipeline run produced - it is only as fresh as the last time
    fetch_prices.py / clean_data.py / build_dataset.py were run. A fully
    live system would trigger that pipeline on a schedule (see README).
    """
    ticker = ticker.upper()

    if ticker not in MODELS:
        raise HTTPException(status_code=404, detail=f"No model available for ticker '{ticker}'")

    cfg = get_config()
    data_path = resolve_path(cfg["paths"]["processed"]) / f"{ticker}_final.csv"

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"No processed data found for {ticker}. Run the data pipeline first.")

    df = pd.read_csv(data_path)
    latest_row = df.iloc[-1]

    expected_features = FEATURE_COLS[ticker]
    features = {col: float(latest_row[col]) for col in expected_features}

    input_df = pd.DataFrame([features])[expected_features]
    model = MODELS[ticker]
    proba = model.predict_proba(input_df)[0]
    prob_up = float(proba[1])

    return PredictionResponse(
        ticker=ticker,
        predicted_direction="up" if prob_up > 0.5 else "down",
        probability_up=round(prob_up, 4),
    )