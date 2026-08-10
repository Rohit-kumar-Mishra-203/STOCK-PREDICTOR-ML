"""
Train an XGBoost baseline classifier to predict next-day price direction.

Why XGBoost as the baseline (not LSTM/Transformer first):
Tree-based models are fast to train, hard to overfit accidentally, and
handle tabular features (technical indicators, sentiment scores) very
well without needing careful scaling/normalization. Establishing this
baseline FIRST is deliberate: if a more complex model (LSTM) can't beat
this, the added complexity isn't justified. You can't know that without
a baseline to compare against.

=== WHY THE TRAIN/TEST SPLIT IS CHRONOLOGICAL, NOT RANDOM ===
This is the most important methodological decision in this file.
A random split (e.g. sklearn's train_test_split with shuffle=True)
would let the model train on data from, say, March 2025 and be
"tested" on data from January 2025 - meaning it could indirectly learn
patterns from the future relative to what it's being tested on. Real
trading never works this way: you only ever have the past to predict
the future. A chronological split (everything before date X = train,
everything after = test) is the only honest way to evaluate this.

=== WHY WE COMPARE AGAINST A NAIVE BASELINE ===
"73% accuracy" means nothing in isolation. If simply guessing "tomorrow
repeats today's direction" already gets 55% accuracy, then our model
only really contributed 18 points of genuine skill, not 73. Every
result in this file is reported alongside this naive baseline for
honest context.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.xgboost

from src.utils.config import get_config, resolve_path
from src.utils.logger import logger

# Columns that must NEVER be used as model features - either identifiers,
# raw text, the target itself, or (new) raw absolute price-level columns
# that we've replaced with normalized/relative versions (see
# add_normalized_features in build_dataset.py for why raw price levels
# are problematic for tree-based models).
NON_FEATURE_COLS = [
    "Date", "ticker", "target", "title", "description", "source",
    "Open", "High", "Low", "Close", "Volume",
    "sma_10", "sma_50",
    "bb_upper", "bb_middle", "bb_lower",
    "macd_line", "macd_signal", "macd_histogram",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
]

def load_ticker_dataset(ticker: str) -> pd.DataFrame:
    cfg = get_config()
    filepath = resolve_path(cfg["paths"]["processed"]) / f"{ticker}_final.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"No final dataset found for {ticker} at {filepath}. Run build_dataset.py first.")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def chronological_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split strictly by date - everything before split_date is train,
    everything on/after is test. No shuffling, no randomness.

    This function existing as its own clearly-named, isolated piece
    (rather than an inline sklearn train_test_split call somewhere)
    makes the chronological-not-random decision impossible to miss
    or accidentally undo later.
    """
    split_ts = pd.Timestamp(split_date)
    train = df[df["Date"] < split_ts].copy()
    test = df[df["Date"] >= split_ts].copy()

    logger.info(f"Train: {len(train)} rows ({train['Date'].min().date()} to {train['Date'].max().date()})")
    logger.info(f"Test:  {len(test)} rows ({test['Date'].min().date()} to {test['Date'].max().date()})")

    if len(test) == 0:
        raise ValueError(f"No test rows after split_date {split_date} - check your data range and config.")

    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_naive_baseline(test_df: pd.DataFrame) -> dict:
    """
    The naive baseline: predict tomorrow repeats today's most recent
    known direction (yesterday's actual movement, i.e. persistence).

    Using the same 'daily_return' feature already in our dataset (safe -
    it's backward-looking, see build_dataset.py), a naive "persistence"
    prediction is: predict UP if the most recent daily_return was positive.
    """
    naive_pred = (test_df["daily_return"] > 0).astype(int)
    actual = test_df["target"]

    return {
        "accuracy": accuracy_score(actual, naive_pred),
        "f1": f1_score(actual, naive_pred, zero_division=0),
    }


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with modest, defensible hyperparameters.

    Why these specific settings:
    - n_estimators=200, max_depth=4: shallow trees, moderate ensemble size.
      Financial data is noisy - deep trees overfit to noise very easily.
      Starting shallow is a deliberate anti-overfitting choice, not an
      oversight.
    - subsample/colsample_bytree < 1.0: each tree sees a random subset of
      rows and columns - reduces overfitting further via randomness,
      similar in spirit to how random forests work.
    - eval_metric='logloss': appropriate for binary classification with
      probability outputs, more informative during training than raw
      accuracy.
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,  # reproducibility - same result every run
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    return metrics


def run_for_ticker(ticker: str, split_date: str) -> dict:
    logger.info(f"=== Training baseline for {ticker} ===")

    df = load_ticker_dataset(ticker)
    train_df, test_df = chronological_split(df, split_date)

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = train_xgboost(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    naive = compute_naive_baseline(test_df)

    logger.info(f"{ticker} - Model accuracy: {metrics['accuracy']:.4f} | Naive baseline accuracy: {naive['accuracy']:.4f}")
    logger.info(f"{ticker} - Model F1: {metrics['f1']:.4f} | Naive baseline F1: {naive['f1']:.4f}")

    edge = metrics["accuracy"] - naive["accuracy"]
    logger.info(f"{ticker} - Edge over naive baseline: {edge:+.4f}")

    return {
        "ticker": ticker,
        "model": model,
        "metrics": metrics,
        "naive_baseline": naive,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }


def main():
    cfg = get_config()
    tickers: list[str] = cfg["data"]["tickers"]
    split_date: str = str(cfg["model"]["train_test_split_date"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    all_results = []

    for ticker in tickers:
        with mlflow.start_run(run_name=f"xgboost_baseline_{ticker}"):
            result = run_for_ticker(ticker, split_date)
            all_results.append(result)

            mlflow.log_param("ticker", ticker)
            mlflow.log_param("split_date", split_date)
            mlflow.log_param("n_features", result["n_features"])
            mlflow.log_metrics(result["metrics"])
            mlflow.log_metric("naive_baseline_accuracy", result["naive_baseline"]["accuracy"])
            mlflow.log_metric("edge_over_naive", result["metrics"]["accuracy"] - result["naive_baseline"]["accuracy"])
            mlflow.xgboost.log_model(result["model"], artifact_path="model")

    logger.info("=== Summary across all tickers ===")
    for r in all_results:
        logger.info(f"{r['ticker']}: accuracy={r['metrics']['accuracy']:.4f}, naive={r['naive_baseline']['accuracy']:.4f}")


if __name__ == "__main__":
    main()